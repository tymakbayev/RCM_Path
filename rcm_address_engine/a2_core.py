"""A2 core (no PostgreSQL): key hash, variant A cascade (JLMADR01), variant B buckets (JLMHPL01), SORT, EF size.

Protocol: A2_protocol.md §4 (edition 2, sha256 in results/a2_protocol.sha256).
"""

from __future__ import annotations

import bisect
import hashlib
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
BLOCK_BITS = 480  # bitmap bits per 64-byte block (u32 rank + 15 u32 words)
F_BITS = 8
GAMMA = 2.0
MAX_LEVELS = 64
U64 = np.uint64


# ------------------------------------------------------------------ hash
def splitmix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=U64)
    with np.errstate(over="ignore"):
        z = x + U64(GOLD)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def splitmix64_int(x: int) -> int:
    z = (x + GOLD) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def key_hash(keys: np.ndarray, seed: int) -> np.ndarray:
    """keys uint8 [n, 16] -> uint64 [n]: h = splitmix64(lo ^ splitmix64(hi ^ seed)), lo/hi little-endian u64."""
    words = np.ascontiguousarray(keys).view("<u8").reshape(-1, 2)
    return splitmix64(words[:, 0] ^ splitmix64(words[:, 1] ^ U64(seed)))


def key_hash_int(key: bytes, seed: int) -> int:
    lo, hi = struct.unpack("<QQ", key)
    return splitmix64_int(lo ^ splitmix64_int(hi ^ seed))


def level_pos(h: np.ndarray, level: int, m: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = splitmix64(h + U64(((level + 1) * GOLD) & M64))
    return ((x >> U64(32)) * U64(m)) >> U64(32)


def level_pos_int(h: int, level: int, m: int) -> int:
    return ((splitmix64_int((h + (level + 1) * GOLD) & M64) >> 32) * m) >> 32


# ------------------------------------------------------------------ bit-packed D
def pack_fixed(values: np.ndarray, width: int) -> np.ndarray:
    """values < 2^width (width <= 57) packed at bit s*width; one spare word so a reader may always load word+1."""
    n = len(values)
    words = np.zeros((n * width + 63) // 64 + 1, dtype=U64)
    bit = np.arange(n, dtype=U64) * U64(width)
    lo, off = (bit >> U64(6)).astype(np.int64), bit & U64(63)
    v = values.astype(U64)
    with np.errstate(over="ignore"):
        np.bitwise_or.at(words, lo, v << off)
        spill = off + U64(width) > U64(64)
        np.bitwise_or.at(words, lo[spill] + 1, v[spill] >> (U64(64) - off[spill]))
    return words


def unpack_fixed(words: np.ndarray, idx: np.ndarray, width: int) -> np.ndarray:
    bit = idx.astype(U64) * U64(width)
    lo, off = (bit >> U64(6)).astype(np.int64), bit & U64(63)
    with np.errstate(over="ignore"):
        low = words[lo] >> off
        high = np.where(off == 0, U64(0), words[lo + 1] << ((U64(64) - off) & U64(63)))
    return (low | high) & U64((1 << width) - 1)


def unpack_fixed_int(words, i: int, width: int) -> int:
    bit = i * width
    lo, off = bit >> 6, bit & 63
    v = int(words[lo]) >> off
    if off + width > 64:
        v |= int(words[lo + 1]) << (64 - off)
    return v & ((1 << width) - 1)


# ------------------------------------------------------------------ variant A: build
@dataclass
class CascadeA:
    seed: int
    r_bits: int
    level_m: list[int]
    level_off: list[int]  # bit offset of each level in the global bitmap (multiple of BLOCK_BITS)
    blocks: np.ndarray  # uint32 [n_blocks, 16]: word 0 = rank before the block, words 1..15 = bits
    d_words: np.ndarray  # uint64 packed (fingerprint | page << F_BITS)
    n_slots: int
    exc_keys: list[bytes]  # sorted
    exc_pages: list[int]


def build_a(keys: np.ndarray, pages: np.ndarray, seed: int, r_bits: int, gamma: float = GAMMA) -> CascadeA:
    n = len(keys)
    h = key_hash(keys, seed)
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    dup = cnt[inv] > 1  # identical full hash: the cascade can never separate them -> exceptions with the full key
    remaining = np.flatnonzero(~dup)
    level_of = np.full(n, -1, np.int16)
    pos_of = np.zeros(n, np.int64)
    level_m: list[int] = []
    bitmaps: list[np.ndarray] = []
    for level in range(MAX_LEVELS):
        if not len(remaining):
            break
        m = max(64, math.ceil(gamma * len(remaining)))
        pos = level_pos(h[remaining], level, m).astype(np.int64)
        alone = np.bincount(pos, minlength=m)[pos] == 1
        bits = np.zeros(m, bool)
        bits[pos[alone]] = True
        level_m.append(m)
        bitmaps.append(bits)
        level_of[remaining[alone]] = level
        pos_of[remaining[alone]] = pos[alone]
        remaining = remaining[~alone]
    exc_idx = np.concatenate([np.flatnonzero(dup), remaining])
    level_off, total = [], 0
    for m in level_m:
        level_off.append(total)
        total += -(-m // BLOCK_BITS) * BLOCK_BITS
    flat = np.zeros(total, bool)
    for off, bits in zip(level_off, bitmaps):
        flat[off: off + len(bits)] = bits
    n_blocks = total // BLOCK_BITS
    packed = np.packbits(flat.reshape(n_blocks, BLOCK_BITS), axis=1, bitorder="little")  # [n_blocks, 60] bytes
    blocks = np.zeros((n_blocks, 16), np.uint32)
    blocks[:, 1:] = packed.view("<u4")
    ones = flat.reshape(n_blocks, BLOCK_BITS).sum(axis=1)
    blocks[:, 0] = np.concatenate([[0], np.cumsum(ones)[:-1]]).astype(np.uint32)
    placed = np.flatnonzero(level_of >= 0)
    gbit = np.asarray(level_off, np.int64)[level_of[placed]] + pos_of[placed]
    rank_all = np.cumsum(flat) - 1  # global rank of every set bit
    slots = rank_all[gbit]
    n_slots = int(flat.sum())
    if r_bits + F_BITS > 57:
        raise ValueError("descriptor too wide")
    values = np.zeros(n_slots, U64)
    fp = h[placed] & U64((1 << F_BITS) - 1)
    values[slots] = fp | (pages[placed].astype(U64) << U64(F_BITS))
    d_words = pack_fixed(values, F_BITS + r_bits)
    order = sorted(range(len(exc_idx)), key=lambda i: bytes(keys[exc_idx[i]]))
    exc_keys = [bytes(keys[exc_idx[i]]) for i in order]
    exc_pages = [int(pages[exc_idx[i]]) for i in order]
    return CascadeA(seed, r_bits, level_m, level_off, blocks, d_words, n_slots, exc_keys, exc_pages)


# ------------------------------------------------------------------ JLMADR01 file
MAGIC_A = b"JLMADR01"
HEAD_A = struct.Struct("<8sHHIIIIQQqQdBBBBIQQQQQQQQ")  # fields below, then u32 crc
HEAD_A_SIZE = 192


def _align(n: int, a: int = 64) -> int:
    return -(-n // a) * a


def write_a(path: Path, ca: CascadeA, *, table_oid: int, relfilenode: int, generation: int, n_rows: int, n_pages: int,
            watermark: int) -> dict:
    n_levels = len(ca.level_m)
    lev = np.array([[m, off] for m, off in zip(ca.level_m, ca.level_off)], dtype="<u8").reshape(-1, 2)
    exc = b"".join(k + struct.pack("<I", p) for k, p in zip(ca.exc_keys, ca.exc_pages))
    off_levels = HEAD_A_SIZE
    off_blocks = _align(off_levels + lev.nbytes)
    off_d = _align(off_blocks + ca.blocks.nbytes)
    off_exc = _align(off_d + ca.d_words.nbytes)
    file_len = off_exc + len(exc) + 32
    head = HEAD_A.pack(MAGIC_A, 1, 0, table_oid, relfilenode, generation, 0, n_rows, n_pages, watermark, ca.seed, GAMMA,
                       F_BITS, ca.r_bits, n_levels, 0, len(ca.exc_keys), len(ca.blocks), ca.n_slots, off_levels,
                       off_blocks, off_d, off_exc, file_len, 0)
    head = head + struct.pack("<I", zlib.crc32(head) & 0xFFFFFFFF)
    body = bytearray(off_exc + len(exc))
    body[: len(head)] = head
    body[off_levels: off_levels + lev.nbytes] = lev.tobytes()
    body[off_blocks: off_blocks + ca.blocks.nbytes] = ca.blocks.astype("<u4").tobytes()
    body[off_d: off_d + ca.d_words.nbytes] = ca.d_words.astype("<u8").tobytes()
    body[off_exc:] = exc
    digest = hashlib.blake2b(bytes(body), digest_size=32).digest()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(body)
        f.write(digest)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"bytes": file_len, "sha256": hashlib.sha256(bytes(body) + digest).hexdigest(), "levels": n_levels,
            "exceptions": len(ca.exc_keys), "blocks": len(ca.blocks), "d_bytes": int(ca.d_words.nbytes)}


class SidecarA:
    """Read-only, mmap-backed variant A address. Opening is O(1) in N: header parse + array views (+ exception list)."""

    def __init__(self, path: Path, *, count_pages: bool = False) -> None:
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        raw = bytes(self.mm[: HEAD_A.size + 4])
        fields = HEAD_A.unpack(raw[: HEAD_A.size])
        (magic, version, _flags, self.table_oid, self.relfilenode, self.generation, _pad, self.n_rows, self.n_pages,
         self.watermark, self.seed, _gamma, f_bits, self.r_bits, n_levels, _p2, n_exc, n_blocks, self.n_slots,
         off_levels, off_blocks, off_d, off_exc, file_len, _z) = fields
        (crc,) = struct.unpack("<I", raw[HEAD_A.size:])
        if magic != MAGIC_A or version != 1 or f_bits != F_BITS:
            raise ValueError("not a JLMADR01 v1 file")
        if zlib.crc32(raw[: HEAD_A.size]) & 0xFFFFFFFF != crc:
            raise ValueError("header crc mismatch")
        if file_len != len(self.mm):
            raise ValueError("file length mismatch")
        self.levels = self.mm[off_levels: off_levels + 16 * n_levels].view("<u8").reshape(-1, 2)
        self.lev = [(int(m), int(o)) for m, o in self.levels]  # n_levels <= 64 pairs: O(1) in N
        self.blocks = self.mm[off_blocks: off_blocks + 64 * n_blocks].view("<u4").reshape(-1, 16)
        self.d = self.mm[off_d: off_exc].view("<u8") if off_exc > off_d else np.zeros(2, U64)
        self.width = F_BITS + self.r_bits
        exc = bytes(self.mm[off_exc: off_exc + 20 * n_exc])
        self.exc_keys = [exc[i * 20: i * 20 + 16] for i in range(n_exc)]
        self.exc_pages = [struct.unpack_from("<I", exc, i * 20 + 16)[0] for i in range(n_exc)]
        self.offsets = {"levels": off_levels, "blocks": off_blocks, "d": off_d, "exc": off_exc}
        self.count_pages = count_pages
        self.touched: set[int] = set()

    def _touch(self, byte_offset: int, n: int = 1) -> None:
        if self.count_pages:
            for p in range(byte_offset >> 12, ((byte_offset + n - 1) >> 12) + 1):
                self.touched.add(p)

    def page_of(self, key: bytes) -> int | None:
        """Heap page of `key`, or None when the address says the key is not in it."""
        h = key_hash_int(key, self.seed)
        self._touch(0, HEAD_A_SIZE)
        if self.exc_keys:
            self._touch(self.offsets["exc"], 20 * len(self.exc_keys))
            i = bisect.bisect_left(self.exc_keys, key)
            if i < len(self.exc_keys) and self.exc_keys[i] == key:
                return self.exc_pages[i]
        for level, (m, off) in enumerate(self.lev):
            self._touch(self.offsets["levels"] + 16 * level, 16)
            g = off + level_pos_int(h, level, m)
            b, i = divmod(g, BLOCK_BITS)
            self._touch(self.offsets["blocks"] + 64 * b, 64)
            row = self.blocks[b]
            w = int(row[1 + (i >> 5)])
            if (w >> (i & 31)) & 1:
                rank = int(row[0]) + sum(int(row[1 + j]).bit_count() for j in range(i >> 5))
                rank += (w & ((1 << (i & 31)) - 1)).bit_count()
                self._touch(self.offsets["d"] + 8 * ((rank * self.width) >> 6), 16)
                v = unpack_fixed_int(self.d, rank, self.width)
                return (v >> F_BITS) if (v & 0xFF) == (h & 0xFF) else None
        return None

    def pages_of(self, keys: np.ndarray) -> np.ndarray:
        """Vectorised page_of: int64 page or -1."""
        n = len(keys)
        h = key_hash(keys, self.seed)
        out = np.full(n, -1, np.int64)
        todo = np.arange(n)
        blocks = np.asarray(self.blocks)
        for level, (m, off) in enumerate(self.lev):
            if not len(todo):
                break
            g = off + level_pos(h[todo], level, m).astype(np.int64)
            b, i = g // BLOCK_BITS, g % BLOCK_BITS
            word = blocks[b, 1 + (i >> 5)].astype(np.uint64)
            hit = ((word >> (i & 31).astype(np.uint64)) & U64(1)) == 1
            hb, hi_ = b[hit], i[hit]
            rows = blocks[hb, 1:].astype(np.uint64)  # [k, 15]
            full = np.arange(15)[None, :] < (hi_ >> 5)[:, None]
            cnt = np.bitwise_count(rows) * full
            part = np.bitwise_count(rows[np.arange(len(hb)), hi_ >> 5] & ((U64(1) << (hi_ & 31).astype(np.uint64)) - U64(1)))
            rank = blocks[hb, 0].astype(np.int64) + cnt.sum(axis=1) + part.astype(np.int64)
            v = unpack_fixed(np.asarray(self.d), rank, self.width)
            ok = (v & U64(0xFF)) == (h[todo[hit]] & U64(0xFF))
            out[todo[hit][ok]] = (v[ok] >> U64(F_BITS)).astype(np.int64)
            todo = todo[~hit]
        if self.exc_keys:
            for j in range(n):
                kb = bytes(keys[j])
                i = bisect.bisect_left(self.exc_keys, kb)
                if i < len(self.exc_keys) and self.exc_keys[i] == kb:
                    out[j] = self.exc_pages[i]
        return out

    def verify(self) -> bool:
        data = bytes(self.mm)
        return hashlib.blake2b(data[:-32], digest_size=32).digest() == data[-32:]

    def close(self) -> None:
        mm = getattr(self.mm, "_mmap", None)
        self.blocks = self.d = self.levels = None
        if mm is not None:
            mm.close()


# ------------------------------------------------------------------ variant B
def bucket_bits(n_rows: int, rows_per_page: float) -> int:
    return int(min(32, max(1, round(math.log2(max(n_rows / rows_per_page, 2.0))))))


def buckets_of(keys: np.ndarray, seed: int, b: int) -> np.ndarray:
    return (key_hash(keys, seed) >> U64(64 - b)).astype(np.int64)


def bucket_starts(bucket: np.ndarray, pages: np.ndarray, b: int) -> np.ndarray:
    """u32 [2^b + 1]: start_j = first page of bucket j; empty -> next non-empty start; last = last page of the table."""
    nb = 1 << b
    first = np.full(nb + 1, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, bucket, pages.astype(np.int64))
    first[nb] = int(pages.max())
    valid = first != np.iinfo(np.int64).max
    # empty bucket -> start of the NEXT non-empty bucket (not a suffix minimum: that could hide a placement violation)
    nxt = np.minimum.accumulate(np.where(valid, np.arange(nb + 1), nb)[::-1])[::-1]
    return first[nxt].astype(np.uint32)


def bucket_layout(bucket: np.ndarray, pages: np.ndarray, seq: np.ndarray, b: int):
    """Addendum 2026-11-16 §3: rows in load order (bucket, seq); a row whose page is below the maximum page of the rows
    loaded before it went back (FSM reuse) -> exception. starts come from the in-order rows only.
    Returns (starts u32[2^b+1], misplaced row indices)."""
    order = np.lexsort((seq, bucket))
    p = pages.astype(np.int64)[order]
    prev_max = np.concatenate([[-1], np.maximum.accumulate(p)[:-1]])
    back = p < prev_max
    keep = order[~back]
    nb = 1 << b
    big = np.iinfo(np.int64).max
    first = np.full(nb + 1, big, np.int64)
    np.minimum.at(first, bucket[keep], pages.astype(np.int64)[keep])
    first[nb] = int(pages.max())
    valid = first != big
    nxt = np.minimum.accumulate(np.where(valid, np.arange(nb + 1), nb)[::-1])[::-1]
    return first[nxt].astype(np.uint32), np.sort(order[back])


def bucket_invariant(bucket: np.ndarray, pages: np.ndarray, starts: np.ndarray, exc_rows: np.ndarray | None = None) -> int:
    """Rows that are neither exceptions nor inside [start_j, start_{j+1}] (protocol §4.3 with addendum §3)."""
    p = pages.astype(np.int64)
    out = (p < starts[bucket]) | (p > starts[bucket + 1])
    if exc_rows is not None and len(exc_rows):
        out[exc_rows] = False
    return int(out.sum())


MAGIC_B = b"JLMHPL01"
HEAD_B = struct.Struct("<8sHHIIIIQQqQBBHII")  # ..., n_starts, n_exceptions


def write_b(path: Path, starts: np.ndarray, *, seed: int, b: int, table_oid: int, relfilenode: int, generation: int,
            n_rows: int, n_pages: int, watermark: int, exc_keys: list[bytes] = (), exc_pages: list[int] = ()) -> dict:
    order = sorted(range(len(exc_keys)), key=lambda i: exc_keys[i])
    exc = b"".join(exc_keys[i] + struct.pack("<I", exc_pages[i]) for i in order)
    head = HEAD_B.pack(MAGIC_B, 1, 0, table_oid, relfilenode, generation, 0, n_rows, n_pages, watermark, seed, b, 0, 0,
                       len(starts), len(exc_keys))
    head += struct.pack("<I", zlib.crc32(head) & 0xFFFFFFFF)
    body = head + bytes(_align(len(head)) - len(head)) + starts.astype("<u4").tobytes() + exc
    digest = hashlib.blake2b(body, digest_size=32).digest()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(body + digest)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"bytes": len(body) + 32, "sha256": hashlib.sha256(body + digest).hexdigest(), "exceptions": len(exc_keys)}


class SidecarB:
    def __init__(self, path: Path, *, skip_exceptions: bool = False) -> None:
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        raw = bytes(self.mm[: HEAD_B.size + 4])
        (magic, version, _f, self.table_oid, self.relfilenode, self.generation, _p, self.n_rows, self.n_pages,
         self.watermark, self.seed, self.b, _p2, _p3, n, n_exc) = HEAD_B.unpack(raw[: HEAD_B.size])
        (crc,) = struct.unpack("<I", raw[HEAD_B.size:])
        if magic != MAGIC_B or version != 1 or zlib.crc32(raw[: HEAD_B.size]) & 0xFFFFFFFF != crc:
            raise ValueError("bad JLMHPL01 header")
        off = _align(HEAD_B.size + 4)
        self.starts = self.mm[off: off + 4 * n].view("<u4")
        exc = bytes(self.mm[off + 4 * n: off + 4 * n + 20 * n_exc])
        self.exc_keys = [] if skip_exceptions else [exc[i * 20: i * 20 + 16] for i in range(n_exc)]
        self.exc_pages = [struct.unpack_from("<I", exc, i * 20 + 16)[0] for i in range(n_exc)]
        self.n_exc = n_exc
        self.off_starts = off

    def touched_pages(self, key: bytes) -> int:
        """4-KiB pages of this file one lookup reads: header, exception list (if any), the two start entries."""
        j = key_hash_int(key, self.seed) >> (64 - self.b)
        pages = {0, (self.off_starts + 4 * j) >> 12, (self.off_starts + 4 * j + 7) >> 12}
        if self.exc_keys:
            first = self.off_starts + 4 * len(self.starts)
            pages |= set(range(first >> 12, (first + 20 * self.n_exc - 1) // 4096 + 1))
        return len(pages)

    def range_of(self, key: bytes) -> tuple[int, int]:
        if self.exc_keys:
            i = bisect.bisect_left(self.exc_keys, key)
            if i < len(self.exc_keys) and self.exc_keys[i] == key:
                return self.exc_pages[i], self.exc_pages[i]
        j = key_hash_int(key, self.seed) >> (64 - self.b)
        return int(self.starts[j]), int(self.starts[j + 1])

    def close(self) -> None:
        mm = getattr(self.mm, "_mmap", None)
        self.starts = None
        if mm is not None:
            mm.close()


# ------------------------------------------------------------------ SORT and EF
def build_sort(keys: np.ndarray, pages: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = (key_hash(keys, seed) >> U64(32)).astype(np.uint32)
    order = np.argsort(prefix, kind="stable")
    return prefix[order], pages[order].astype(np.uint32)


def sort_c1(keys: np.ndarray, pages: np.ndarray, seed: int, prefix: np.ndarray, spages: np.ndarray) -> int:
    """Rows whose own page is missing from their equal-prefix range (exhaustive C1 for SORT)."""
    q = (key_hash(keys, seed) >> U64(32)).astype(np.uint32)
    lo, hi = np.searchsorted(prefix, q, "left"), np.searchsorted(prefix, q, "right")
    single = hi - lo == 1
    bad = int((spages[lo[single]] != pages[single]).sum())
    for i in np.flatnonzero(~single):
        if pages[i] not in set(spages[lo[i]: hi[i]].tolist()):
            bad += 1
    return bad


def ef_bytes(keys: np.ndarray, seed: int, r_bits: int) -> float:
    """Exact Elias-Fano size of the sorted 32-bit hash prefixes plus r page bits per key, bytes per key."""
    prefix = np.sort((key_hash(keys, seed) >> U64(32)).astype(np.uint64))
    n, universe = len(prefix), 1 << 32
    low = max(0, int(math.floor(math.log2(universe / n))))
    high_bits = n + (int(prefix[-1]) >> low) + 1
    return (n * low + high_bits + n * r_bits) / 8 / n
