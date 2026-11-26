"""A5: stepwise major compaction over a hash-sectioned address (protocol A5_protocol.md edition 2).

Levels, newest first: tail (JLMDLOG1 from a4_lsm, unchanged) -> runs (priority order) -> base. Every address file is
split into sections by the top bits of h = key_hash(k, SEED_SECT); a lookup probes ONE JLMADR01 per level. All
maintenance is done in bounded steps between writer batches:
  * seal of the tail (as A4),
  * a tier merge, one output section per step (§2.2), output priority = max input priority, inputs pinned,
  * a round (§2.3): forced seal + W, S_base scan steps over page slices into per-section work files, S_base build steps
    publishing one base section each, pre-W runs dropped at the end.
Correctness rests on PostgreSQL as in A4: every page is a candidate, the row is accepted after `k = $1` + visibility.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

import a2_core as C
from a4_lsm import FANOUT, LOGREC, RefuseOpen, TailLog, r_bits_needed  # noqa: F401  (RefuseOpen re-exported)

SEED_SECT = 0xA5A5_5EC7
SEED_RUN = 0xA5A4_0000
SEED_BASE = 0xA5B4_0000
SIGMA = 1 << 17
LOGDT = np.dtype([("k", "V16"), ("p", "<u4")])
U64 = np.uint64
MANIFEST = "a5.manifest.json"


def bits_for(n: int, sigma: int) -> int:
    """log2 S with S = 2^ceil(log2(n / sigma)), S >= 1 (§2.1)."""
    return max(0, math.ceil(math.log2(n / sigma))) if n > sigma else 0


def sect_np(h: np.ndarray, bits: int) -> np.ndarray:
    return np.right_shift(h, U64(64 - bits)).astype(np.int64) if bits else np.zeros(len(h), np.int64)


def sect_int(h: int, bits: int) -> int:
    return h >> (64 - bits) if bits else 0


def kwords(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Big-endian words of the 16-byte key: lexicographic key order == (w0, w1) order."""
    w = np.ascontiguousarray(keys).view(">u8").reshape(-1, 2)
    return w[:, 0].astype(U64), w[:, 1].astype(U64)


def as_keys(raw: np.ndarray) -> np.ndarray:
    return np.frombuffer(raw.tobytes(), np.uint8).reshape(-1, 16) if len(raw) else np.zeros((0, 16), np.uint8)


def write_durable(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_adr(path: Path, keys: np.ndarray, pages: np.ndarray, seed: int, generation: int) -> tuple[C.SidecarA | None, int]:
    """One JLMADR01 section; an empty section has no file (None, 0)."""
    n = len(keys)
    if n == 0:
        return None, 0
    ca = C.build_a(keys, pages, seed, r_bits_needed(int(pages.max()) + 1))
    info = C.write_a(path, ca, table_oid=0, relfilenode=0, generation=generation, n_rows=n,
                     n_pages=int(pages.max()) + 1, watermark=0)
    return C.SidecarA(path), int(info["bytes"])


class Hasher:
    """h = key_hash(k, SEED_SECT); `trunc` keeps only the top bits (scenario R4: artificial collisions of h)."""

    def __init__(self, trunc: int | None = None) -> None:
        self.mask = ((1 << trunc) - 1) << (64 - trunc) if trunc else (1 << 64) - 1

    def np(self, keys: np.ndarray) -> np.ndarray:
        return C.key_hash(keys, SEED_SECT) & U64(self.mask)

    def one(self, key: bytes) -> int:
        return C.key_hash_int(key, SEED_SECT) & self.mask


# ------------------------------------------------------------------ runs
class Run:
    """Immutable run: log (key 16 | page u32) sorted by (h, k) + one JLMADR01 per section of the log."""

    def __init__(self, d: Path, rid: int, prio: int, tier: int, bits: int, offsets: list[int], pre: bool = False) -> None:
        self.dir, self.id, self.prio, self.tier, self.bits, self.offsets, self.pre = d, rid, prio, tier, bits, offsets, pre
        self.addr: list[C.SidecarA | None] = []
        self._arr = None

    @property
    def log_path(self) -> Path:
        return self.dir / f"r{self.id:06d}.log"

    def adr_path(self, j: int) -> Path:
        return self.dir / f"r{self.id:06d}_s{j:04d}.adr"

    @property
    def n(self) -> int:
        return self.offsets[-1]

    def open(self) -> None:
        self.addr = [C.SidecarA(self.adr_path(j)) if self.offsets[j + 1] > self.offsets[j] else None
                     for j in range(1 << self.bits)]

    def meta(self) -> list:
        return [self.id, self.prio, self.tier, self.bits, self.offsets, self.pre]

    @classmethod
    def from_meta(cls, d: Path, m: list) -> "Run":
        r = cls(d, *m)
        r.open()
        return r

    @classmethod
    def build(cls, d: Path, rid: int, prio: int, tier: int, keys: np.ndarray, pages: np.ndarray, hz: Hasher,
              sigma: int) -> tuple["Run", dict]:
        """Seal: a run from unique keys (any order)."""
        t0 = time.perf_counter()
        h = hz.np(keys)
        w0, w1 = kwords(keys)
        order = np.lexsort((w1, w0, h))
        keys, pages, h = keys[order], pages[order], h[order]
        bits = bits_for(len(keys), sigma)
        bounds = [int(x) for x in np.searchsorted(sect_np(h, bits), np.arange((1 << bits) + 1))] if len(keys) else [0] * ((1 << bits) + 1)
        r = cls(d, rid, prio, tier, bits, bounds)
        log = np.zeros(len(keys), LOGDT)
        if len(keys):
            log["k"] = keys.view("V16").ravel()
            log["p"] = pages
        write_durable(r.log_path, log.tobytes())
        adr_bytes = 0
        for j in range(1 << bits):
            lo, hi = bounds[j], bounds[j + 1]
            a, nb = build_adr(r.adr_path(j), keys[lo:hi], pages[lo:hi], SEED_RUN ^ (rid << 8) ^ j, rid)
            r.addr.append(a)
            adr_bytes += nb
        r._arr = (keys, pages, h)
        return r, {"run": rid, "tier": tier, "keys": len(keys), "sections": 1 << bits, "adr_bytes": adr_bytes,
                   "log_bytes": r.log_path.stat().st_size, "build_s": time.perf_counter() - t0}

    def arrays(self, hz: Hasher) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._arr is None:
            log = np.frombuffer(self.log_path.read_bytes(), LOGDT)
            keys = as_keys(log["k"])
            self._arr = (keys, log["p"].astype(np.uint32), hz.np(keys))
        return self._arr

    def entries(self) -> dict[bytes, int]:
        log = np.frombuffer(self.log_path.read_bytes(), LOGDT)
        kb = log["k"].tobytes()
        return {kb[i * 16:(i + 1) * 16]: int(p) for i, p in enumerate(log["p"].tolist())}

    def page_of(self, key: bytes, h: int) -> int | None:
        a = self.addr[sect_int(h, self.bits)]
        return a.page_of(key) if a is not None else None

    def files(self) -> list[Path]:
        return [self.log_path] + [self.adr_path(j) for j in range(1 << self.bits) if self.offsets[j + 1] > self.offsets[j]]

    def nbytes(self) -> int:
        return sum(p.stat().st_size for p in self.files())

    def close(self) -> None:
        for a in self.addr:
            if a is not None:
                a.close()
        self.addr = []
        self._arr = None

    def remove(self) -> None:
        self.close()
        for p in self.files():
            p.unlink(missing_ok=True)


class MergeJob:
    """A tier merge executed one output section per step (§2.2)."""

    def __init__(self, lsm: "AddressLSM", inputs: list[Run], rid: int, tier: int) -> None:
        self.lsm, self.inputs, self.id, self.tier = lsm, inputs, rid, tier
        self.input_ids = {r.id for r in inputs}
        self.bits = bits_for(sum(r.n for r in inputs), lsm.sigma)
        self.done = 0
        self.offsets = [0]
        self.addr: list[C.SidecarA | None] = []
        self.out = Run(lsm.dir, rid, max(r.prio for r in inputs), tier, self.bits, [], pre=any(r.pre for r in inputs))
        self.part = lsm.dir / f"r{rid:06d}.log.part"
        self.part_f = open(self.part, "wb")
        self.read_bytes = self.written_bytes = 0
        self.max_in = self.max_out = 0

    @property
    def finished(self) -> bool:
        return self.done == (1 << self.bits)

    def step(self) -> dict:
        t0 = time.perf_counter()
        j, bits, hz, mut = self.done, self.bits, self.lsm.hz, self.lsm.mutant
        ks, ps, hs, pr = [], [], [], []
        for r in self.inputs:
            keys, pages, h = r.arrays(hz)
            if bits:
                lo = int(np.searchsorted(h, U64(j) << U64(64 - bits)))
                hi = int(np.searchsorted(h, U64(j + 1) << U64(64 - bits))) if j + 1 < (1 << bits) else len(h)
            else:
                lo, hi = 0, len(h)
            ks.append(keys[lo:hi]); ps.append(pages[lo:hi]); hs.append(h[lo:hi])
            pr.append(np.full(hi - lo, r.prio, np.int64))
            self.read_bytes += (hi - lo) * LOGDT.itemsize
        keys, pages, h, prio = np.concatenate(ks), np.concatenate(ps), np.concatenate(hs), np.concatenate(pr)
        self.max_in = max(self.max_in, len(keys))
        w0, w1 = kwords(keys)
        sign = 1 if mut == "b" else -1  # newest (highest priority) first; mutant (b): oldest first
        if mut == "l":  # mutant (l): versions grouped by h only, not by the full key
            order = np.lexsort((sign * prio, h))
            g = h[order]
            first = np.ones(len(order), bool)
            first[1:] = g[1:] != g[:-1]
        else:
            order = np.lexsort((sign * prio, w1, w0))
            a0, a1 = w0[order], w1[order]
            first = np.ones(len(order), bool)
            first[1:] = (a0[1:] != a0[:-1]) | (a1[1:] != a1[:-1])
        sel = order[first]
        keys, pages, h = keys[sel], pages[sel], h[sel]
        w0, w1 = kwords(keys)
        o2 = np.lexsort((w1, w0, h))
        keys, pages = keys[o2], pages[o2]
        log = np.zeros(len(keys), LOGDT)
        if len(keys):
            log["k"] = keys.view("V16").ravel()
            log["p"] = pages
        self.part_f.write(log.tobytes())
        self.offsets.append(self.offsets[-1] + len(keys))
        a, nb = build_adr(self.out.adr_path(j), keys, pages, SEED_RUN ^ (self.id << 8) ^ j, self.id)
        self.addr.append(a)
        self.written_bytes += nb + log.nbytes
        self.max_out = max(self.max_out, len(keys))
        self.done += 1
        return {"kind": "merge_step", "tier": self.tier, "section": j, "sections": 1 << bits, "keys_in": int(sum(len(x) for x in ks)),
                "keys_out": int(len(keys)), "s": time.perf_counter() - t0, "_keys": keys}

    def page_of(self, key: bytes, h: int) -> int | None:
        j = sect_int(h, self.bits)
        if j >= len(self.addr) or self.addr[j] is None:
            return None
        return self.addr[j].page_of(key)

    def finish(self) -> Run:
        self.part_f.flush()
        os.fsync(self.part_f.fileno())
        self.part_f.close()
        os.replace(self.part, self.out.log_path)
        self.out.offsets = self.offsets
        self.out.addr = self.addr
        return self.out

    def abandon(self) -> None:
        if not self.part_f.closed:
            self.part_f.close()
        for a in self.addr:
            if a is not None:
                a.close()
        self.part.unlink(missing_ok=True)
        for j in range(1 << self.bits):
            self.out.adr_path(j).unlink(missing_ok=True)

    def nbytes(self) -> int:
        b = self.part.stat().st_size if self.part.exists() else 0
        return b + sum(self.out.adr_path(j).stat().st_size for j in range(self.done) if self.addr[j] is not None)


# ------------------------------------------------------------------ the level set
class AddressLSM:
    """Mutants (§4): a single-state tail; b oldest wins in merges; e pre-W runs dropped at the first section;
    f last page slice skipped; g pre-W and post-W runs merged together; i no forced seal at W, tail dropped at the end;
    k merge output priority = completion counter; l merge groups versions by h only; z lookup of a not-ready merge
    section uses the output."""

    def __init__(self, d: Path, *, t_run: int = 30_000, f_major: float = 0.1, sigma: int = SIGMA,
                 s_base_bits: int | None = None, mutant: str | None = None, h_trunc: int | None = None) -> None:
        self.dir = d
        self.t_run, self.f_major, self.sigma, self.mutant = t_run, f_major, sigma, mutant
        self.s_base_bits = s_base_bits
        self.hz = Hasher(h_trunc)
        self.h_trunc = h_trunc
        self.base: list[list] = []  # per section: [gen, n, SidecarA | None]
        self.bits_base = 0
        self.runs: list[Run] = []  # highest priority (newest) first
        self.tail: TailLog | None = None
        self.job: MergeJob | None = None
        self.round: dict | None = None
        self.round_pending = False
        self.gen = 0
        self.base_gen = 1
        self.next_id = 1
        self.next_prio = 1
        self.tail_no = 1
        self.opseq = 0
        self.batch_no = 0
        self.ops_total = 0
        self.round_start_ops = 0
        self.n0 = 0
        self.events: list[dict] = []
        self.peak_published = 0
        self.peak_layer = 0
        self.spill_peak = 0
        self.zlib_probe: dict | None = None

    # a4_run.A4System.apply adds committed operations to `ops_since_major`; in A5 that is the monotonic counter
    @property
    def ops_since_major(self) -> int:
        return self.ops_total

    @ops_since_major.setter
    def ops_since_major(self, v: int) -> None:
        self.ops_total = v

    # ---- files
    @property
    def manifest(self) -> Path:
        return self.dir / MANIFEST

    def base_path(self, s: int, gen: int) -> Path:
        return self.dir / f"b{s:04d}_g{gen:04d}.adr"

    def spill_path(self, s: int) -> Path:
        return self.dir / f"w{s:04d}.spill"

    def publish(self) -> None:
        self.gen += 1
        m = {"gen": self.gen, "bits_base": self.bits_base, "base": [[g, n] for g, n, _ in self.base],
             "runs": [r.meta() for r in self.runs], "tail": self.tail.path.name, "opseq": self.opseq,
             "batch_no": self.batch_no, "ops_total": self.ops_total, "round_start_ops": self.round_start_ops,
             "n0": self.n0, "next_id": self.next_id, "next_prio": self.next_prio, "tail_no": self.tail_no,
             "base_gen": self.base_gen, "tail_ops": self.tail.committed_ops,
             "job": self.job.id if self.job else None,
             "round": None if self.round is None else {k: self.round[k] for k in ("W", "W_ops", "P0", "phase", "j")}}
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(m), encoding="utf-8")
        os.replace(tmp, self.manifest)
        self.track()

    @classmethod
    def create(cls, d: Path, keys: np.ndarray, pages: np.ndarray, n_live: int, **kw) -> "AddressLSM":
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
        s = cls(d, **kw)
        s.tail = TailLog(d / f"tail{s.tail_no:06d}.log", create=True, single_state=s.mutant == "a")
        s.bits_base = s.s_base_bits if s.s_base_bits is not None else bits_for(n_live, s.sigma)
        h = s.hz.np(keys)
        sec = sect_np(h, s.bits_base)
        order = np.argsort(sec, kind="stable")
        bounds = np.searchsorted(sec[order], np.arange((1 << s.bits_base) + 1))
        for j in range(1 << s.bits_base):
            idx = order[bounds[j]:bounds[j + 1]]
            a, _ = build_adr(s.base_path(j, 1), keys[idx], pages[idx], SEED_BASE ^ (1 << 12) ^ j, 1)
            s.base.append([1, int(len(idx)), a])
        s.n0 = n_live
        s.publish()
        return s

    @classmethod
    def open(cls, d: Path, *, ignore_open_batch: bool = False, **kw) -> "AddressLSM":
        """Reopen from the manifest. An unfinished round or tier merge is abandoned (§2.3, R3)."""
        m = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
        s = cls(d, **kw)
        for k in ("gen", "bits_base", "opseq", "batch_no", "ops_total", "round_start_ops", "n0", "next_id", "next_prio",
                  "tail_no", "base_gen"):
            setattr(s, k, m[k])
        s.base = [[g, n, C.SidecarA(s.base_path(j, g)) if n else None] for j, (g, n) in enumerate(m["base"])]
        s.runs = [Run.from_meta(d, r) for r in m["runs"]]
        for r in s.runs:
            r.pre = False
        s.tail = TailLog(d / m["tail"], create=False, ignore_open_batch=ignore_open_batch)
        s.opseq = max(s.opseq, s.tail.last_opseq)
        s.batch_no = max(s.batch_no, s.tail.last_batch)
        s.ops_total = m["ops_total"] + max(0, s.tail.committed_ops - m["tail_ops"])  # commits after the last publish
        abandoned = {}
        if m["job"] is not None:
            for p in d.glob(f"r{m['job']:06d}*"):
                p.unlink(missing_ok=True)
            abandoned["merge"] = m["job"]
        if m["round"] is not None:
            for p in d.glob("w*.spill"):
                p.unlink(missing_ok=True)
            s.round_start_ops = m["round"]["W_ops"]  # the counter keeps the abandoned round's start (§2.3)
            abandoned["round"] = m["round"]
        s.abandoned = abandoned
        return s

    # ---- lookup
    def candidates(self, key: bytes, writer: bool) -> list[int]:
        out = self.tail.candidates(key, writer)
        h = self.hz.one(key)
        job = self.job
        use_out = job is not None and (sect_int(h, job.bits) < job.done or self.mutant == "z")
        emitted = False
        for r in self.runs:
            if job is not None and r.id in job.input_ids:
                if use_out:
                    if not emitted:
                        emitted = True
                        p = job.page_of(key, h)
                        if p is not None:
                            out.append(p)
                    continue
            p = r.page_of(key, h)
            if p is not None:
                out.append(p)
        b = self.base[sect_int(h, self.bits_base)][2]
        if b is not None:
            p = b.page_of(key)
            if p is not None:
                out.append(p)
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # ---- maintenance: seal
    def new_tail(self) -> TailLog:
        old = self.tail
        self.tail_no += 1
        self.tail = TailLog(self.dir / f"tail{self.tail_no:06d}.log", create=True, single_state=self.mutant == "a")
        return old

    def seal(self, forced: bool = False) -> dict:
        t0 = time.perf_counter()
        items = [(k, p) for k, p in self.tail.committed.items() if p is not None]
        keys = np.frombuffer(b"".join(k for k, _ in items), np.uint8).reshape(-1, 16) if items else np.zeros((0, 16), np.uint8)
        pages = np.fromiter((p for _, p in items), np.uint32, len(items))
        run, info = Run.build(self.dir, self.next_id, self.next_prio, 0, keys, pages, self.hz, self.sigma)
        self.next_id += 1
        self.next_prio += 1
        self.runs.insert(0, run)
        old = self.new_tail()
        self.publish()
        old.close()
        old.path.unlink(missing_ok=True)
        ev = {"kind": "seal", "forced": forced, "after_op": self.opseq, "run": info, "s": time.perf_counter() - t0,
              "_keys": keys}
        return ev

    # ---- maintenance: tier merges (§2.2)
    def eligible(self) -> list[Run]:
        if self.round is not None and self.mutant != "g":
            return [r for r in self.runs if not r.pre]
        return list(self.runs)

    def maybe_start_merge(self) -> bool:
        if self.job is not None or self.round_pending:
            return False
        el = self.eligible()
        for tier in sorted({r.tier for r in el}):
            same = sorted((r for r in el if r.tier == tier), key=lambda r: r.prio)
            if len(same) >= FANOUT:
                self.job = MergeJob(self, same[:FANOUT], self.next_id, tier + 1)
                self.next_id += 1
                return True
        return False

    def merge_step(self) -> dict:
        t0 = time.perf_counter()
        job = self.job
        ev = job.step()
        if job.finished:
            out = job.finish()
            if self.mutant == "k":
                out.prio = self.next_prio  # mutant (k): priority = completion counter
                self.next_prio += 1
            elif self.mutant == "k2":
                out.prio = -max(r.prio for r in job.inputs)  # mutant (k'), addendum 2026-11-16: not inherited
            self.runs = [r for r in self.runs if r.id not in job.input_ids] + [out]
            self.runs.sort(key=lambda r: -r.prio)
            self.job = None
            self.publish()
            for r in job.inputs:
                r.remove()
            ev.update(finished=True, read_bytes=job.read_bytes, written_bytes=job.written_bytes, max_in=job.max_in,
                      max_out=job.max_out, out_keys=out.n, out_tier=out.tier)
        else:
            self.track()
        ev["s"] = time.perf_counter() - t0
        return ev

    # ---- maintenance: the round (§2.3)
    def round_due(self) -> bool:
        return self.round is None and not self.round_pending and \
            self.ops_total - self.round_start_ops >= self.f_major * self.n0

    def round_start(self, p0: int) -> dict:
        t0 = time.perf_counter()
        ev = {"kind": "round_start", "after_op": self.opseq}
        if self.mutant != "i":
            ev["seal"] = self.seal(forced=True)
        for r in self.runs:
            r.pre = True
        nb = 1 << self.bits_base
        self.round = {"W": self.opseq, "W_ops": self.ops_total, "P0": p0, "phase": "scan", "j": 0,
                      "bounds": [j * p0 // nb for j in range(nb)] + [p0], "spill_bytes": 0, "rows": 0, "dups": 0,
                      "read_rows": 0, "max_slice_rows": 0, "max_section_keys": 0, "steps": 0, "t0_op": self.opseq,
                      "pre_dropped": False,
                      # new section files never reuse a generation already on disk (an abandoned round published some)
                      "gen": max(g for g, _, _ in self.base) + 1}
        for s in range(nb):
            self.spill_path(s).write_bytes(b"")
        self.round_pending = False
        self.publish()
        ev["s"] = time.perf_counter() - t0
        ev["P0"] = p0
        return ev

    def next_slice(self) -> tuple[int, int] | None:
        rd = self.round
        if rd is None or rd["phase"] != "scan":
            return None
        j = rd["j"]
        return rd["bounds"][j], rd["bounds"][j + 1]

    def scan_add(self, keys: np.ndarray, pages: np.ndarray) -> dict:
        """One scan step: rows of slice j appended to the work file of their section."""
        t0 = time.perf_counter()
        rd = self.round
        j = rd["j"]
        if self.mutant == "f" and j == len(rd["bounds"]) - 2:
            keys, pages = keys[:0], pages[:0]  # mutant (f): the last slice is not read
        sec = sect_np(self.hz.np(keys), self.bits_base)
        order = np.argsort(sec, kind="stable")
        bounds = np.searchsorted(sec[order], np.arange((1 << self.bits_base) + 1))
        log = np.zeros(len(keys), LOGDT)
        if len(keys):
            log["k"] = keys.view("V16").ravel()
            log["p"] = pages
        log = log[order]
        for s in range(1 << self.bits_base):
            lo, hi = bounds[s], bounds[s + 1]
            if hi > lo:
                with open(self.spill_path(s), "ab") as f:
                    f.write(log[lo:hi].tobytes())
        rd["spill_bytes"] += log.nbytes
        rd["rows"] += len(keys)
        rd["max_slice_rows"] = max(rd["max_slice_rows"], len(keys))
        rd["j"] += 1
        rd["steps"] += 1
        if rd["j"] == len(rd["bounds"]) - 1:
            rd["phase"], rd["j"] = "build", 0
        self.publish()
        self.spill_peak = max(self.spill_peak, self.spill_bytes())
        return {"kind": "scan_step", "slice": j, "rows": int(len(keys)), "pages": rd["bounds"][j + 1] - rd["bounds"][j],
                "lo": rd["bounds"][j], "hi": rd["bounds"][j + 1], "s_add": time.perf_counter() - t0, "_keys": keys,
                "_pages": pages}

    def build_step(self) -> dict:
        t0 = time.perf_counter()
        rd = self.round
        s = rd["j"]
        raw = self.spill_path(s).read_bytes()
        if s == 0 and self.zlib_probe is None and self.mutant is None and len(raw):
            import zlib
            tz = time.perf_counter()
            z = zlib.compress(raw, 6)
            self.zlib_probe = {"section_bytes": len(raw), "zlib_bytes": len(z), "ratio": len(z) / len(raw),
                               "compress_s": time.perf_counter() - tz, "roundtrip_ok": zlib.decompress(z) == raw}
            t0 += time.perf_counter() - tz  # the descriptive probe is not maintenance
        log = np.frombuffer(raw, LOGDT)
        keys = as_keys(log["k"])
        pages = log["p"].astype(np.uint32)
        if len(keys):
            w0, w1 = kwords(keys)
            order = np.lexsort((w1, w0))
            a0, a1 = w0[order], w1[order]
            first = np.ones(len(order), bool)
            first[1:] = (a0[1:] != a0[:-1]) | (a1[1:] != a1[:-1])
            rd["dups"] += int((~first).sum())
            sel = order[first]
            keys, pages = keys[sel], pages[sel]
        gen = rd["gen"]
        a, nb = build_adr(self.base_path(s, gen), keys, pages, SEED_BASE ^ (gen << 12) ^ s, gen)
        old = self.base[s]
        self.base[s] = [gen, int(len(keys)), a]
        rd["max_section_keys"] = max(rd["max_section_keys"], len(keys))
        rd["j"] += 1
        rd["steps"] += 1
        last = rd["j"] == len(self.base)
        if self.mutant == "e" and not rd["pre_dropped"]:
            self.drop_pre()  # mutant (e): pre-W runs dropped at the FIRST section
        self.publish()
        if old[2] is not None:
            old[2].close()
        self.base_path(s, old[0]).unlink(missing_ok=True)
        self.spill_path(s).unlink(missing_ok=True)
        ev = {"kind": "build_step", "section": s, "keys": int(len(keys)), "adr_bytes": nb, "s": time.perf_counter() - t0,
              "_keys": keys, "last": last}
        if last:
            ev["round_end"] = self.round_end()
        return ev

    def drop_pre(self) -> None:
        pre = [r for r in self.runs if r.pre]
        self.runs = [r for r in self.runs if not r.pre]
        self.round["pre_dropped"] = True
        self._dropped = pre

    def round_end(self) -> dict:
        t0 = time.perf_counter()
        rd = self.round
        if not rd["pre_dropped"]:
            self.drop_pre()
        dropped = getattr(self, "_dropped", [])
        self._dropped = []
        old_tail = None
        if self.mutant == "i":
            old_tail = self.new_tail()  # mutant (i): the tail "before W" is dropped together with everything after W
        self.base_gen = rd["gen"]
        self.round_start_ops = rd["W_ops"]
        self.n0 = sum(n for _, n, _ in self.base)
        summary = {k: rd[k] for k in ("W", "P0", "rows", "dups", "spill_bytes", "max_slice_rows", "max_section_keys",
                                       "steps")}
        summary["window_ops"] = self.opseq - rd["t0_op"]
        self.round = None
        self.publish()
        for r in dropped:
            r.remove()
        if old_tail is not None:
            old_tail.close()
            old_tail.path.unlink(missing_ok=True)
        summary["s"] = time.perf_counter() - t0
        return summary

    # ---- one batch boundary (§2.2, §2.3): seal, then at most one merge step, then at most one round step
    def boundary(self, scan_fn, p0_fn) -> list[dict]:
        evs = []
        if self.tail.committed_ops >= self.t_run:
            evs.append(self.seal())
        if self.round_due():
            self.round_pending = True
        if self.job is not None or self.maybe_start_merge():
            evs.append(self.merge_step())
        if self.round_pending and self.round is None and self.job is None:
            evs.append(self.round_start(p0_fn()))
        elif self.round is not None:
            if self.round["phase"] == "scan":
                lo, hi = self.next_slice()
                t0 = time.perf_counter()
                keys, pages, info = scan_fn(lo, hi)
                t_read = time.perf_counter() - t0
                ev = self.scan_add(keys, pages)
                ev.update(info)
                ev["s_read"] = t_read
                ev["s"] = t_read + ev["s_add"]
                evs.append(ev)
            else:
                evs.append(self.build_step())
        for e in evs:
            e.setdefault("after_op", self.opseq)
        self.events.extend(evs)
        return evs

    def run_round_now(self, scan_fn, p0_fn) -> list[dict]:
        """Final round: steps back to back without operations (§3)."""
        self.round_pending = self.round is None
        evs = []
        guard = 0
        while self.round_pending or self.round is not None or self.job is not None:
            evs.extend(self.boundary(scan_fn, p0_fn))
            guard += 1
            if guard > 100_000:
                raise RuntimeError("round did not finish")
        return evs

    # ---- accounting
    def spill_bytes(self) -> int:
        if self.round is None:
            return 0
        return sum(self.spill_path(s).stat().st_size for s in range(len(self.base)) if self.spill_path(s).exists())

    def nbytes(self) -> int:
        """Published structure (S3): base sections, runs (incl. pinned merge inputs), built sections of an active
        merge, tail, manifest."""
        total = sum(self.base_path(j, g).stat().st_size for j, (g, n, _) in enumerate(self.base) if n)
        total += sum(r.nbytes() for r in self.runs)
        if self.job is not None:
            total += self.job.nbytes()
        total += self.tail.nbytes()
        if self.manifest.exists():
            total += self.manifest.stat().st_size
        return total

    def track(self) -> None:
        nb = self.nbytes()
        self.peak_published = max(self.peak_published, nb)
        self.peak_layer = max(self.peak_layer, nb + self.spill_bytes())

    def levels(self) -> int:
        return len(self.runs) + 2

    def close(self) -> None:
        if self.job is not None:
            self.job.abandon()
            self.job = None
        for _, _, a in self.base:
            if a is not None:
                a.close()
        for r in self.runs:
            r.close()
        if self.tail is not None:
            self.tail.close()
