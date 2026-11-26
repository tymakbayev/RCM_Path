"""A6 §2.3: client copy of PostgreSQL's hash-partition routing for one uuid column.

g(k) = hash_combine64(0, uuid_hash_extended(k, HASH_PARTITION_SEED)) mod S
uuid_hash_extended = hash_bytes_extended (Jenkins lookup3 as in PostgreSQL src/common/hashfn.c) over the 16 raw bytes.
Little-endian host only (the server here is x86-64 Windows). Verified against the server: `selftest` below.
"""

from __future__ import annotations

import numpy as np

HASH_PARTITION_SEED = 0x7A5B22367996DCFD
COMBINE = 0x49A0F4DD15E5A8E3
M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


# ------------------------------------------------------------------ scalar
def _rot(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & M32


def _mix(a: int, b: int, c: int) -> tuple[int, int, int]:
    a = (a - c) & M32; a ^= _rot(c, 4); c = (c + b) & M32
    b = (b - a) & M32; b ^= _rot(a, 6); a = (a + c) & M32
    c = (c - b) & M32; c ^= _rot(b, 8); b = (b + a) & M32
    a = (a - c) & M32; a ^= _rot(c, 16); c = (c + b) & M32
    b = (b - a) & M32; b ^= _rot(a, 19); a = (a + c) & M32
    c = (c - b) & M32; c ^= _rot(b, 4); b = (b + a) & M32
    return a, b, c


def _final(a: int, b: int, c: int) -> tuple[int, int, int]:
    c ^= b; c = (c - _rot(b, 14)) & M32
    a ^= c; a = (a - _rot(c, 11)) & M32
    b ^= a; b = (b - _rot(a, 25)) & M32
    c ^= b; c = (c - _rot(b, 16)) & M32
    a ^= c; a = (a - _rot(c, 4)) & M32
    b ^= a; b = (b - _rot(a, 14)) & M32
    c ^= b; c = (c - _rot(b, 24)) & M32
    return a, b, c


def hash16_extended(k: bytes, seed: int) -> int:
    """hash_bytes_extended for exactly 16 bytes (one 12-byte block + 4 trailing bytes)."""
    a = b = c = (0x9E3779B9 + 16 + 3923095) & M32
    if seed:
        a = (a + (seed >> 32)) & M32
        b = (b + (seed & M32)) & M32
        a, b, c = _mix(a, b, c)
    w = int.from_bytes
    a = (a + w(k[0:4], "little")) & M32
    b = (b + w(k[4:8], "little")) & M32
    c = (c + w(k[8:12], "little")) & M32
    a, b, c = _mix(a, b, c)
    a = (a + w(k[12:16], "little")) & M32
    a, b, c = _final(a, b, c)
    return (b << 32) | c


def part_hash(k: bytes) -> int:
    return (hash16_extended(k, HASH_PARTITION_SEED) + COMBINE) & M64


def part_of(k: bytes, modulus: int) -> int:
    return part_hash(k) % modulus


# ------------------------------------------------------------------ numpy (vectorised over keys uint8[n,16])
def _rot_np(x, k):
    return (x << np.uint32(k)) | (x >> np.uint32(32 - k))


def _mix_np(a, b, c):
    a -= c; a ^= _rot_np(c, 4); c += b
    b -= a; b ^= _rot_np(a, 6); a += c
    c -= b; c ^= _rot_np(b, 8); b += a
    a -= c; a ^= _rot_np(c, 16); c += b
    b -= a; b ^= _rot_np(a, 19); a += c
    c -= b; c ^= _rot_np(b, 4); b += a
    return a, b, c


def _final_np(a, b, c):
    c ^= b; c -= _rot_np(b, 14)
    a ^= c; a -= _rot_np(c, 11)
    b ^= a; b -= _rot_np(a, 25)
    c ^= b; c -= _rot_np(b, 16)
    a ^= c; a -= _rot_np(c, 4)
    b ^= a; b -= _rot_np(a, 14)
    c ^= b; c -= _rot_np(b, 24)
    return a, b, c


def part_hash_np(keys: np.ndarray) -> np.ndarray:
    keys = np.ascontiguousarray(keys, dtype=np.uint8)
    w = keys.view("<u4").reshape(-1, 4)
    n = len(w)
    seed = HASH_PARTITION_SEED
    with np.errstate(over="ignore"):
        a = np.full(n, (0x9E3779B9 + 16 + 3923095) & M32, np.uint32)
        b = a.copy()
        c = a.copy()
        a += np.uint32(seed >> 32)
        b += np.uint32(seed & M32)
        a, b, c = _mix_np(a, b, c)
        a += w[:, 0]
        b += w[:, 1]
        c += w[:, 2]
        a, b, c = _mix_np(a, b, c)
        a += w[:, 3]
        a, b, c = _final_np(a, b, c)
        h = (b.astype(np.uint64) << np.uint64(32)) | c.astype(np.uint64)
        return h + np.uint64(COMBINE)


def part_of_np(keys: np.ndarray, modulus: int) -> np.ndarray:
    return (part_hash_np(keys) % np.uint64(modulus)).astype(np.int64)


# ------------------------------------------------------------------ check against the server
def selftest(n: int = 20_000) -> dict:
    import uuid

    import a2_pg as G
    rng = np.random.default_rng(5)
    keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    conn = G.connect()
    signed = HASH_PARTITION_SEED - (1 << 64) if HASH_PARTITION_SEED >> 63 else HASH_PARTITION_SEED
    us = [uuid.UUID(bytes=bytes(k)) for k in keys]
    srv = [r[0] for r in conn.execute("SELECT uuid_hash_extended(u, %s) FROM unnest(%s::uuid[]) AS u",
                                      (signed, us)).fetchall()]
    raw = [hash16_extended(bytes(k), HASH_PARTITION_SEED) for k in keys]
    bad_raw = sum(1 for s, r in zip(srv, raw) if (s & M64) != r)
    vec = part_hash_np(keys)
    bad_vec = int(np.sum(vec != np.array([(r + COMBINE) & M64 for r in raw], np.uint64)))
    # routing: a throw-away hash-partitioned table, rows routed by the server, compared by tableoid
    out = {"n": n, "bad_uuid_hash_extended": bad_raw, "bad_numpy_vs_scalar": bad_vec}
    for s in (8, 128):
        conn.execute("DROP TABLE IF EXISTS a6_hashcheck")
        conn.execute("CREATE TABLE a6_hashcheck (k uuid NOT NULL) PARTITION BY HASH (k)")
        for j in range(s):
            conn.execute(f"CREATE TABLE a6_hashcheck_p{j} PARTITION OF a6_hashcheck FOR VALUES WITH (MODULUS {s}, REMAINDER {j})")
        conn.execute("INSERT INTO a6_hashcheck SELECT unnest(%s::uuid[])", (us,))
        got = dict((r[0].bytes, int(r[1].rsplit("_p", 1)[1])) for r in
                   conn.execute("SELECT k, tableoid::regclass::text FROM a6_hashcheck").fetchall())
        mine = part_of_np(keys, s)
        out[f"bad_routing_S{s}"] = int(sum(1 for k, p in zip(keys, mine) if got[bytes(k)] != p))
        # a direct insert into the wrong partition must be rejected
        k0 = bytes(keys[0])
        wrong = (part_of(k0, s) + 1) % s
        try:
            conn.execute(f"INSERT INTO a6_hashcheck_p{wrong} VALUES (%s)", (uuid.UUID(bytes=k0),))
            out[f"wrong_partition_rejected_S{s}"] = False
        except Exception as e:  # noqa: BLE001
            out[f"wrong_partition_rejected_S{s}"] = type(e).__name__
    conn.execute("DROP TABLE IF EXISTS a6_hashcheck")
    return out


if __name__ == "__main__":
    print(selftest())
