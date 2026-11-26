"""Offline checks of a2_core (no PostgreSQL): splitmix64 vectors, cascade build/lookup, packing edges, B, SORT, EF.
Exit code 0 only if every check holds."""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

import a2_core as C

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")


# splitmix64: numpy vs scalar with explicit masks
rng = random.Random(5)
vals = [0, 1, 1 << 63, C.M64] + [rng.getrandbits(64) for _ in range(10_000)]
got = C.splitmix64(np.array(vals, dtype=np.uint64))
check("splitmix64 numpy == scalar", all(int(g) == C.splitmix64_int(v) for g, v in zip(got, vals)), f"n={len(vals)}")
keys = np.random.default_rng(1).integers(0, 256, size=(5000, 16), dtype=np.uint8)
hv = C.key_hash(keys, 77)
check("key_hash numpy == scalar", all(int(hv[i]) == C.key_hash_int(bytes(keys[i]), 77) for i in range(5000)))
for m in (64, 1000, 20_000_000):
    lp = C.level_pos(hv, 3, m)
    check(f"level_pos numpy == scalar m={m}", all(int(lp[i]) == C.level_pos_int(int(hv[i]), 3, m) for i in range(2000))
          and int(lp.max()) < m)

# packing edges: widths crossing word boundaries, first/last slot
for width in (8, 21, 25, 28, 31, 57):
    v = np.random.default_rng(width).integers(0, 1 << width, size=1000, dtype=np.uint64)
    v[0], v[-1] = (1 << width) - 1, (1 << width) - 1
    w = C.pack_fixed(v, width)
    idx = np.arange(1000)
    check(f"pack/unpack width={width}", np.array_equal(C.unpack_fixed(w, idx, width), v)
          and all(C.unpack_fixed_int(w, i, width) == int(v[i]) for i in (0, 1, 2, 63, 64, 999)))

# cascade build / file round trip / lookups
for n in (1, 63, 480, 100_000):
    keys = np.random.default_rng(n).integers(0, 256, size=(n, 16), dtype=np.uint8)
    pages = np.random.default_rng(n + 1).integers(0, 1 << 17, size=n).astype(np.uint32)
    ca = C.build_a(keys, pages, seed=12345, r_bits=17)
    with tempfile.TemporaryDirectory() as d:
        info = C.write_a(Path(d) / "a.adr", ca, table_oid=1, relfilenode=2, generation=1, n_rows=n, n_pages=1 << 17,
                         watermark=0)
        sa = C.SidecarA(Path(d) / "a.adr")
        got = sa.pages_of(keys)
        check(f"A n={n}: every key -> its page (vector)", np.array_equal(got, pages.astype(np.int64)),
              f"levels={info['levels']} exc={info['exceptions']} bytes/key={info['bytes'] / n:.3f}")
        sample = range(0, n, max(1, n // 500))
        check(f"A n={n}: scalar == vector", all(sa.page_of(bytes(keys[i])) == int(pages[i]) for i in sample))
        absent = np.random.default_rng(999).integers(0, 256, size=(20_000, 16), dtype=np.uint8)
        fp = (sa.pages_of(absent) >= 0).mean()
        check(f"A n={n}: absent false-positive rate ~2^-8", fp < 0.008, f"{fp:.4f}")
        check(f"A n={n}: verify digest", sa.verify())
        sa.close()

# forced full-hash collision -> exception list (mutant (d) path)
keys = np.random.default_rng(3).integers(0, 256, size=(1000, 16), dtype=np.uint8)
pages = np.arange(1000, dtype=np.uint32)
real_hash = C.key_hash


def colliding(k, seed):
    h = real_hash(k, seed)
    if len(k) == 1000:
        h = h.copy()
        h[7] = h[500]
    return h


C.key_hash = colliding
ca = C.build_a(keys, pages, seed=9, r_bits=10)
C.key_hash = real_hash
check("collision -> both keys in exceptions", len(ca.exc_keys) == 2 and sorted(ca.exc_pages) == [7, 500])

# B: buckets, empty buckets, invariant
keys = np.random.default_rng(4).integers(0, 256, size=(3000, 16), dtype=np.uint8)
b = 10  # 1024 buckets for 3000 keys: some empty
bucket = C.buckets_of(keys, 5, b)
order = np.lexsort((np.arange(3000), bucket))
pages = np.empty(3000, np.int64)
pages[order] = np.arange(3000) // 14  # sequential fill, 14 rows/page
starts = C.bucket_starts(bucket, pages, b)
check("B invariant holds on sequential fill", C.bucket_invariant(bucket, pages, starts) == 0,
      f"empty buckets={int((np.bincount(bucket, minlength=1 << b) == 0).sum())}")
bad = pages.copy()
bad[order[100]] = 0  # one row placed out of order
check("B invariant catches a misplaced row", C.bucket_invariant(bucket, bad, C.bucket_starts(bucket, bad, b)) > 0)
seq = np.arange(3000)
st, exc_rows = C.bucket_layout(bucket, bad, seq, b)  # addendum §3: the back-filled row becomes an exception
check("B layout: back-filled row -> exception, invariant 0 with exceptions",
      list(exc_rows) == [order[100]] and C.bucket_invariant(bucket, bad, st, exc_rows) == 0
      and C.bucket_invariant(bucket, bad, st) == 1)
st0, exc0 = C.bucket_layout(bucket, pages, seq, b)
check("B layout on sequential fill == bucket_starts, no exceptions",
      len(exc0) == 0 and np.array_equal(st0, C.bucket_starts(bucket, pages, b)))
with tempfile.TemporaryDirectory() as d:
    i = int(order[100])
    C.write_b(Path(d) / "b.hpl", st, seed=5, b=b, table_oid=1, relfilenode=2, generation=1, n_rows=3000,
              n_pages=int(bad.max()) + 1, watermark=0, exc_keys=[bytes(keys[i])], exc_pages=[int(bad[i])])
    sb = C.SidecarB(Path(d) / "b.hpl")
    okb = all(sb.range_of(bytes(keys[j]))[0] <= bad[j] <= sb.range_of(bytes(keys[j]))[1] for j in range(3000))
    check("B file round trip: every key inside its range (exceptions first)", okb and sb.range_of(bytes(keys[i])) == (0, 0))
    sbm = C.SidecarB(Path(d) / "b.hpl", skip_exceptions=True)
    check("B mutant (e) skip exceptions -> the back-filled key falls outside", not (
        sbm.range_of(bytes(keys[i]))[0] <= bad[i] <= sbm.range_of(bytes(keys[i]))[1]))
    sb.close()
    sbm.close()

# SORT: duplicate prefixes on different pages
keys = np.random.default_rng(6).integers(0, 256, size=(200_000, 16), dtype=np.uint8)
pages = np.random.default_rng(7).integers(0, 1 << 15, size=200_000).astype(np.uint32)
pre, sp = C.build_sort(keys, pages, 11)
dups = int((np.diff(pre.astype(np.int64)) == 0).sum())
check("SORT C1 with equal prefixes", C.sort_c1(keys, pages, 11, pre, sp) == 0, f"equal-prefix pairs={dups}")
check("EF size plausible", 2.0 < C.ef_bytes(keys, 11, 15) < 6.0, f"{C.ef_bytes(keys, 11, 15):.3f} B/key")

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
