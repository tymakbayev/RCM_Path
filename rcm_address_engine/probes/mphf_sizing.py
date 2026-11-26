"""Sizing probe (not the engine): cascade perfect hash (BBHash scheme) on random 64-bit keys.
Reports bits/key of the collision bitmaps (the 'path' P), rank overhead, levels and mean probes per lookup."""
import sys, time
import numpy as np

M1, M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def mix(x, level):
    x = x + np.uint64(0x9E3779B97F4A7C15) * np.uint64(level + 1)
    x = (x ^ (x >> np.uint64(30))) * M1
    x = (x ^ (x >> np.uint64(27))) * M2
    return x ^ (x >> np.uint64(31))


def cascade(keys, gamma, max_levels=40):
    n, left, bits, probes, level = len(keys), keys, 0, 0, 0
    while len(left) and level < max_levels:
        m = max(64, int(gamma * len(left)))
        pos = ((mix(left, level) >> np.uint64(32)) * np.uint64(m)) >> np.uint64(32)
        alone = np.bincount(pos.astype(np.int64), minlength=m)[pos.astype(np.int64)] == 1
        bits += m
        level += 1
        probes += level * int(alone.sum())
        left = left[~alone]
    return bits / n, level, probes / n, len(left)


for n in (1_000_000, 10_000_000):
    keys = np.random.default_rng(1).integers(0, 2**63, size=n, dtype=np.uint64)
    for gamma in (1.0, 1.5, 2.0):
        t = time.perf_counter()
        bpk, levels, probes, fallback = cascade(keys, gamma)
        dt = time.perf_counter() - t
        rank = bpk * 32 / 512  # one u32 rank sample per 512 bitmap bits
        print(f"n={n:>9,} gamma={gamma:.1f}: P {bpk:5.2f} + rank {rank:4.2f} bits/key, levels {levels:2d}, "
              f"probes {probes:4.2f}, unplaced {fallback}, build {n / dt / 1e6:5.2f} M keys/s (numpy, 1 thread)")
