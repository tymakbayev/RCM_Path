"""Offline property test of a5_lsm (no SQL): an in-memory table (key -> page), random I/U/D batches, seals, stepwise
tier merges and rounds driven by AddressLSM.boundary. After every boundary, every live key must have its true page
among the candidates. Also run per mutant: a harmful mutant must break the property (or a merge witness).
Run: .venv-pg/Scripts/python.exe a5_selftest_lsm.py [mutant ...]"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np

import a4_lsm as L4
import a5_lsm as L

ROOT = Path(__file__).resolve().parent


def run(mutant: str | None, n: int = 3000, batches: int = 400, seed: int = 1) -> dict:
    d = ROOT / "data" / f"a5_selftest_{mutant or 'ok'}"
    if d.exists():
        shutil.rmtree(d)
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    pages = rng.integers(0, 300, n).astype(np.uint32)
    table = {bytes(k): int(p) for k, p in zip(keys, pages)}
    lsm = L.AddressLSM.create(d, keys, pages, n, t_run=40, f_major=0.1, sigma=64, s_base_bits=3, mutant=mutant)
    prng = random.Random(seed)
    top = [300]
    bad_total, rounds, merges, checks = 0, 0, 0, 0

    def scan_fn(lo, hi):
        items = [(k, p) for k, p in table.items() if lo <= p < hi]
        if mutant == "f_test":
            items = []
        ks = np.frombuffer(b"".join(k for k, _ in items), np.uint8).reshape(-1, 16) if items else np.zeros((0, 16), np.uint8)
        return ks, np.fromiter((p for _, p in items), np.uint32, len(items)), {"read_bytes": 0}

    def p0_fn():
        return top[0]

    for b in range(batches):
        lsm.batch_no += 1
        for _ in range(10):
            lsm.opseq += 1
            r = prng.random()
            if r < 0.4:
                kb = prng.randbytes(16)
                pg = prng.randrange(top[0] + 5)
                top[0] = max(top[0], pg + 1)
                table[kb] = pg
                lsm.tail.append(L4.I_, kb, pg, lsm.opseq)
            elif r < 0.8:
                kb = prng.choice(list(table)) if len(table) < 50 else prng.choice(tuple(table))
                pg = prng.randrange(top[0])
                table[kb] = pg
                lsm.tail.append(L4.U_, kb, pg, lsm.opseq)
            else:
                kb = prng.choice(tuple(table))
                del table[kb]
                lsm.tail.append(L4.D_, kb, 0, lsm.opseq)
        lsm.ops_total += lsm.tail.commit(lsm.batch_no)
        evs = lsm.boundary(scan_fn, p0_fn)
        rounds += sum(1 for e in evs if e["kind"] == "round_start")
        merges += sum(1 for e in evs if e["kind"] == "merge_step" and e.get("finished"))
        if b % 5 == 0 or evs:
            sample = prng.sample(tuple(table), min(400, len(table)))
            bad = sum(1 for kb in sample if table[kb] not in lsm.candidates(kb, False))
            bad_total += bad
            checks += 1
        if b == batches // 2:  # restart in the middle
            lsm.close()
            lsm = L.AddressLSM.open(d, t_run=40, f_major=0.1, sigma=64, s_base_bits=3, mutant=mutant)
    # final: every live key
    bad_final = sum(1 for kb in table if table[kb] not in lsm.candidates(kb, False))
    lsm.close()
    return {"mutant": mutant, "ok": bad_total == 0 and bad_final == 0, "bad_sampled": bad_total, "bad_final": bad_final,
            "rounds": rounds, "merges": merges, "checks": checks, "live": len(table)}


if __name__ == "__main__":
    ms = sys.argv[1:] or [None]
    for m in ms:
        print(run(None if m == "none" else m))
