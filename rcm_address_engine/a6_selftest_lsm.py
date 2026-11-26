"""Offline property test of a6_lsm (no SQL), protocol A6 §4: an in-memory partitioned table (key -> (g(k), page)),
random I/U/D batches, seals, stepwise tier merges and rounds by partitions driven by AddressLSM6.boundary, a restart
in the middle of a round (the round resumes). After every boundary every sampled live key must have its true
(partition, page) among the targets; at the end every live key. Run per mutant: a harmful mutant must break it.
Run: .venv-pg/Scripts/python.exe a6_selftest_lsm.py [mutant ...]"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np

import a4_lsm as L4
import a6_lsm as L
import a6_pghash as PH

ROOT = Path(__file__).resolve().parent
S_BITS = 3


def run(mutant: str | None, n: int = 3000, batches: int = 400, seed: int = 1) -> dict:
    d = ROOT / "data" / f"a6_selftest_{mutant or 'ok'}"
    if d.exists():
        shutil.rmtree(d)
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    pages = rng.integers(0, 60, n).astype(np.uint32)
    S = 1 << S_BITS
    table = {bytes(k): int(p) for k, p in zip(keys, pages)}  # true partition = PostgreSQL's g(k)
    kw = dict(t_run=40, f_major=0.1, sigma=64, s_base_bits=S_BITS, mutant=mutant)
    lsm = L.AddressLSM6.create(d, keys, pages, n, **kw)
    prng = random.Random(seed)
    top = [60]
    bad_total, rounds, merges, checks, restarts = 0, 0, 0, 0, 0
    resumed = None

    def read_fn(s):
        items = [(k, p) for k, p in table.items() if PH.part_of(k, S) == s]
        ks = np.frombuffer(b"".join(k for k, _ in items), np.uint8).reshape(-1, 16) if items else np.zeros((0, 16), np.uint8)
        return ks, np.fromiter((p for _, p in items), np.uint32, len(items)), {}

    def ok(kb):
        return (PH.part_of(kb, S), table[kb]) in lsm.targets(kb, False)

    for b in range(batches):
        lsm.batch_no += 1
        for _ in range(10):
            lsm.opseq += 1
            r = prng.random()
            if r < 0.4:
                kb = prng.randbytes(16)
                pg = prng.randrange(top[0] + 2)
                top[0] = max(top[0], pg + 1)
                table[kb] = pg
                lsm.n_live += 1
                lsm.tail.append(L4.I_, kb, pg, lsm.opseq)
            elif r < 0.8:
                kb = prng.choice(tuple(table))
                pg = prng.randrange(top[0])
                table[kb] = pg
                lsm.tail.append(L4.U_, kb, pg, lsm.opseq)
            else:
                kb = prng.choice(tuple(table))
                del table[kb]
                lsm.n_live -= 1
                lsm.tail.append(L4.D_, kb, 0, lsm.opseq)
        lsm.ops_total += lsm.tail.commit(lsm.batch_no)
        evs = lsm.boundary(read_fn)
        rounds += sum(1 for e in evs if e["kind"] == "round_start")
        merges += sum(1 for e in evs if e["kind"] == "merge_step" and e.get("finished"))
        if b % 5 == 0 or evs:
            sample = prng.sample(tuple(table), min(400, len(table)))
            bad_total += sum(1 for kb in sample if not ok(kb))
            checks += 1
        # restart in the middle of a round (twice): the round must resume
        if lsm.round is not None and lsm.round["s"] == S // 2 and restarts < 2:
            lsm.close()
            lsm = L.AddressLSM6.open(d, **kw, writer=True)
            resumed = lsm.resumed
            restarts += 1
    lsm.run_round_now(read_fn)
    bad_final = sum(1 for kb in table if not ok(kb))
    stray = lsm.stray_files()
    s5 = list(lsm.s5_violations)
    lsm.close()
    return {"mutant": mutant, "ok": bad_total == 0 and bad_final == 0 and not stray and not s5 and restarts == 2,
            "bad_sampled": bad_total, "bad_final": bad_final, "rounds": rounds, "merges": merges, "checks": checks,
            "restarts_mid_round": restarts, "resumed": resumed, "stray_files": stray, "s5_violations": s5[:3],
            "live": len(table)}


if __name__ == "__main__":
    ms = sys.argv[1:] or ["none"]
    for m in ms:
        try:
            print(run(None if m == "none" else m))
        except Exception as e:  # noqa: BLE001 - a crash is a catch too
            print({"mutant": m, "ok": False, "exception": f"{type(e).__name__}: {e}"})
