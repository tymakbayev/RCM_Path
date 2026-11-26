"""A5 runs (protocol A5_protocol.md edition 2 + addendum 2026-11-16): selfcheck (scenarios + mutants),
dynamics N1|N10, replay N1|N10, open-probe. Run with .venv-pg:
  a5_run.py selfcheck | dynamics N1 | dynamics N10 | replay N1 | replay N10"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import a2_pg as G
import a4_lsm as L4
import a5_lsm as L
from a2_dyn import BATCH, OPS, Model, checkpoint, load_extra, make_ops, ops_sha256
from a2_runs import DATA, RES, log, sha16
from a4_run import A4System, Control, moving_update, wide_payload

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "A5_protocol.md"
PY = ROOT / ".venv-pg" / "Scripts" / "python.exe"
VACUUMS = (240_000, 480_000)
CP_EVERY = 50_000
F_MAJOR = 0.1
T_RUN = 30_000
FORCED_N10 = 150_000
STREAM_SHA = {"N1": "1f685ebe", "N10": "1bdeb7fb"}
MUTANTS = ("a", "b", "c", "d", "e", "f", "g", "i", "k", "k2", "l", "z")
MUST_CATCH = tuple(m for m in MUTANTS if m != "k")  # (k) is equivalent (addendum §1), (k') = k2 replaces it


def check_protocol() -> None:
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != (RES / "a5_protocol.sha256").read_text().split()[0]:
        raise SystemExit(f"A5 protocol changed: {sha}")


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_mb() -> dict | None:
    """Peak / current working set of this process (fixed: explicit HANDLE types, K32GetProcessMemoryInfo)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    f = k32.K32GetProcessMemoryInfo
    f.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    f.restype = wt.BOOL
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not f(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return {"peak_mb": c.PeakWorkingSetSize / 2**20, "current_mb": c.WorkingSetSize / 2**20}


def clean(x):
    """Drop private fields (numpy witness arrays) before JSON."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def read_slice(conn, table: str, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of heap blocks [lo, hi): Tid Range Scan, binary COPY of k and the ctid page (§2.3)."""
    sql = (f"COPY (SELECT k, (ctid::text::point)[0]::int4 FROM {table} WHERE ctid >= '({lo},0)'::tid "
           f"AND ctid < '({hi},0)'::tid) TO STDOUT (FORMAT binary)")
    buf = bytearray()
    with conn.cursor() as cur, cur.copy(sql) as cp:
        for data in cp:
            buf += data
    dt = np.dtype([("n", ">i2"), ("l1", ">i4"), ("k", "V16"), ("l2", ">i4"), ("p", ">i4")])
    n = (len(buf) - 19 - 2) // dt.itemsize
    arr = np.frombuffer(bytes(buf), dt, count=n, offset=19)
    keys = np.frombuffer(arr["k"].tobytes(), np.uint8).reshape(n, 16) if n else np.zeros((0, 16), np.uint8)
    return keys, arr["p"].astype(np.uint32)


class A5System(A4System):
    def __init__(self, name: str, table: str, seq: str, d: Path, lsm: L.AddressLSM) -> None:
        super().__init__(name, table, seq, d, lsm)
        self.kind = "A5"

    def scan(self, lo: int, hi: int):
        keys, pages = read_slice(self.conn, self.table, lo, hi)
        return keys, pages, {"read_bytes": (hi - lo) * 8192}

    def p0(self) -> int:
        return int(self.conn.execute("SELECT pg_relation_size(%s)", (self.table,)).fetchone()[0]) // 8192

    def maintain(self, live_rows: int | None = None) -> list[dict]:
        return self.lsm.boundary(self.scan, self.p0)

    def final_round(self) -> list[dict]:
        return self.lsm.run_round_now(self.scan, self.p0)

    def sizes(self) -> dict:
        b = self.lsm.nbytes()
        sp = self.lsm.spill_bytes()
        return {"address_main": b, "address_total": b + sp, "spill_bytes": sp,
                "table_bytes": G.relinfo(self.conn, self.table)["table_bytes"], "levels": self.lsm.levels(),
                "runs": [[r.id, r.prio, r.tier, r.n, r.pre] for r in self.lsm.runs],
                "tail_committed_keys": len(self.lsm.tail.committed), "job": None if self.lsm.job is None else
                [self.lsm.job.tier, self.lsm.job.done, 1 << self.lsm.job.bits],
                "round": None if self.lsm.round is None else [self.lsm.round["phase"], self.lsm.round["j"]]}


def setup_a5(conn, table: str, n: int, ref: Path, d: Path, **kw) -> A5System:
    G.load(conn, table, 0, n, ref)
    conn.execute(f"DROP SEQUENCE IF EXISTS {table}_seq")
    conn.execute(f"CREATE SEQUENCE {table}_seq START {n + 1}")
    keys, pages, _ = G.read_kp(conn, table)
    lsm = L.AddressLSM.create(d, keys, pages, n, **kw)
    return A5System(table, table, f"{table}_seq", d, lsm)


def state_ok(s, model: Model, kb: bytes) -> bool:
    row = s.lookup(kb)
    st = model.state.get(kb)
    return (row is None) if st is None else (row is not None and row[1] == st[0] and sha16(row[3]) == st[1])


def witness_keys(s: A5System, ev: dict, rng: random.Random, n: int = 200) -> list[bytes]:
    """§3: touched keys of the step; scan step: keys on the first and last page of the slice (R5); build step: keys
    of the section plus keys of the section changed after W (tail)."""
    k = ev.get("_keys")
    out: list[bytes] = []
    if ev["kind"] == "scan_step" and k is not None and len(k):
        pg = ev["_pages"]
        edge = np.flatnonzero((pg == ev["lo"]) | (pg == ev["hi"] - 1))
        out = [bytes(k[i]) for i in edge[:n]]
    elif k is not None and len(k):
        idx = rng.sample(range(len(k)), min(n if ev["kind"] != "build_step" else 150, len(k)))
        out = [bytes(k[i]) for i in idx]
    if ev["kind"] == "build_step":
        tail = list(s.lsm.tail.committed)
        cand = rng.sample(tail, min(3000, len(tail)))
        out += [kb for kb in cand if L.sect_int(s.lsm.hz.one(kb), s.lsm.bits_base) == ev["section"]][:50]
    if ev["kind"] == "round_start" and "seal" in ev:
        out = witness_keys(s, ev["seal"], rng, n)
    return out


def check_witnesses(s: A5System, model: Model, keys: list[bytes]) -> int:
    return sum(1 for kb in keys if (kb in model.state or kb in model.deleted) and not state_ok(s, model, kb))


# ------------------------------------------------------------------ scenarios and mutants (§4)
def collision_check(mutant: str | None) -> dict:
    """R4: artificial collisions of h (only the top 6 bits kept); versions must be grouped by the full key."""
    d = DATA / f"a5_coll_{mutant or 'ok'}"
    rng = np.random.default_rng(7)
    keys = rng.integers(0, 256, (2000, 16), dtype=np.uint8)
    pages = rng.integers(0, 500, 2000).astype(np.uint32)
    lsm = L.AddressLSM.create(d, keys, pages, 2000, t_run=50, f_major=1e9, sigma=256, s_base_bits=2, mutant=mutant,
                              h_trunc=6)
    expect = {bytes(k): int(p) for k, p in zip(keys, pages)}
    kl = list(expect)
    prng = random.Random(9)
    changed = set()
    for _ in range(4):
        lsm.batch_no += 1
        for kb in prng.sample(kl, 300):
            lsm.opseq += 1
            pg = prng.randrange(500, 1000)
            lsm.tail.append(L4.U_, kb, pg, lsm.opseq)
            expect[kb] = pg
            changed.add(kb)
        lsm.ops_total += lsm.tail.commit(lsm.batch_no)
        lsm.seal()
    while lsm.job is not None or lsm.maybe_start_merge():
        lsm.merge_step()
    merged = [r for r in lsm.runs if r.tier == 1]
    ent = merged[0].entries() if merged else {}
    bad_merged = sum(1 for kb in changed if ent.get(kb) != expect[kb])
    bad_cand = sum(1 for kb in kl if expect[kb] not in lsm.candidates(kb, False))
    lsm.close()
    return {"scenario": "R4: collisions of h (6-bit h), merge by full key", "ok": bad_merged == 0 and bad_cand == 0,
            "changed": len(changed), "bad_merged": bad_merged, "bad_candidates": bad_cand,
            "distinct_h": len({lsm.hz.one(kb) for kb in kl})}


def scenario_suite(mutant: str | None = None) -> dict:
    conn = G.connect()
    d = DATA / f"a5_scn_{mutant or 'ok'}"
    kw = dict(t_run=50, f_major=1e9, sigma=256, s_base_bits=3, mutant=mutant)
    s = setup_a5(conn, "a5_scn", 10_000, DATA / "a5_scn", d, **kw)
    conn.execute("ALTER TABLE a5_scn ALTER COLUMN payload SET STORAGE PLAIN")  # no TOAST: wide rows really move
    keys = [bytes(k) for k in np.load(DATA / "a5_scn.keys")]
    model = Model(keys, [bytes(x) for x in np.load(DATA / "a5_scn.dig")])
    extra = load_extra(DATA / "dyn.copy")[-6000:]
    pay = iter(extra)
    res: list[dict] = []
    mrng = random.Random(5)
    order = list(keys)
    random.Random(3).shuffle(order)
    fresh = iter(order)

    def untouched(n):
        out = []
        for k in fresh:
            if model.state.get(k, [0, 0, ""])[2] == "base":
                out.append(k)
                if len(out) == n:
                    break
        return out

    def run(label, ops, expect, check, *, rollback=False, record=True):
        got = s.apply(ops, rollback=rollback)
        if not rollback:
            for (kind, kb, pl), g in zip(ops, got):
                if g in ("ins", "upd", "del"):
                    model.apply(0, kind, kb, pl)
        ok = got == expect and all(state_ok(s, model, kb) for kb in check)
        if record:
            res.append({"scenario": label, "ok": bool(ok), "outcomes": got[:5]})
        return ok

    def check_all(label, ks, **info):
        bad = [kb for kb in ks if not state_ok(s, model, kb)]
        res.append({"scenario": label, "ok": not bad, "checked": len(ks), "bad": len(bad)} | info)
        return not bad

    p = lambda: next(pay)[1]  # noqa: E731

    def wide(label, ks):
        return run(label, [("U", k, wide_payload(mrng)) for k in ks], ["upd"] * len(ks), [], record=False)

    def filler_seal(n=1):
        run("filler UPDATE", [("U", k, p()) for k in untouched(n)], ["upd"] * n, [], record=False)
        s.lsm.seal()

    def finish_merges():
        while s.lsm.job is not None or s.lsm.maybe_start_merge():
            s.lsm.merge_step()

    def empty_tier(t):
        guard = 0
        while any(r.tier == t for r in s.lsm.runs) or s.lsm.job is not None:
            filler_seal()
            finish_merges()
            guard += 1
            if guard > 200:
                raise RuntimeError("tier does not empty")

    # ---- R1 counterexample (A4): key committed only in the tail, moved by a rolled-back UPDATE / DELETE
    f1, f2 = next(pay), next(pay)
    run("R1: INSERT + COMMIT (tail only)", [("I", f1[0], f1[1]), ("I", f2[0], f2[1])], ["ins", "ins"], [f1[0], f2[0]])
    moved, _ = moving_update(s, model, f1[0], mrng, rollback=True)
    check_all("R1: UPDATE moved to another page, then ROLLBACK", [f1[0]], moved_page_differs=moved)
    res[-1]["ok"] &= moved
    run("R1: DELETE, then ROLLBACK", [("D", f2[0], None)], ["del"], [f2[0]], rollback=True)
    # ---- A4 basics in three phases
    for phase in ("tail", "after_seal", "after_merge"):
        k1, k2, k3, k4, k5 = untouched(5)
        ghost = next(pay)[0]
        run(f"{phase}: UPDATE twice", [("U", k1, p()), ("U", k1, p())], ["upd", "upd"], [k1])
        run(f"{phase}: three UPDATEs -> v = 4", [("U", k2, p()), ("U", k2, p()), ("U", k2, p())], ["upd"] * 3, [k2])
        run(f"{phase}: UPDATE -> DELETE", [("U", k3, p()), ("D", k3, None)], ["upd", "del"], [k3])
        run(f"{phase}: DELETE -> INSERT same key", [("D", k4, None), ("I", k4, p())], ["del", "ins"], [k4])
        run(f"{phase}: repeated DELETE", [("D", k5, None), ("D", k5, None)], ["del", "none"], [k5])
        run(f"{phase}: UPDATE of an absent key", [("U", ghost, p())], ["none"], [ghost])
        victims, movers = untouched(100), untouched(100)
        run(f"{phase}: DELETE 100 + UPDATE 100", [("D", k, None) for k in victims] + [("U", k, p()) for k in movers],
            ["del"] * 100 + ["upd"] * 100, victims[:30] + movers[:30])
        conn.execute("VACUUM a5_scn")
        newk = [next(pay) for _ in range(120)]
        later = untouched(50)
        run(f"{phase}: after VACUUM: INSERT 120 + UPDATE 50", [("I", k, pl) for k, pl in newk] +
            [("U", k, p()) for k in later], ["ins"] * 120 + ["upd"] * 50, [k for k, _ in newk] + later)
        check_all(f"{phase}: R1 keys", [f1[0], f2[0]])
        if phase == "tail":
            s.lsm.seal()
        elif phase == "after_seal":
            guard = 0
            while not any(r.tier >= 1 for r in s.lsm.runs):
                filler_seal()
                finish_merges()
                guard += 1
            res.append({"scenario": "tier merge", "ok": True, "runs": [[r.id, r.prio, r.tier] for r in s.lsm.runs]})
    # ---- mutant (b) witness: a key changed in three runs on different pages, then a tier merge
    empty_tier(0)
    w = untouched(1)[0]
    pages_w = [s.lsm.candidates(w, False)[0]]
    for _ in range(3):
        moving_update(s, model, w, mrng, rollback=False)
        pages_w.append(s.lsm.tail.committed.get(w))
        s.lsm.seal()
    filler_seal()
    finish_merges()
    merged = [r for r in s.lsm.runs if r.tier == 1 and w in r.entries()]
    check_all("witness after a tier merge (3 versions on different pages)", [w], pages=pages_w,
              distinct_pages=len(set(pages_w)) == 4,
              merged_points_to_newest=bool(merged) and merged[0].entries().get(w) == pages_w[-1])
    # ---- lookup during a multi-section tier merge (mutant (z))
    empty_tier(0)
    groups = []
    for _ in range(4):
        g = untouched(70)
        wide("70 wide UPDATEs", g)
        groups += g
        s.lsm.seal()
    started = s.lsm.maybe_start_merge()
    sections = 1 << s.lsm.job.bits if started else 0
    s.lsm.merge_step()
    check_all("lookup during a tier merge (ready + not-ready sections)", groups, sections=sections,
              done=s.lsm.job.done if s.lsm.job else None)
    res[-1]["ok"] &= sections >= 2
    finish_merges()
    check_all("after the multi-section merge", groups)
    # ---- R2 (Codex 234): merge -> moved witness -> seal X -> finish merge -> next merges -> lookup; mutant (k')
    guard = 0
    while any(r.tier == 1 for r in s.lsm.runs) or any(r.tier == 0 for r in s.lsm.runs):
        filler_seal()
        finish_merges()
        guard += 1
        if guard > 300:
            raise RuntimeError("tiers do not empty")
    w2 = untouched(1)[0]
    era = []
    for i in range(4):
        g = untouched(70)
        wide("70 wide UPDATEs", g)
        if i == 0:
            moving_update(s, model, w2, mrng, rollback=False)
            era.append(s.lsm.tail.committed.get(w2))
        s.lsm.seal()
    s.lsm.maybe_start_merge()
    s.lsm.merge_step()  # M1 is now half done
    moving_update(s, model, w2, mrng, rollback=False)
    era.append(s.lsm.tail.committed.get(w2))
    s.lsm.seal()  # run X: newer than every input of M1
    finish_merges()  # M1 -> O1
    check_all("R2: after the old merge finished", [w2])
    for _ in range(3):
        filler_seal()
    finish_merges()  # X + 3 -> O2
    check_all("R2: after the next merge", [w2])
    n_t2 = sum(1 for r in s.lsm.runs if r.tier == 2)
    while sum(1 for r in s.lsm.runs if r.tier == 2) == n_t2:  # two more tier-1 runs -> O1, O2 merge into tier 2
        filler_seal()
        finish_merges()
    t2 = [r for r in s.lsm.runs if r.tier == 2 and w2 in r.entries()]
    check_all("R2: after the tier-2 merge", [w2], era_pages=era, distinct=len(set(era)) == 2,
              tier2_points_to_newest=bool(t2) and t2[0].entries().get(w2) == era[-1])
    # ---- R4: artificial collisions of h (pure, no SQL)
    res.append(collision_check(mutant))
    # ---- the round (§2.3, §4): pre-W changes, moves during the scan, inserts, deletes, VACUUM, post-W changes
    empty_tier(0)
    s.lsm.t_run = 10**9  # seals in this scenario are explicit: the run composition is part of the mutant (g) witness
    p1 = untouched(40)
    wide("pre-W moved, sealed", p1)
    s.lsm.seal()
    p2 = untouched(20)
    wide("pre-W moved, in the tail at W", p2)
    s.lsm.round_pending = True
    ev0 = s.maintain()
    rd = s.lsm.round
    res.append({"scenario": "round started", "ok": rd is not None and any(e["kind"] == "round_start" for e in ev0),
                "P0": rd["P0"] if rd else None})
    bounds = rd["bounds"]
    edge_pages = sorted({b for b in bounds[:-1]} | {b - 1 for b in bounds[1:]})
    edge = [uuid.UUID(str(r[0])).bytes for r in conn.execute(
        "SELECT k FROM a5_scn WHERE (ctid::text::point)[0]::int4 = ANY(%s)", (edge_pages,)).fetchall()]
    edge = [k for k in edge if model.state.get(k, [0, 0, ""])[2] == "base"][:200]
    moves, inserted, deleted = [], [], []
    scanned_at: dict[int, int] = {}
    step = 0
    while s.lsm.round is not None and s.lsm.round["phase"] == "scan":
        for kb in untouched(8):
            old = s.lsm.candidates(kb, False)[0]
            moving_update(s, model, kb, mrng, rollback=False)
            moves.append((kb, old, s.lsm.tail.committed.get(kb), step))
        ins = [next(pay) for _ in range(5)]
        run("INSERT during the round", [("I", k, pl) for k, pl in ins], ["ins"] * 5, [], record=False)
        inserted += [k for k, _ in ins]
        dl = untouched(3)
        run("DELETE during the round", [("D", k, None) for k in dl], ["del"] * 3, [], record=False)
        deleted += dl
        if step == 4:
            conn.execute("VACUUM a5_scn")
            ins = [next(pay) for _ in range(20)]
            run("INSERT after VACUUM during the round", [("I", k, pl) for k, pl in ins], ["ins"] * 20, [], record=False)
            inserted += [k for k, _ in ins]
        j = s.lsm.round["j"]
        s.maintain()
        scanned_at[j] = step
        step += 1
    p0 = rd["P0"]

    def slice_of(page):
        return None if page >= p0 else int(np.searchsorted(bounds, page, side="right") - 1)

    seen0 = seen2 = 0
    for kb, old, new, at in moves:
        so, sn = slice_of(old), slice_of(new)
        old_after = so is not None and scanned_at.get(so, -1) >= at  # old page scanned after the move: not seen there
        new_before = sn is None or scanned_at.get(sn, 10**9) < at  # new page scanned before the move (or >= P0)
        seen0 += old_after and new_before
        seen2 += (not old_after) and (not new_before)
    ins_high = sum(1 for k in inserted if (s.lsm.tail.committed.get(k) or 0) >= p0)
    # build phase: post-W changes, three post-W seals (mutant (g)), keys left in the tail (mutant (i))
    g1, mid_checked = [], False
    builds = 0
    while s.lsm.round is not None:
        s.maintain()
        builds += 1
        if builds == 2 and not mid_checked:
            mid_checked = True
            check_all("mid-round: pre-W changes (sections partly rebuilt)", p1 + p2)
        if builds in (1, 2, 3):
            g = untouched(30)
            wide("post-W, sealed", g)
            g1 += g
            s.lsm.seal()
        if builds == 4:
            g2 = untouched(20)
            wide("post-W, left in the tail", g2)
    base_sample = untouched(1500)
    check_all("after the round: pre-W changes", p1 + p2)
    check_all("after the round: moved during the scan", [m[0] for m in moves], seen_0_times=seen0, seen_2_times=seen2)
    check_all("after the round: inserted during the round", inserted, on_pages_ge_P0=ins_high)
    check_all("after the round: deleted during the round", deleted)
    check_all("after the round: post-W sealed", g1)
    check_all("after the round: post-W in the tail", g2)
    check_all("after the round: keys on slice edge pages", edge)
    check_all("after the round: untouched base keys", base_sample)
    res.append({"scenario": "round coverage (descriptive)", "ok": True, "moves": len(moves), "seen_0_times": seen0,
                "seen_2_times": seen2, "inserts_on_pages_ge_P0": ins_high})
    s.lsm.t_run = 50
    # ---- R3: restart in the middle of a round with an active tier merge
    empty_tier(0)
    s.lsm.round_pending = True
    s.maintain()
    while s.lsm.round is not None and s.lsm.round["phase"] == "scan":
        s.maintain()
    s.maintain()  # one build step
    r3 = []
    for _ in range(4):
        g = untouched(70)
        wide("post-W wide", g)
        r3 += g
        s.lsm.seal()
    s.maintain()  # merge step 0 (post-W runs) + build step
    state = {"job": s.lsm.job is not None, "round": s.lsm.round is not None}
    s.lsm.close()
    s.lsm = L.AddressLSM.open(d, **kw)
    check_all("R3: after restart mid-round + mid-merge", r3 + p1 + p2 + g1 + g2 + base_sample[:500],
              state_before=state, abandoned=sorted(s.lsm.abandoned))
    finish_merges()
    s.lsm.round_pending = True
    while s.lsm.round_pending or s.lsm.round is not None or s.lsm.job is not None:
        s.maintain()
    check_all("R3: after the next full round", r3 + p1 + p2 + g1 + g2 + base_sample)
    # ---- reader in another instance: refresh vs mutant (c)
    reader = L.AddressLSM.open(d, **kw)
    rs = A5System("reader", "a5_scn", "a5_scn_seq", d, reader)
    rs.reader_mode = True
    rs.reader_refresh = mutant != "c"
    nk = next(pay)
    run("writer INSERT after the reader opened", [("I", nk[0], nk[1])], ["ins"], [nk[0]])
    res.append({"scenario": "reader sees a key committed after it opened", "ok": rs.lookup(nk[0]) is not None})
    reader.close()
    rs.conn.close()
    # ---- refuse to open: a batch without an outcome marker, a torn record
    s.lsm.batch_no += 1
    s.lsm.tail.append(L4.I_, bytes(16), 1, s.lsm.opseq + 1)
    s.lsm.tail.f.write(b"".join(s.lsm.tail._buf))
    s.lsm.tail.f.flush()
    s.lsm.tail._buf = []
    refused = False
    try:
        L.AddressLSM.open(d, ignore_open_batch=(mutant == "d"), **kw).close()
    except L4.RefuseOpen:
        refused = True
    res.append({"scenario": "refuse to open with a batch without outcome", "ok": refused})
    with open(s.lsm.tail.path, "ab") as f:
        f.write(b"\x01\x02\x03")
    torn = False
    try:
        L.AddressLSM.open(d, **kw).close()
    except L4.RefuseOpen:
        torn = True
    res.append({"scenario": "refuse to open with a torn record", "ok": torn})
    s.lsm.close()
    s.conn.close()
    conn.execute("DROP TABLE IF EXISTS a5_scn")
    conn.close()
    return {"mutant": mutant, "all_ok": all(r["ok"] for r in res), "results": res,
            "failed": [r["scenario"] for r in res if not r["ok"]]}


def cmd_selfcheck() -> None:
    out = {}
    r = subprocess.run([str(PY), str(ROOT / "a2_selftest_core.py")], capture_output=True, text=True, cwd=str(ROOT))
    out["core_selftest"] = {"exit": r.returncode, "tail": r.stdout.strip().splitlines()[-1]}
    out["ok"] = scenario_suite(None)
    log("scenarios ok-run:", out["ok"]["all_ok"], out["ok"]["failed"])
    for m in MUTANTS:
        try:
            res = scenario_suite(m)
            out[f"mutant_{m}"] = {"caught": not res["all_ok"], "failed": res["failed"]}
        except Exception as e:  # noqa: BLE001 - a crash is also a catch, recorded as such
            out[f"mutant_{m}"] = {"caught": True, "failed": [f"exception: {type(e).__name__}: {e}"]}
        log("mutant", m, out[f"mutant_{m}"])
    out["mutant_k_note"] = "equivalent under §2.2 (addendum §1); reported, not required"
    out["all_ok"] = (out["core_selftest"]["exit"] == 0 and out["ok"]["all_ok"]
                     and all(out[f"mutant_{m}"]["caught"] for m in MUST_CATCH))
    G.write_json(RES / "a5_selfcheck.json", out)
    log("selfcheck all_ok", out["all_ok"])


# ------------------------------------------------------------------ dynamics (§3, §5)
def open_probe(d: Path, keys: list[bytes], table: str, runs: int = 21) -> dict:
    res = []
    for i in range(runs):
        r = subprocess.run([str(PY), str(ROOT / "a5_run.py"), "open-probe", str(d), table, keys[i].hex()],
                           capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        res.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return {k: G.pct([x[k] for x in res]) for k in res[0] if isinstance(res[0][k], (int, float))}


def cmd_open_probe(d: str, table: str, keyhex: str) -> None:
    t0 = time.perf_counter()
    conn = G.connect()
    t1 = time.perf_counter()
    lsm = L.AddressLSM.open(Path(d))
    t2 = time.perf_counter()
    s = A5System("probe", table, f"{table}_seq", Path(d), lsm)
    s.conn.close()
    s.conn = conn
    row = s.lookup(bytes.fromhex(keyhex))
    t3 = time.perf_counter()
    print(json.dumps({"connect_ms": (t1 - t0) * 1e3, "open_ms": (t2 - t1) * 1e3, "first_answer_ms": (t3 - t2) * 1e3,
                      "found": row is not None, "levels": lsm.levels(), "files": len(list(Path(d).glob("*.adr"))),
                      "tail_records": lsm.tail.offset // L4.LOGREC.size}))


def round_summaries(events: list[dict], vac_ops: tuple[int, ...]) -> list[dict]:
    rounds, cur = [], None
    for e in events:
        if e["kind"] == "round_start":
            cur = {"start_op": e["after_op"], "P0": e["P0"], "start_s": e["s"], "scan_s": 0.0, "build_s": 0.0,
                   "scan_read_bytes": 0, "scan_steps": 0, "build_steps": 0, "max_scan_s": 0.0, "max_build_s": 0.0,
                   "forced": e.get("forced_round", False), "final": e.get("final", False),
                   "delta_bytes_at_start": e.get("published_bytes")}
        elif cur is not None and e["kind"] == "scan_step":
            cur["scan_s"] += e["s"]
            cur["scan_read_bytes"] += e["read_bytes"]
            cur["scan_steps"] += 1
            cur["max_scan_s"] = max(cur["max_scan_s"], e["s"])
        elif cur is not None and e["kind"] == "build_step":
            end_s = e["round_end"]["s"] if "round_end" in e else 0.0
            cur["build_s"] += e["s"] - end_s
            cur["build_steps"] += 1
            cur["max_build_s"] = max(cur["max_build_s"], e["s"])
            if "round_end" in e:
                cur.update({k: v for k, v in e["round_end"].items() if k != "s"})
                cur["end_s"] = end_s
                cur["end_op"] = e["after_op"]
                cur["B_s"] = cur["scan_s"] + cur["build_s"]
                cur["vacuum_in_window"] = [v for v in vac_ops if cur["start_op"] <= v <= cur["end_op"]]
                rounds.append(cur)
                cur = None
    return rounds


def cmd_dynamics(size: str) -> None:
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower()
    conn = G.connect()
    ref = DATA / f"ref_a5_{tag}"
    log(size, "load A5 table")
    a = setup_a5(conn, f"a5_{tag}", n, ref, DATA / f"a5_{tag}", t_run=T_RUN, f_major=F_MAJOR)
    log(size, "load control table")
    G.load(conn, f"c5_{tag}", 0, n, ref)
    conn.execute(f"DROP SEQUENCE IF EXISTS c5_{tag}_seq")
    conn.execute(f"CREATE SEQUENCE c5_{tag}_seq START {n + 1}")
    conn.execute(f"CREATE UNIQUE INDEX c5_{tag}_bt ON c5_{tag} USING btree (k)")
    conn.execute(f"ANALYZE c5_{tag}")
    c = Control(f"c5_{tag}", f"c5_{tag}", f"c5_{tag}_seq")
    keys = [bytes(k) for k in np.load(f"{ref}.keys")]
    digs = [bytes(x) for x in np.load(f"{ref}.dig")]
    extra = load_extra(DATA / "dyn.copy")
    ever = np.concatenate([np.load(f"{ref}.keys"), np.load(DATA / "dyn.keys")])
    ops = make_ops(keys, extra, 250_000, 150_000, 100_000, 13)
    out = {"size": size, "rows": n, "ops_sha256": ops_sha256(ops), "bits_base": a.lsm.bits_base, "checkpoints": [],
           "batches": [], "events": [], "witness_errors": 0, "s3_batch_violations": [], "vacuum": {}}
    out["stream_matches_a4"] = out["ops_sha256"].startswith(STREAM_SHA[size])
    log(size, "stream sha256", out["ops_sha256"], "matches A4:", out["stream_matches_a4"], "S_base", 1 << a.lsm.bits_base)
    model = Model(keys, digs)
    del keys, digs
    systems = (a, c)
    cp = 0
    out["checkpoints"].append(checkpoint(cp, "base", 0, systems, model, ever, 0))
    t_ops = {a.name: 0.0, c.name: 0.0}
    t_maint = 0.0
    t_vac = {a.name: 0.0, c.name: 0.0}
    wrng = random.Random(77)
    half = (1 << a.lsm.bits_base) // 2 - 1
    for start in range(0, OPS, BATCH):
        batch = ops[start: start + BATCH]
        expected = ["ins" if k == "I" else "upd" if k == "U" else "del" for k, _, _ in batch]
        t = time.perf_counter()
        got = a.apply(batch)
        ta = time.perf_counter() - t
        t = time.perf_counter()
        gotc = c.apply(batch)
        t_ops[c.name] += time.perf_counter() - t
        a.errors += sum(1 for x, y in zip(got, expected) if x != y)
        c.errors += sum(1 for x, y in zip(gotc, expected) if x != y)
        for i, (kind, kb, payload) in enumerate(batch):
            model.apply(start + i, kind, kb, payload)
        t_ops[a.name] += ta
        op_no = start + BATCH
        if size == "N10" and op_no == FORCED_N10:
            a.lsm.round_pending = True
            log(size, "forced round requested at op", op_no)
        published_before = a.lsm.nbytes()
        tm = time.perf_counter()
        evs = a.maintain()
        tmaint = time.perf_counter() - tm
        t_maint += tmaint
        mid = False
        for ev in evs:
            if ev["kind"] == "round_start":
                ev["forced_round"] = size == "N10"
                ev["published_bytes"] = published_before
                log(size, "round start at op", op_no, "P0", ev["P0"])
            wk = witness_keys(a, ev, wrng)
            ev["witness_errors"] = check_witnesses(a, model, wk)
            ev["witnesses"] = len(wk)
            out["witness_errors"] += ev["witness_errors"]
            if ev["kind"] == "build_step" and ev["section"] == half:
                mid = True
            if ev["kind"] == "build_step" and "round_end" in ev:
                ev["round_end"]["live_rows"] = len(model.state)
                log(size, "round end at op", op_no, {k: ev["round_end"][k] for k in ("rows", "dups", "steps", "window_ops")})
            out["events"].append(clean(ev))
        nb = a.lsm.nbytes()
        per = nb / len(model.state)
        sp = a.lsm.spill_bytes()
        out["batches"].append([op_no, round(ta, 4), round(tmaint, 4), nb, round(per, 4), a.lsm.levels(), sp,
                               "+".join(e["kind"] for e in evs)])
        if per > 8.0:
            out["s3_batch_violations"].append([op_no, per])
        if op_no in VACUUMS:
            for s_ in systems:
                tv = time.perf_counter()
                conn.execute(f"VACUUM {s_.table}")
                t_vac[s_.name] += time.perf_counter() - tv
        if mid:
            cp += 1
            out["checkpoints"].append(checkpoint(cp, "mid_round", op_no, systems, model, ever, 0))
        if op_no % CP_EVERY == 0:
            cp += 1
            rec = checkpoint(cp, "every_50k", op_no, systems, model, ever, 0)
            rec["candidates_hist"] = dict(a.cand_hist)
            a.cand_hist = {}
            rec["rss"] = rss_mb()
            out["checkpoints"].append(rec)
            log(size, "ops", op_no, {k: round(v, 1) for k, v in t_ops.items()}, "maint", round(t_maint, 1),
                "levels", a.lsm.levels(), "max pause", round(max(b[2] for b in out["batches"]), 3))
    # restart, reopen from disk, checkpoint, open probe
    log(size, "restart")
    G.pg_ctl("restart")
    conn = G.connect()
    state_before = {"job": a.lsm.job is not None, "round": a.lsm.round is not None}
    a.lsm.close()
    t = time.perf_counter()
    a.lsm = L.AddressLSM.open(a.dir, t_run=T_RUN, f_major=F_MAJOR)
    out["reopen_s"] = time.perf_counter() - t
    out["restart_state"] = state_before | {"abandoned": a.lsm.abandoned}
    a.reopen()
    c.reopen()
    cp += 1
    out["checkpoints"].append(checkpoint(cp, "after_restart", OPS, systems, model, ever, 0))
    out["open_probe"] = open_probe(a.dir, model.live[:21], a.table)
    log(size, "open probe", out["open_probe"].get("open_ms"))
    # final round: steps back to back (§3, R7)
    published_before = a.lsm.nbytes()
    t = time.perf_counter()
    fevs = a.final_round()
    out["final_round_continuous_s"] = time.perf_counter() - t
    for ev in fevs:
        if ev["kind"] == "round_start":
            ev["final"] = True
            ev["published_bytes"] = published_before
        if ev["kind"] == "build_step" and "round_end" in ev:
            ev["round_end"]["live_rows"] = len(model.state)
    out["final_events"] = [clean(e) for e in fevs]
    cp += 1
    out["checkpoints"].append(checkpoint(cp, "after_final_round", OPS, systems, model, ever, 0))
    out["ops_time_s"] = t_ops
    out["maintenance_s"] = t_maint
    out["vacuum_s"] = t_vac
    out["errors"] = {s_.name: s_.errors for s_ in systems}
    out["full_cost_us_per_op"] = {a.name: (t_ops[a.name] + t_maint + t_vac[a.name]) / OPS * 1e6,
                                  c.name: (t_ops[c.name] + t_vac[c.name]) / OPS * 1e6}
    pauses = [b[2] for b in out["batches"]]
    out["boundary_pause_s"] = {"max": max(pauses), "p99": float(np.percentile(pauses, 99)),
                               "p95": float(np.percentile(pauses, 95)), "sum": sum(pauses),
                               "nonzero": sum(1 for x in pauses if x > 0.001)}
    by_kind: dict[str, list[float]] = {}
    for e in out["events"]:
        by_kind.setdefault(e["kind"], []).append(e["s"])
    out["step_s_by_kind"] = {k: {"n": len(v), "max": max(v), "p99": float(np.percentile(v, 99)), "sum": sum(v)}
                             for k, v in by_kind.items()}
    bt = [b[1] + b[2] for b in out["batches"]]
    out["batch_time_s"] = {"p50": float(np.percentile(bt, 50)), "p95": float(np.percentile(bt, 95)),
                           "p99": float(np.percentile(bt, 99)), "max": max(bt)}
    out["max_client_stop_s"] = {"address_pause": max(pauses), "vacuum": {k: v for k, v in t_vac.items()}}
    out["rounds"] = round_summaries(out["events"], VACUUMS)
    out["final_round"] = (round_summaries(out["final_events"], ()) or [None])[0]
    merges = [e for e in out["events"] if e["kind"] == "merge_step"]
    out["merge_work"] = {"steps": len(merges), "max_keys_in": max((e["keys_in"] for e in merges), default=0),
                         "max_keys_out": max((e["keys_out"] for e in merges), default=0),
                         "finished": sum(1 for e in merges if e.get("finished"))}
    out["peaks"] = {"published_bytes": a.lsm.peak_published, "layer_bytes": a.lsm.peak_layer,
                    "spill_bytes": a.lsm.spill_peak,
                    "spill_share_of_batches": sum(1 for b in out["batches"] if b[6] > 0) / len(out["batches"])}
    out["rss"] = rss_mb()
    out["zlib_probe"] = a.lsm.zlib_probe
    a.lsm.close()
    for t_ in (a.table, c.table):
        conn.execute(f"DROP TABLE IF EXISTS {t_}")
    G.write_json(RES / f"a5_dynamics_{tag}.json", out)
    log(size, "done errors", out["errors"], "witness", out["witness_errors"], "S3 batch violations",
        len(out["s3_batch_violations"]), "max pause", round(out["boundary_pause_s"]["max"], 3))


# ------------------------------------------------------------------ replay of a natural cycle without SQL (§5, R6)
def cmd_replay(size: str) -> None:
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower()
    keys_arr = np.load(DATA / f"ref_a4_{tag}.keys")
    n_pages = {"N1": 76_232, "N10": 762_888}[size]
    rng = np.random.default_rng(13)
    k_cycle = int(F_MAJOR * n)
    cap = n + k_cycle * 2
    tkeys = np.zeros((cap, 16), np.uint8)
    tkeys[:n] = keys_arr
    tpages = np.zeros(cap, np.uint32)
    tpages[:n] = rng.integers(0, n_pages, size=n).astype(np.uint32)
    alive = np.zeros(cap, bool)
    alive[:n] = True
    live_pos = np.arange(n, dtype=np.int64)
    n_live, n_used = n, n
    d = DATA / f"a5_replay_{tag}"
    lsm = L.AddressLSM.create(d, keys_arr, tpages[:n], n, t_run=T_RUN, f_major=F_MAJOR)
    del keys_arr
    prng = random.Random(13)
    top = n_pages + 100_000

    def scan_fn(lo, hi):
        m = alive[:n_used] & (tpages[:n_used] >= lo) & (tpages[:n_used] < hi)
        idx = np.flatnonzero(m)
        return tkeys[idx], tpages[idx], {"read_bytes": 0}

    def p0_fn():
        return top

    bounds_work, t_log = [], 0.0
    ops_done = 0
    round_seen = False
    while True:
        t = time.perf_counter()
        lsm.batch_no += 1
        for _ in range(BATCH):
            lsm.opseq += 1
            r = prng.random()
            if r < 0.5:
                if n_used >= cap:
                    raise RuntimeError("replay capacity")
                kb = prng.randbytes(16)
                tkeys[n_used] = np.frombuffer(kb, np.uint8)
                pg = prng.randrange(top)
                tpages[n_used] = pg
                alive[n_used] = True
                if n_live < len(live_pos):
                    live_pos[n_live] = n_used
                else:
                    live_pos = np.concatenate([live_pos, np.zeros(len(live_pos) // 4 + 1, np.int64)])
                    live_pos[n_live] = n_used
                n_live += 1
                n_used += 1
                lsm.tail.append(L4.I_, kb, pg, lsm.opseq)
            elif r < 0.8:
                i = int(live_pos[prng.randrange(n_live)])
                pg = prng.randrange(top)
                tpages[i] = pg
                lsm.tail.append(L4.U_, tkeys[i].tobytes(), pg, lsm.opseq)
            else:
                j = prng.randrange(n_live)
                i = int(live_pos[j])
                live_pos[j] = live_pos[n_live - 1]
                n_live -= 1
                alive[i] = False
                lsm.tail.append(L4.D_, tkeys[i].tobytes(), 0, lsm.opseq)
        lsm.ops_total += lsm.tail.commit(lsm.batch_no)
        t_log += time.perf_counter() - t
        ops_done += BATCH
        t = time.perf_counter()
        evs = lsm.boundary(scan_fn, p0_fn)
        wall = time.perf_counter() - t
        read_s = sum(e.get("s_read", 0.0) for e in evs)  # the in-memory stand-in for the PostgreSQL pass
        kinds = [e["kind"] for e in evs]
        bounds_work.append({"op": ops_done, "work_s": wall - read_s, "kinds": kinds,
                            "seal_s": sum(e["s"] for e in evs if e["kind"] == "seal"),
                            "merge_s": sum(e["s"] for e in evs if e["kind"] == "merge_step"),
                            "round_start_s": sum(e["s"] for e in evs if e["kind"] == "round_start"),
                            "scan_add_s": sum(e.get("s_add", 0.0) for e in evs if e["kind"] == "scan_step"),
                            "build_s": sum(e["s"] - (e["round_end"]["s"] if "round_end" in e else 0.0)
                                           for e in evs if e["kind"] == "build_step"),
                            "round_end_s": sum(e["round_end"]["s"] for e in evs if "round_end" in e),
                            "merge_read": sum(e.get("read_bytes", 0) for e in evs if e.get("finished")),
                            "merge_written": sum(e.get("written_bytes", 0) for e in evs if e.get("finished")),
                            "seal_written": sum(e["run"]["adr_bytes"] + e["run"]["log_bytes"] for e in evs
                                                if e["kind"] == "seal")})
        if any(k == "round_start" for k in kinds):
            round_seen = True
        if round_seen and lsm.round is None and not lsm.round_pending and lsm.job is None:
            break
        if ops_done > 3 * k_cycle:
            raise RuntimeError("no natural round in the replay")
    # first term (§5, R8): seals and merge steps are a rate per operation; round start (forced seal) and round end
    # happen once per cycle of K operations; scan and build steps are B (measured in PostgreSQL)
    per_op = sum(b["seal_s"] + b["merge_s"] for b in bounds_work) / ops_done
    per_cycle = sum(b["round_start_s"] + b["round_end_s"] for b in bounds_work) / k_cycle
    per_op_b = sum(b["seal_written"] + b["merge_read"] + b["merge_written"] for b in bounds_work) / ops_done
    multi = [b for b in bounds_work if len(b["kinds"]) >= 2]
    out = {"size": size, "cycle_ops": k_cycle, "ops_done": ops_done, "bits_base": lsm.bits_base,
           "first_term_us_per_op": (per_op + per_cycle) * 1e6, "seal_merge_us_per_op": per_op * 1e6,
           "round_start_end_us_per_op": per_cycle * 1e6,
           "first_term_bytes_per_op": per_op_b, "log_us_per_op": t_log / ops_done * 1e6,
           "replay_build_s": sum(b["build_s"] for b in bounds_work),
           "replay_scan_add_s": sum(b["scan_add_s"] for b in bounds_work),
           "max_boundary_work_s": max(b["work_s"] for b in bounds_work),
           "max_boundary_work_at": max(bounds_work, key=lambda b: b["work_s"])["op"],
           "max_multi_kind_work_s": max((b["work_s"] for b in multi), default=0.0),
           "multi_kind_boundaries": len(multi),
           "max_by_kind": {k: max((b[k] for b in bounds_work), default=0.0) for k in
                           ("seal_s", "merge_s", "round_start_s", "scan_add_s", "build_s", "round_end_s")},
           "seals": sum(b["kinds"].count("seal") for b in bounds_work),
           "merge_steps": sum(b["kinds"].count("merge_step") for b in bounds_work),
           "levels_end": lsm.levels(), "tiers_end": [r.tier for r in lsm.runs],
           "peak_published_bytes": lsm.peak_published, "peak_layer_bytes": lsm.peak_layer, "rss": rss_mb()}
    lsm.close()
    G.write_json(RES / f"a5_replay_{tag}.json", out)
    log("replay", size, {k: out[k] for k in ("first_term_us_per_op", "first_term_bytes_per_op", "max_boundary_work_s",
                                              "max_multi_kind_work_s", "seals", "merge_steps", "levels_end")})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "open-probe":
        cmd_open_probe(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    check_protocol()
    if cmd == "replay":
        cmd_replay(sys.argv[2])
        sys.exit(0)
    import a2 as _a2
    with _a2.Server():
        G.connect().execute("CREATE EXTENSION IF NOT EXISTS pageinspect")
        if cmd == "selfcheck":
            cmd_selfcheck()
        elif cmd == "dynamics":
            cmd_dynamics(sys.argv[2])
        else:
            raise SystemExit(__doc__)
