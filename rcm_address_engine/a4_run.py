"""A4 runs (protocol A4_protocol.md edition 2): selfcheck (scenarios + mutants), dynamics N1|N10, replay N1|N10,
open-probe. Run with .venv-pg:  a4_run.py selfcheck | dynamics N1 | dynamics N10 | replay N1 | replay N10"""

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
import psycopg

import a2_core as C
import a2_pg as G
import a4_lsm as L
from a2_dyn import BATCH, OPS, Model, checkpoint, load_extra, make_ops, ops_sha256
from a2_runs import DATA, RES, absent_keys, log, sha16

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "A4_protocol.md"
PY = ROOT / ".venv-pg" / "Scripts" / "python.exe"
VACUUMS = (240_000, 480_000)
CP_EVERY = 50_000
F_MAJOR = 0.1
T_RUN = 30_000
COLS = "ctid, v, seq, payload"


def check_protocol() -> None:
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != (RES / "a4_protocol.sha256").read_text().split()[0]:
        raise SystemExit(f"A4 protocol changed: {sha}")


def peak_rss_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def page_of_ctid(ctid: str) -> int:
    return int(ctid[1:ctid.index(",")])


# ------------------------------------------------------------------ A4 system on PostgreSQL
class Query:
    """Builds the candidate-pages lookup (§2.6): one UNION ALL ... LIMIT 1, newest candidate first."""

    def __init__(self, sys_: "A4System") -> None:
        self.s = sys_
        self._sql: dict[int, str] = {}

    def sql(self, n: int) -> str:
        if n not in self._sql:
            one = f"(SELECT {COLS} FROM {self.s.table} WHERE ctid >= %s::tid AND ctid < %s::tid AND k = %s)"
            self._sql[n] = " UNION ALL ".join([one] * n) + " LIMIT 1"
        return self._sql[n]

    def query(self, kb: bytes, writer: bool = False):
        if self.s.reader_refresh:
            self.s.lsm.tail.refresh() if self.s.reader_mode else None
        pages = self.s.lsm.candidates(kb, writer)
        self.s.last_candidates = len(pages)
        if not pages:
            return None
        k = uuid.UUID(bytes=kb)
        params = []
        for p in pages:
            params += [f"({p},0)", f"({p + 1},0)", k]
        return self.sql(len(pages)), tuple(params)


class A4System:
    def __init__(self, name: str, table: str, seq: str, d: Path, lsm: L.AddressLSM) -> None:
        self.name, self.table, self.seq, self.dir, self.lsm = name, table, seq, d, lsm
        self.kind = "A4"
        self.conn = G.connect()
        self.client = Query(self)
        self.delta = None
        self.errors = 0
        self.reader_mode = False
        self.reader_refresh = True
        self.last_candidates = 0
        self.cand_hist: dict[int, int] = {}
        self.last_pages: dict[bytes, int] = {}

    def reopen(self) -> None:
        self.conn.close()
        self.conn = G.connect()

    def lookup(self, kb: bytes, writer: bool = False):
        q = self.client.query(kb, writer)
        self.cand_hist[self.last_candidates] = self.cand_hist.get(self.last_candidates, 0) + 1
        if q is None:
            return None
        rows = self.conn.execute(q[0], q[1], prepare=True).fetchall()
        return rows[0] if rows else None

    def apply(self, ops, *, rollback: bool = False) -> list[str]:
        """One writer batch (§2.7): records appended before COMMIT, outcome marker + fsync after it."""
        out = []
        lsm = self.lsm
        lsm.batch_no += 1
        t = self.table
        self.last_pages = {}
        try:
            with self.conn.transaction():
                for kind, kb, payload in ops:
                    k = uuid.UUID(bytes=kb)
                    row = self.lookup(kb, writer=True)
                    if kind == "I":
                        if row is not None:
                            out.append("dup")
                            continue
                        r = self.conn.execute(f"INSERT INTO {t} (seq, k, v, payload) VALUES (nextval('{self.seq}'), %s, 1, %s)"
                                              " RETURNING ctid", (k, payload), prepare=True).fetchone()
                        lsm.opseq += 1
                        lsm.tail.append(L.I_, kb, page_of_ctid(r[0]), lsm.opseq)
                        out.append("ins")
                        continue
                    if row is None:
                        out.append("none")
                        continue
                    if kind == "U":
                        r = self.conn.execute(f"UPDATE {t} SET payload = %s, v = v + 1, seq = nextval('{self.seq}') "
                                              "WHERE ctid = %s::tid AND k = %s RETURNING ctid",
                                              (payload, row[0], k), prepare=True).fetchone()
                        if r is None:
                            self.errors += 1
                            out.append("lost")
                            continue
                        lsm.opseq += 1
                        lsm.tail.append(L.U_, kb, page_of_ctid(r[0]), lsm.opseq)
                        self.last_pages[kb] = page_of_ctid(r[0])
                        out.append("upd")
                    else:
                        cur = self.conn.execute(f"DELETE FROM {t} WHERE ctid = %s::tid AND k = %s", (row[0], k),
                                                prepare=True)
                        if cur.rowcount != 1:
                            self.errors += 1
                        lsm.opseq += 1
                        lsm.tail.append(L.D_, kb, 0, lsm.opseq)
                        out.append("del")
                if rollback:
                    raise psycopg.Rollback()
        except Exception:
            lsm.tail.rollback(lsm.batch_no)
            raise
        if rollback:
            lsm.tail.rollback(lsm.batch_no)
        else:
            lsm.ops_since_major += lsm.tail.commit(lsm.batch_no)
        return out

    def maintain(self, live_rows: int):
        """Seal / tiered merges / major compaction when due (§2.2-§2.5), between batches; returns the event or None."""
        due = self.lsm.due()
        if due == "seal":
            return self.lsm.seal()
        if due == "major":
            return self.major(live_rows)
        return None

    def major(self, live_rows: int) -> dict:
        t0 = time.perf_counter()
        self.conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
        keys, pages, _ = G.read_kp(self.conn, self.table)
        self.conn.execute("COMMIT")
        read_s = time.perf_counter() - t0
        read_bytes = G.relinfo(self.conn, self.table)["main_bytes"]
        return self.lsm.major(keys, pages, live_rows, read_bytes, read_s)

    def sizes(self) -> dict:
        b = self.lsm.nbytes()
        return {"address_main": b, "address_total": max(b, self.lsm.peak_coexist_bytes),
                "table_bytes": G.relinfo(self.conn, self.table)["table_bytes"], "levels": len(self.lsm.runs) + 2,
                "runs": [[r.id, r.tier, r.n] for r in self.lsm.runs], "tail_committed_keys": len(self.lsm.tail.committed)}


class Control:
    """B-tree control (A2 §4.5): full unique B-tree, commands by key."""

    def __init__(self, name: str, table: str, seq: str) -> None:
        self.name, self.table, self.seq = name, table, seq
        self.kind = "C"
        self.conn = G.connect()
        self.client = G.Client(self.conn, table, "btree")
        self.delta = None
        self.errors = 0

    def reopen(self) -> None:
        self.conn.close()
        self.conn = G.connect()
        self.client = G.Client(self.conn, self.table, "btree")

    def lookup(self, kb: bytes):
        return self.client.get(kb)

    def apply(self, ops) -> list[str]:
        out, t = [], self.table
        with self.conn.transaction():
            for kind, kb, payload in ops:
                k = uuid.UUID(bytes=kb)
                row = self.lookup(kb)
                if kind == "I":
                    if row is not None:
                        out.append("dup")
                        continue
                    self.conn.execute(f"INSERT INTO {t} (seq, k, v, payload) VALUES (nextval('{self.seq}'), %s, 1, %s)",
                                      (k, payload), prepare=True)
                    out.append("ins")
                    continue
                if row is None:
                    out.append("none")
                    continue
                if kind == "U":
                    cur = self.conn.execute(f"UPDATE {t} SET payload = %s, v = v + 1, seq = nextval('{self.seq}') WHERE k = %s",
                                            (payload, k), prepare=True)
                else:
                    cur = self.conn.execute(f"DELETE FROM {t} WHERE k = %s", (k,), prepare=True)
                if cur.rowcount != 1:
                    self.errors += 1
                out.append("upd" if kind == "U" else "del")
        return out

    def sizes(self) -> dict:
        ix = G.index_size(self.conn, f"{self.table}_bt")
        return {"address_main": ix["main_bytes"], "address_total": ix["total_bytes"],
                "table_bytes": G.relinfo(self.conn, self.table)["table_bytes"]}


def setup_a4(conn, table: str, n: int, ref: Path, d: Path, **kw) -> A4System:
    G.load(conn, table, 0, n, ref)
    conn.execute(f"DROP SEQUENCE IF EXISTS {table}_seq")
    conn.execute(f"CREATE SEQUENCE {table}_seq START {n + 1}")
    keys, pages, _ = G.read_kp(conn, table)
    lsm = L.AddressLSM.create(d, keys, pages, n, t_run=kw.get("t_run", T_RUN), f_major=kw.get("f_major", F_MAJOR),
                              mutant=kw.get("mutant"))
    return A4System(table, table, f"{table}_seq", d, lsm)


def witnesses(s: A4System, model: Model, ev: dict | None, rng: random.Random, n: int = 200) -> int:
    """After a seal / merge: 200 keys touched by the new run (or random live keys after a major) against the model."""
    if ev is None:
        return 0
    if ev["kind"] == "seal" and s.lsm.runs:
        keys = list(s.lsm.runs[0].entries())
    else:
        keys = model.live
    keys = [k for k in keys if k in model.state or k in model.deleted]
    bad = 0
    for kb in rng.sample(keys, min(n, len(keys))):
        row = s.lookup(kb)
        st = model.state.get(kb)
        if st is None:
            bad += row is not None
        else:
            bad += row is None or row[1] != st[0] or sha16(row[3]) != st[1]
    return bad


# ------------------------------------------------------------------ scenarios and mutants (§4)
def wide_payload(rng: random.Random) -> str:
    """~5 KB of random hex; with STORAGE PLAIN on the scenario table two versions cannot share a page."""
    return json.dumps({"pad": rng.randbytes(2500).hex()})


def moving_update(s: "A4System", model: Model, kb: bytes, rng: random.Random, *, rollback: bool, tries: int = 10):
    """UPDATE until the new version lands on another page (HOT can keep it on the same page); returns (moved, outcome)."""
    old = s.lsm.candidates(kb, False)
    for _ in range(tries):
        payload = wide_payload(rng)
        got = s.apply([("U", kb, payload)], rollback=rollback)
        if got != ["upd"]:
            return False, got
        if not rollback:
            model.apply(0, "U", kb, payload)
        new = s.last_pages.get(kb)
        if new is not None and new not in old:
            return True, got
        old = old + [new]
    return False, ["upd"]


def scenario_suite(mutant: str | None = None) -> dict:
    conn = G.connect()
    d = DATA / f"a4_scn_{mutant or 'ok'}"
    s = setup_a4(conn, "a4_scn", 10_000, DATA / "a4_scn", d, t_run=50, f_major=1e9, mutant=mutant)
    conn.execute("ALTER TABLE a4_scn ALTER COLUMN payload SET STORAGE PLAIN")  # no TOAST: wide rows really move
    keys = [bytes(k) for k in np.load(DATA / "a4_scn.keys")]
    model = Model(keys, [bytes(x) for x in np.load(DATA / "a4_scn.dig")])
    extra = load_extra(DATA / "dyn.copy")[-3000:]
    pay = iter(extra)
    res: list[dict] = []

    def run(label, ops, expect, check, *, rollback=False):
        got = s.apply(ops, rollback=rollback)
        if not rollback:
            for (kind, kb, p), g in zip(ops, got):
                if g in ("ins", "upd", "del"):
                    model.apply(0, kind, kb, p)
        ok = got == expect
        for kb in check:
            row = s.lookup(kb)
            st = model.state.get(kb)
            ok &= (row is None) if st is None else (row is not None and row[1] == st[0] and sha16(row[3]) == st[1])
        res.append({"scenario": label, "ok": bool(ok), "outcomes": got})
        return ok

    def check_all(label, ks):
        ok = True
        for kb in ks:
            row = s.lookup(kb)
            st = model.state.get(kb)
            ok &= (row is None) if st is None else (row is not None and row[1] == st[0] and sha16(row[3]) == st[1])
        res.append({"scenario": label, "ok": bool(ok)})

    p = lambda: next(pay)[1]  # noqa: E731
    base = list(keys)
    mrng = random.Random(5)
    # R1 counterexample: key committed only in the tail, moved by a rolled-back UPDATE / DELETE
    fresh1, fresh2 = next(pay), next(pay)
    run("R1: INSERT + COMMIT (tail only)", [("I", fresh1[0], fresh1[1]), ("I", fresh2[0], fresh2[1])], ["ins", "ins"],
        [fresh1[0], fresh2[0]])
    moved, got = moving_update(s, model, fresh1[0], mrng, rollback=True)
    check_all("R1: UPDATE moved to another page, then ROLLBACK", [fresh1[0]])
    res[-1]["moved_page_differs"] = moved
    res[-1]["ok"] &= moved
    run("R1: DELETE, then ROLLBACK", [("D", fresh2[0], None)], ["del"], [fresh2[0]], rollback=True)
    for phase in ("tail", "after_seal", "after_merge"):
        b = [k for k in base if model.state.get(k, [0, 0, ""])[2] == "base"]
        k1, k2, k3, k4, k5 = b[:5]
        ghost = next(pay)[0]
        run(f"{phase}: UPDATE twice", [("U", k1, p()), ("U", k1, p())], ["upd", "upd"], [k1])
        run(f"{phase}: three UPDATEs -> v = 4", [("U", k2, p()), ("U", k2, p()), ("U", k2, p())], ["upd"] * 3, [k2])
        run(f"{phase}: UPDATE -> DELETE", [("U", k3, p()), ("D", k3, None)], ["upd", "del"], [k3])
        run(f"{phase}: DELETE -> INSERT same key", [("D", k4, None), ("I", k4, p())], ["del", "ins"], [k4])
        run(f"{phase}: repeated DELETE", [("D", k5, None), ("D", k5, None)], ["del", "none"], [k5])
        run(f"{phase}: UPDATE of an absent key", [("U", ghost, p())], ["none"], [ghost])
        victims, movers = b[5:105], b[105:205]
        run(f"{phase}: DELETE 100 + UPDATE 100", [("D", k, None) for k in victims] + [("U", k, p()) for k in movers],
            ["del"] * 100 + ["upd"] * 100, victims[:30] + movers[:30])
        conn.execute("VACUUM a4_scn")
        newk = [next(pay) for _ in range(120)]
        run(f"{phase}: after VACUUM: INSERT 120 + UPDATE 50", [("I", k, pl) for k, pl in newk] +
            [("U", k, p()) for k in b[205:255]], ["ins"] * 120 + ["upd"] * 50, [k for k, _ in newk] + b[205:1205])
        check_all(f"{phase}: R1 keys", [fresh1[0], fresh2[0]])
        if phase == "tail":
            ev = s.lsm.seal()
            res.append({"scenario": "seal", "ok": True, "levels": ev["levels"]})
        elif phase == "after_seal":
            while not any(r.tier >= 1 for r in s.lsm.runs):  # seal filler runs until a tier merge has happened
                k = [x for x in base if model.state.get(x, [0, 0, ""])[2] == "base"][0]
                run("filler UPDATE", [("U", k, p())], ["upd"], [k])
                s.lsm.seal()
            res.append({"scenario": "tier merge", "ok": True, "runs": [[r.id, r.tier] for r in s.lsm.runs]})
    # key changed in three runs on different pages, then a tier merge (mutant (b) witness)
    w = [k for k in base if model.state.get(k, [0, 0, ""])[2] == "base"][0]
    def filler_seal():
        k = [x for x in base if model.state.get(x, [0, 0, ""])[2] == "base"][0]
        run("filler UPDATE", [("U", k, p())], ["upd"], [k])
        s.lsm.seal()
    while any(r.tier == 0 for r in s.lsm.runs):  # empty tier 0 first: the witness runs then merge in ONE group
        filler_seal()
    pages_w = [s.lsm.candidates(w, False)[0]]
    for i in range(3):  # three tier-0 runs, each holding a moved version of the witness
        moving_update(s, model, w, mrng, rollback=False)
        pages_w.append(s.lsm.tail.committed.get(w))
        s.lsm.seal()
    filler_seal()  # the 4th tier-0 run: exactly the three witness runs + this one are merged
    merged = [r for r in s.lsm.runs if r.tier >= 1][0]
    in_merged = merged.entries().get(w)
    check_all("witness after tier merges (3 versions on different pages)", [w])
    res[-1]["pages"] = pages_w
    res[-1]["distinct_pages"] = len(set(pages_w)) == 4  # base page + three moved versions
    res[-1]["merged_run_page"] = in_merged
    res[-1]["merged_run_points_to_newest"] = in_merged == pages_w[-1]
    # major compaction in the middle
    s.major(len(model.state))
    check_all("after major compaction", [k for k in base[:2000]] + [fresh1[0], fresh2[0]])
    # reader in another instance: refresh vs mutant (c)
    reader = L.AddressLSM.open(d)
    rs = A4System("reader", "a4_scn", "a4_scn_seq", d, reader)
    rs.reader_mode = True
    rs.reader_refresh = mutant != "c"
    nk = next(pay)
    run("writer INSERT after the reader opened", [("I", nk[0], nk[1])], ["ins"], [nk[0]])
    row = rs.lookup(nk[0])
    res.append({"scenario": "reader sees a key committed after it opened", "ok": row is not None})
    reader.close()
    # refuse to open: a batch without an outcome marker, a torn record (mutant (d) ignores the open batch)
    s.lsm.batch_no += 1
    s.lsm.tail.append(L.I_, bytes(16), 1, s.lsm.opseq + 1)
    s.lsm.tail.f.write(b"".join(s.lsm.tail._buf))
    s.lsm.tail.f.flush()
    s.lsm.tail._buf = []
    refused = False
    try:
        L.AddressLSM.open(d, ignore_open_batch=(mutant == "d")).close()
    except L.RefuseOpen:
        refused = True
    res.append({"scenario": "refuse to open with a batch without outcome", "ok": refused})
    with open(s.lsm.tail.path, "ab") as f:
        f.write(b"\x01\x02\x03")
    torn = False
    try:
        L.AddressLSM.open(d).close()
    except L.RefuseOpen:
        torn = True
    res.append({"scenario": "refuse to open with a torn record", "ok": torn})
    s.lsm.close()
    s.conn.close()
    conn.execute("DROP TABLE IF EXISTS a4_scn")
    conn.close()
    return {"mutant": mutant, "all_ok": all(r["ok"] for r in res), "results": res,
            "failed": [r["scenario"] for r in res if not r["ok"]]}


def cmd_selfcheck() -> None:
    out = {}
    r = subprocess.run([str(PY), str(ROOT / "a2_selftest_core.py")], capture_output=True, text=True, cwd=str(ROOT))
    out["core_selftest"] = {"exit": r.returncode, "tail": r.stdout.strip().splitlines()[-1]}
    out["ok"] = scenario_suite(None)
    log("scenarios ok-run:", out["ok"]["all_ok"], out["ok"]["failed"])
    for m in ("a", "b", "c", "d"):
        try:
            res = scenario_suite(m)
            out[f"mutant_{m}"] = {"caught": not res["all_ok"], "failed": res["failed"]}
        except Exception as e:  # noqa: BLE001 - a crash is also a catch, recorded as such
            out[f"mutant_{m}"] = {"caught": True, "failed": [f"exception: {type(e).__name__}: {e}"]}
        log("mutant", m, out[f"mutant_{m}"])
    out["all_ok"] = (out["core_selftest"]["exit"] == 0 and out["ok"]["all_ok"]
                     and all(out[f"mutant_{m}"]["caught"] for m in "abcd"))
    G.write_json(RES / "a4_selfcheck.json", out)
    log("selfcheck all_ok", out["all_ok"])


# ------------------------------------------------------------------ dynamics (§3, §5)
def open_probe(d: Path, keys: list[bytes], table: str, runs: int = 21) -> dict:
    res = []
    for i in range(runs):
        r = subprocess.run([str(PY), str(ROOT / "a4_run.py"), "open-probe", str(d), table, keys[i].hex()],
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
    s = A4System("probe", table, f"{table}_seq", Path(d), lsm)
    s.conn.close()
    s.conn = conn
    row = s.lookup(bytes.fromhex(keyhex))
    t3 = time.perf_counter()
    print(json.dumps({"connect_ms": (t1 - t0) * 1e3, "open_ms": (t2 - t1) * 1e3, "first_answer_ms": (t3 - t2) * 1e3,
                      "found": row is not None, "levels": len(lsm.runs) + 2,
                      "tail_records": lsm.tail.offset // L.LOGREC.size}))


def cmd_dynamics(size: str) -> None:
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower()
    conn = G.connect()
    ref = DATA / f"ref_a4_{tag}"
    log(size, "load A4 table")
    a = setup_a4(conn, f"a4_{tag}", n, ref, DATA / f"a4_{tag}")
    log(size, "load control table")
    G.load(conn, f"c4_{tag}", 0, n, ref)
    conn.execute(f"DROP SEQUENCE IF EXISTS c4_{tag}_seq")
    conn.execute(f"CREATE SEQUENCE c4_{tag}_seq START {n + 1}")
    conn.execute(f"CREATE UNIQUE INDEX c4_{tag}_bt ON c4_{tag} USING btree (k)")
    conn.execute(f"ANALYZE c4_{tag}")
    c = Control(f"c4_{tag}", f"c4_{tag}", f"c4_{tag}_seq")
    keys = [bytes(k) for k in np.load(f"{ref}.keys")]
    digs = [bytes(x) for x in np.load(f"{ref}.dig")]
    extra = load_extra(DATA / "dyn.copy")
    ever = np.concatenate([np.load(f"{ref}.keys"), np.load(DATA / "dyn.keys")])
    ops = make_ops(keys, extra, 250_000, 150_000, 100_000, 13)
    out = {"size": size, "rows": n, "ops_sha256": ops_sha256(ops), "checkpoints": [], "batches": [], "events": [],
           "vacuum": {}, "witness_errors": 0, "s3_batch_violations": []}
    log(size, "stream sha256", out["ops_sha256"])
    model = Model(keys, digs)
    del keys, digs
    systems = (a, c)
    cp = 0
    out["checkpoints"].append(checkpoint(cp, "base", 0, systems, model, ever, 0))
    t_ops = {a.name: 0.0, c.name: 0.0}
    t_maint = 0.0
    t_vac = {a.name: 0.0, c.name: 0.0}
    wrng = random.Random(77)
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
        tm = time.perf_counter()
        before = a.lsm.nbytes()
        ev = a.maintain(len(model.state))
        tmaint = time.perf_counter() - tm
        if ev is not None:
            t_maint += tmaint
            ev = dict(ev, bytes_before=before, bytes_after=a.lsm.nbytes(), live=len(model.state), wall_s=tmaint)
            ev["witness_errors"] = witnesses(a, model, ev, wrng)
            out["witness_errors"] += ev["witness_errors"]
            out["events"].append(ev)
            if ev["kind"] == "major":
                log(size, "major compaction", round(ev["pause_s"], 2), "s at op", start + BATCH)
        op_no = start + BATCH
        nb = a.lsm.nbytes()
        per = nb / len(model.state)
        out["batches"].append([op_no, round(ta + tmaint, 4), nb, round(per, 4), len(a.lsm.runs) + 2])
        if per > 8.0:
            out["s3_batch_violations"].append([op_no, per])
        if op_no in VACUUMS:
            for s in systems:
                tv = time.perf_counter()
                conn.execute(f"VACUUM {s.table}")
                t_vac[s.name] += time.perf_counter() - tv
        if op_no % CP_EVERY == 0:
            cp += 1
            rec = checkpoint(cp, "every_50k", op_no, systems, model, ever, 0)
            rec["candidates_hist"] = dict(a.cand_hist)
            a.cand_hist = {}
            rec["peak_rss_mb"] = peak_rss_mb()
            out["checkpoints"].append(rec)
            log(size, "ops", op_no, {k: round(v, 1) for k, v in t_ops.items()}, "maint", round(t_maint, 1),
                "levels", len(a.lsm.runs) + 2)
    # restart, reopen from disk (persistence of the level set), checkpoint
    log(size, "restart")
    G.pg_ctl("restart")
    conn = G.connect()
    a.lsm.close()
    t = time.perf_counter()
    a.lsm = L.AddressLSM.open(a.dir)
    out["reopen_s"] = time.perf_counter() - t
    a.reopen()
    c.reopen()
    cp += 1
    out["checkpoints"].append(checkpoint(cp, "after_restart", OPS, systems, model, ever, 0))
    out["open_probe"] = open_probe(a.dir, model.live[:21], a.table)
    log(size, "open probe", out["open_probe"].get("open_ms"))
    final = a.major(len(model.state))
    out["final_major"] = final
    cp += 1
    out["checkpoints"].append(checkpoint(cp, "after_final_major", OPS, systems, model, ever, 0))
    out["ops_time_s"] = t_ops
    out["maintenance_s"] = t_maint
    out["vacuum_s"] = t_vac
    out["errors"] = {s.name: s.errors for s in systems}
    out["full_cost_us_per_op"] = {a.name: (t_ops[a.name] + t_maint + t_vac[a.name]) / OPS * 1e6,
                                  c.name: (t_ops[c.name] + t_vac[c.name]) / OPS * 1e6}
    pauses = [e["wall_s"] for e in out["events"]]
    out["pauses"] = {"count": len(pauses), "sum_s": sum(pauses), "max_s": max(pauses) if pauses else 0.0,
                     "share": sum(pauses) / (t_ops[a.name] + t_maint + t_vac[a.name])}
    bt = [b[1] for b in out["batches"]]
    out["batch_time_s"] = {"p50": float(np.percentile(bt, 50)), "p95": float(np.percentile(bt, 95)), "max": max(bt)}
    out["peak_rss_mb"] = peak_rss_mb()
    out["peak_coexist_bytes"] = a.lsm.peak_coexist_bytes
    a.lsm.close()
    for t in (a.table, c.table):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    G.write_json(RES / f"a4_dynamics_{tag}.json", out)
    log(size, "done errors", out["errors"], "witness", out["witness_errors"], "S3 batch violations",
        len(out["s3_batch_violations"]))


# ------------------------------------------------------------------ replay of a full cycle without SQL (§5, R6)
def cmd_replay(size: str) -> None:
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower()
    ref = DATA / f"ref_a4_{tag}"
    keys_arr = np.load(f"{ref}.keys")
    n_pages = {"N1": 76_232, "N10": 762_888}[size]
    rng = np.random.default_rng(13)
    pages = rng.integers(0, n_pages, size=n).astype(np.uint32)
    d = DATA / f"a4_replay_{tag}"
    lsm = L.AddressLSM.create(d, keys_arr, pages, n, t_run=T_RUN, f_major=1e9)
    k_cycle = int(F_MAJOR * n)
    live = [bytes(k) for k in keys_arr]
    del keys_arr
    prng = random.Random(13)
    kinds = ["I"] * (k_cycle // 2) + ["U"] * (k_cycle * 3 // 10) + ["D"] * (k_cycle - k_cycle // 2 - k_cycle * 3 // 10)
    prng.shuffle(kinds)
    t_log = t_maint = 0.0
    seals = merges = 0
    read_b = written_b = 0
    first_run_entries = None
    for start in range(0, k_cycle, BATCH):
        t = time.perf_counter()
        lsm.batch_no += 1
        for kind in kinds[start: start + BATCH]:
            lsm.opseq += 1
            if kind == "I":
                kb = prng.randbytes(16)
                live.append(kb)
                lsm.tail.append(L.I_, kb, prng.randrange(n_pages + 100_000), lsm.opseq)
            elif kind == "U":
                lsm.tail.append(L.U_, live[prng.randrange(len(live))], prng.randrange(n_pages + 100_000), lsm.opseq)
            else:
                i = prng.randrange(len(live))
                kb = live[i]
                live[i] = live[-1]
                live.pop()
                lsm.tail.append(L.D_, kb, 0, lsm.opseq)
        lsm.ops_since_major += lsm.tail.commit(lsm.batch_no)
        t_log += time.perf_counter() - t
        if lsm.due() == "seal":
            t = time.perf_counter()
            if first_run_entries is None:
                first_run_entries = {k: p for k, p in lsm.tail.committed.items() if p is not None}
            ev = lsm.seal()
            t_maint += time.perf_counter() - t
            seals += 1
            written_b += ev["run"]["adr_bytes"] + ev["run"]["log_bytes"]
            read_b += 0
            for m in ev["merges"]:
                merges += 1
                read_b += m["read_bytes"]
                written_b += m["written_bytes"]
    out = {"size": size, "cycle_ops": k_cycle, "seals": seals, "merges": merges, "maint_s": t_maint, "log_s": t_log,
           "maint_us_per_op": t_maint / k_cycle * 1e6, "log_us_per_op": t_log / k_cycle * 1e6,
           "maint_bytes_per_op": (read_b + written_b) / k_cycle, "read_bytes": read_b, "written_bytes": written_b,
           "levels_end": len(lsm.runs) + 2, "tiers_end": [r.tier for r in lsm.runs], "address_bytes_end": lsm.nbytes(),
           "live_end": len(live)}
    # R9 control: one real run of 30 000 changes, JLMADR01 vs a sorted (key, page) segment
    ks = sorted(first_run_entries)
    sk = np.frombuffer(b"".join(ks), np.uint8).reshape(len(ks), 16)
    sp = np.fromiter((first_run_entries[k] for k in ks), np.uint32, len(ks))
    t = time.perf_counter()
    ca = C.build_a(sk, sp, 1, L.r_bits_needed(int(sp.max()) + 1))
    t_build = time.perf_counter() - t
    pth = d / "r9.adr"
    info = C.write_a(pth, ca, table_oid=0, relfilenode=0, generation=0, n_rows=len(ks), n_pages=int(sp.max()) + 1, watermark=0)
    sa = C.SidecarA(pth)
    probe = ks[::30]
    t = time.perf_counter()
    for k in probe:
        sa.page_of(k)
    t_mphf = (time.perf_counter() - t) / len(probe)
    import bisect
    t = time.perf_counter()
    for k in probe:
        i = bisect.bisect_left(ks, k)
        _ = sp[i]
    t_sorted = (time.perf_counter() - t) / len(probe)
    sa.close()
    out["r9"] = {"keys": len(ks), "mphf_bytes_per_key": info["bytes"] / len(ks), "sorted_bytes_per_key": 20.0,
                 "mphf_build_s": t_build, "sorted_build_s": 0.0, "mphf_lookup_us": t_mphf * 1e6,
                 "sorted_lookup_us": t_sorted * 1e6,
                 "note": "the sorted segment is the run log itself; A4 keeps both (log for merges, address for lookups)"}
    lsm.close()
    G.write_json(RES / f"a4_replay_{tag}.json", out)
    log("replay", size, {k: out[k] for k in ("seals", "merges", "maint_us_per_op", "maint_bytes_per_op", "levels_end")})


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
