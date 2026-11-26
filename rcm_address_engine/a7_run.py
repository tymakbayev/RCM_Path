"""A7 runs (protocol A7_protocol.md edition 2): the cost of hash partitioning in one run.
Run with .venv-pg:
  a7_run.py selfcheck | a7_run.py dynamics N1 r1 | a7_run.py dynamics N1 r2 | a7_run.py dynamics N10 r1
A2-A6 code is used unchanged (A6System, a6_lsm, a6_pghash, create_parted, read_part, function_check, witnesses)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import statistics as st
import subprocess
import sys
import time
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg

import a2_pg as G
import a6_lsm as L
import a6_pghash as PH
import a6_run as A6
from a2_dyn import BATCH, OPS, SAMPLE, Model, load_extra, make_ops, ops_sha256
from a2_runs import DATA, RES, absent_keys, log, sha16
from a5_run import clean, rss_mb

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "A7_protocol.md"
COLS = A6.COLS
VACUUMS = (240_000, 480_000)
CP_AT = (0, 100_000, 200_000, 300_000, 400_000, 500_000)
SCHED_SEED = 707
ORDERS = (("s", "p", "a7"), ("p", "a7", "s"), ("a7", "s", "p"))
COMBOS = tuple((o, r) for o in range(3) for r in ("par", "dir"))
BLOCK_BATCHES = 50
DIAG_N, DIAG_WARM = 2000, 200
N_EXPLAIN_EXEC, N_EXEC_HISTORY, N_EXPLAIN_LIT = 50, 10, 500
SETTINGS = ("server_version", "shared_buffers", "effective_cache_size", "work_mem", "maintenance_work_mem",
            "autovacuum", "checkpoint_timeout", "max_wal_size", "synchronous_commit", "plan_cache_mode", "jit")
MUTANTS = ("M1", "M2", "M3", "M4", "M5")
MIN_FREE_N10 = 24e9
SIZES = {"N1": 1_000_000, "N10": 10_000_000}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_protocol() -> None:
    want = (RES / "a7_protocol.sha256").read_text(encoding="utf-8").split()[0]
    if sha_file(PROTOCOL) != want:
        raise SystemExit("A7_protocol.md changed after it was frozen")


def impl_hashes() -> dict:
    out = A6.impl_hashes()
    out["a6_run.py"] = sha_file(ROOT / "a6_run.py")
    out["a7_run.py"] = sha_file(ROOT / "a7_run.py")
    return out


# ------------------------------------------------------------------ schedule (§2.3)
def schedule(nb: int, mutant: str | None = None) -> list[tuple[int, str]]:
    rng = random.Random(SCHED_SEED)
    out: list[tuple[int, str]] = []
    while len(out) < nb:
        perm = list(COMBOS)
        rng.shuffle(perm)
        out += perm
    out = out[:nb]
    if mutant == "M3":
        out = [(o, "par") for o, _ in out]
    if mutant == "M4":
        out = [(0, r) for _, r in out]
    return out


def schedule_balance(used: list[tuple[int, str]]) -> dict:
    """(п1): every combination n/6 (±1), routes n/2 (±1)."""
    n = len(used)
    c = Counter(used)
    routes = Counter(r for _, r in used)
    combos_ok = len(c) == len(COMBOS) and all(abs(c[x] - n / len(COMBOS)) <= 1 for x in COMBOS)
    routes_ok = all(abs(routes[r] - n / 2) <= 1 for r in ("par", "dir"))
    return {"n": n, "combos": {f"{o}:{r}": c[(o, r)] for o, r in COMBOS}, "routes": dict(routes),
            "ok": combos_ok and routes_ok}


# ------------------------------------------------------------------ one B-tree client for three routes (§2.2)
class B7:
    """Control.apply of A4 (lookup by key, then INSERT / UPDATE / DELETE by key, prepare=True, one transaction per
    batch); only the table name differs by route. dir: partition g(k) with the A6 LRU (2^18)."""

    def __init__(self, route: str, table: str, seq: str, nparts: int, mutant: str | None = None, conn=None) -> None:
        self.route, self.table, self.seq, self.nparts, self.mutant = route, table, seq, nparts, mutant
        self.conn = conn if conn is not None else G.connect()
        self.conn.prepared_max = 1_000_000
        self._g = lru_cache(maxsize=1 << 18)(self._g_uncached)
        self.g_ns = 0
        self.g_calls = 0
        self.kind_ns = {k: [0, 0, 0, 0, 0] for k in "IUD"}  # lookup, write, operations, whole op, routing+uuid
        self.batch_timer_ns = 0
        self.errors = 0

    def _g_uncached(self, kb: bytes) -> int:
        j = PH.part_of(kb, self.nparts)
        return (j + 1) % self.nparts if self.mutant == "M2" else j

    def rel(self, kb: bytes) -> str:
        if self.route != "dir" or self.mutant == "M1":
            return self.table
        t = time.perf_counter_ns()
        j = self._g(kb)
        self.g_ns += time.perf_counter_ns() - t
        self.g_calls += 1
        return f"{self.table}_p{j}"

    def get(self, rel: str, k: uuid.UUID):
        rows = self.conn.execute(f"SELECT {COLS} FROM {rel} WHERE k = %s", (k,), prepare=True).fetchall()
        if len(rows) > 1:
            raise RuntimeError(f"{len(rows)} visible rows for one key")
        return rows[0] if rows else None

    def lookup(self, kb: bytes):
        return self.get(self.rel(kb), uuid.UUID(bytes=kb))

    def apply(self, ops) -> list[str]:
        out = []
        tsum = 0
        skip_insert = self.mutant == "M5" and self.route == "dir"
        with self.conn.transaction():
            for kind, kb, payload in ops:
                t0 = time.perf_counter_ns()
                rel = self.rel(kb)
                k = uuid.UUID(bytes=kb)
                t1 = time.perf_counter_ns()
                row = self.get(rel, k)
                t2 = time.perf_counter_ns()
                if kind == "I":
                    if row is not None:
                        res = "dup"
                    else:
                        if not skip_insert:
                            cur = self.conn.execute(f"INSERT INTO {rel} (seq, k, v, payload) VALUES "
                                                    f"(nextval('{self.seq}'), %s, 1, %s)", (k, payload), prepare=True)
                            if cur.rowcount != 1:
                                self.errors += 1
                        res = "ins"
                elif row is None:
                    res = "none"
                elif kind == "U":
                    cur = self.conn.execute(f"UPDATE {rel} SET payload = %s, v = v + 1, seq = nextval('{self.seq}') "
                                            "WHERE k = %s", (payload, k), prepare=True)
                    if cur.rowcount != 1:
                        self.errors += 1
                    res = "upd"
                else:
                    cur = self.conn.execute(f"DELETE FROM {rel} WHERE k = %s", (k,), prepare=True)
                    if cur.rowcount != 1:
                        self.errors += 1
                    res = "del"
                t3 = time.perf_counter_ns()
                kn = self.kind_ns[kind]
                kn[0] += t2 - t1
                kn[1] += t3 - t2
                kn[2] += 1
                kn[3] += t3 - t0
                kn[4] += t1 - t0
                tsum += t3 - t0
                out.append(res)
        self.batch_timer_ns = tsum
        return out

    def kind_summary(self) -> dict:
        out = {k: {"ops": v[2], "lookup_us": v[0] / v[2] / 1000 if v[2] else None,
                   "write_us": v[1] / v[2] / 1000 if v[2] else None, "op_us": v[3] / v[2] / 1000 if v[2] else None,
                   "route_us": v[4] / v[2] / 1000 if v[2] else None} for k, v in self.kind_ns.items()}
        ci = self._g.cache_info()
        out["g"] = {"calls": self.g_calls, "us_per_call": self.g_ns / self.g_calls / 1000 if self.g_calls else None,
                    "cache_hits": ci.hits, "cache_misses": ci.misses}
        return out


# ------------------------------------------------------------------ loading: one generator, three COPY (§2.1, R6)
def load_three(tag: str, n: int, ref: Path, bits: int, d: Path, t_run: int) -> tuple[dict, A6.A6System]:
    nparts = 1 << bits
    ts, tp, ta = f"s7_{tag}", f"p7_{tag}", f"a7_{tag}"
    cs, cpp, ca = G.connect(), G.connect(), G.connect()
    G.create_table(cs, ts)
    A6.create_parted(cpp, tp, nparts)
    A6.create_parted(ca, ta, nparts)
    t0 = time.perf_counter()
    proc = subprocess.Popen([str(G.GEN_PY), str(G.GEN), "0", str(n), "--ref", str(ref), "--workers", "6"],
                            stdout=subprocess.PIPE, cwd=str(ROOT))
    sql = "COPY {} (seq, k, v, payload) FROM STDIN"
    with cs.cursor() as c1, c1.copy(sql.format(ts)) as w1, cpp.cursor() as c2, c2.copy(sql.format(tp)) as w2, \
            ca.cursor() as c3, c3.copy(sql.format(ta)) as w3:
        while chunk := proc.stdout.read(1 << 20):
            w1.write(chunk)
            w2.write(chunk)
            w3.write(chunk)
    if proc.wait() != 0:
        raise RuntimeError("a2_gen failed")
    info = {"copy_s": time.perf_counter() - t0}
    for t in (ts, tp):  # as the A5 control after a2_pg.load: VACUUM (ANALYZE), sequence, unique B-tree, ANALYZE
        t1 = time.perf_counter()
        cs.execute(f"VACUUM (ANALYZE) {t}")
        cs.execute(f"DROP SEQUENCE IF EXISTS {t}_seq")
        cs.execute(f"CREATE SEQUENCE {t}_seq START {n + 1}")
        cs.execute(f"CREATE UNIQUE INDEX {t}_bt ON {t} USING btree (k)")
        cs.execute(f"ANALYZE {t}")
        info[f"post_{t[:2]}_s"] = time.perf_counter() - t1
    t1 = time.perf_counter()  # as setup_a6 after load_parted
    cs.execute(f"VACUUM (ANALYZE) {ta}")
    cs.execute(f"DROP SEQUENCE IF EXISTS {ta}_seq")
    cs.execute(f"CREATE SEQUENCE {ta}_seq START {n + 1}")
    ks, ps = [], []
    for j in range(nparts):
        k, p = A6.read_part(cs, f"{ta}_p{j}")
        ks.append(k)
        ps.append(p)
    lsm = L.AddressLSM6.create(d, np.concatenate(ks), np.concatenate(ps), n, s_base_bits=bits, t_run=t_run,
                               f_major=A6.F_MAJOR)
    lsm.n_live = int(sum(len(k) for k in ks))
    a = A6.A6System(ta, ta, f"{ta}_seq", d, lsm)
    info["post_a7_s"] = time.perf_counter() - t1
    # checks after loading (§2.1)
    same = total = 0
    for j in range(nparts):
        kp, pp = A6.read_part(cs, f"{tp}_p{j}")
        da = dict(zip(map(bytes, ks[j]), ps[j].tolist()))
        same += sum(1 for k, p in zip(map(bytes, kp), pp.tolist()) if da.get(k) == p)
        total += len(kp)
    info["layout_p7_a7"] = {"same": same, "total": total, "share": same / max(1, total)}
    info["rows"] = {t: int(cs.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in (ts, tp, ta)}
    info["pages"] = {"s7": G.relinfo(cs, ts)["pages"], "p7_sum": sum(x[1] for x in A6.part_sizes(cs, tp)),
                     "a7_sum": sum(x[1] for x in A6.part_sizes(cs, ta))}
    for c in (cs, cpp, ca):
        c.close()
    return info, a


def sizes_now(conn, tag: str, a: A6.A6System) -> dict:
    ts, tp, ta = f"s7_{tag}", f"p7_{tag}", f"a7_{tag}"
    ix = G.index_size(conn, f"{ts}_bt")
    pm, pt = conn.execute("SELECT sum(pg_relation_size(relid)), sum(pg_total_relation_size(relid)) FROM "
                          "pg_partition_tree(%s) WHERE isleaf", (f"{tp}_bt",)).fetchone()
    return {"s7": {"index_main_bytes": ix["main_bytes"], "table_bytes": G.relinfo(conn, ts)["table_bytes"]},
            "p7": {"index_main_bytes": int(pm), "index_total_bytes": int(pt),
                   "table_bytes": A6.parted_bytes(conn, tp)["table_bytes"]},
            "a7": {"address_bytes": a.lsm.nbytes(), "table_bytes": A6.parted_bytes(conn, ta)["table_bytes"]}}


def heights(conn, tag: str) -> dict:
    s = int(conn.execute("SELECT level FROM bt_metap(%s)", (f"s7_{tag}_bt",)).fetchone()[0]) + 1
    return {"s7": s, "p7": sorted(set(A6.btree_heights(conn, f"p7_{tag}")))}


# ------------------------------------------------------------------ prepared statements of the working connections
def prepared_stats(conn) -> list[list]:
    rows = conn.execute("SELECT statement, generic_plans, custom_plans FROM pg_prepared_statements",
                        prepare=False).fetchall()
    return [[r[0], int(r[1]), int(r[2])] for r in rows]


def prepared_rule(tp: str, par_rows: list, dir_rows: list) -> list[str]:
    """§3: no statement of the dir connection names the parent; none of the par connection names a partition."""
    parent = re.compile(rf"\b{re.escape(tp)}\b")
    leaf = re.compile(rf"\b{re.escape(tp)}_p\d+\b")
    bad = [f"dir: {r[0][:80]}" for r in dir_rows if parent.search(r[0])]
    bad += [f"par: {r[0][:80]}" for r in par_rows if leaf.search(r[0])]
    return bad


def plan_counts(rows: list) -> dict:
    out: dict[str, list[int]] = {}
    for stmt, gen, cus in rows:
        kind = stmt.lstrip("( ").split(None, 1)[0].upper()
        v = out.setdefault(kind, [0, 0, 0])
        v[0] += gen
        v[1] += cus
        v[2] += 1
    return {k: {"generic": v[0], "custom": v[1], "statements": v[2]} for k, v in out.items()}


# ------------------------------------------------------------------ diagnostics at C7 (§4, R5)
def has_key(node, key: str) -> bool:
    if key in node:
        return True
    return any(has_key(ch, key) for ch in node.get("Plans", []))


def index_conds(node) -> list[str]:
    out = [node["Index Cond"]] if "Index Cond" in node else []
    for ch in node.get("Plans", []):
        out += index_conds(ch)
    return out


class Ctx:
    def __init__(self, tag: str, nparts: int, mutant: str | None) -> None:
        self.tag, self.nparts, self.mutant = tag, nparts, mutant
        self.ts, self.tp, self.ta = f"s7_{tag}", f"p7_{tag}", f"a7_{tag}"

    def table(self, route: str) -> str:
        return self.ts if route == "s" else self.tp

    def reader(self, route: str, conn=None) -> B7:
        t = self.table(route)
        return B7(route, t, f"{t}_seq", self.nparts, self.mutant, conn)


def diag_conn(mode: str):
    """A new connection with the plan cache mode set explicitly, auto included (R6 Codex 249)."""
    conn = G.connect()
    conn.execute(f"SET plan_cache_mode = {mode}")
    return conn


def diag_latency(ctx: Ctx, route: str, mode: str, keys: list[bytes]) -> dict:
    conn = diag_conn(mode)
    r = ctx.reader(route, conn)
    for kb in keys[:DIAG_WARM]:
        r.lookup(kb)
    times = []
    for kb in keys[DIAG_WARM:]:
        t = time.perf_counter_ns()
        r.lookup(kb)
        times.append((time.perf_counter_ns() - t) / 1000)
    conn.close()
    return {"n": len(times), "client_us": G.pct(times)}


def diag_pairs(ctx: Ctx, keys: list[bytes]) -> dict:
    a, b = ctx.reader("par", diag_conn("auto")), ctx.reader("dir", diag_conn("auto"))
    for kb in keys[:DIAG_WARM]:
        a.lookup(kb)
        b.lookup(kb)
    tp, td, diff = [], [], []
    for i, kb in enumerate(keys[DIAG_WARM:]):
        pair = (a, b) if i % 2 == 0 else (b, a)
        got = {}
        for c in pair:
            t = time.perf_counter_ns()
            c.lookup(kb)
            got[c.route] = (time.perf_counter_ns() - t) / 1000
        tp.append(got["par"])
        td.append(got["dir"])
        diff.append(got["par"] - got["dir"])
    a.conn.close()
    b.conn.close()
    return {"n": len(diff), "par_us": G.pct(tp), "dir_us": G.pct(td), "diff_median_us": float(np.median(diff)),
            "diff_mean_us": float(np.mean(diff)), "par_slower_share": float(np.mean([x > 0 for x in diff]))}


def stmt_counts(conn, name: str) -> tuple[int, int]:
    g, c = conn.execute("SELECT generic_plans, custom_plans FROM pg_prepared_statements WHERE name = %s", (name,),
                        prepare=False).fetchone()
    return int(g), int(c)


def diag_explain_execute(ctx: Ctx, route: str, mode: str, keys: list[bytes], live: list[bytes] | None = None) -> dict:
    """(д4): EXPLAIN EXECUTE of a named prepared lookup after 10 executions with live keys of the same table (for
    dir: of the same partition). The plan kind of the explained execution is the increment of generic_plans /
    custom_plans of that statement (R4 Codex 249); '$1' in an index condition is kept as a second sign."""
    conn = diag_conn(mode)
    r = ctx.reader(route, conn)
    names: dict[str, str] = {}
    pool = keys[N_EXPLAIN_EXEC:]
    hist_n, plan_ms, exec_ms, bufs, removed, by_count, by_dollar = [], [], [], [], [], [], []
    for kb in keys[:N_EXPLAIN_EXEC]:
        rel = r.rel(kb)
        if rel not in names:
            names[rel] = f"q7_{len(names)}"
            conn.execute(f"PREPARE {names[rel]} (uuid) AS SELECT {COLS} FROM {rel} WHERE k = $1")
            hist = [hk for hk in pool if r.rel(hk) == rel][:N_EXEC_HISTORY]
            if live and len(hist) < N_EXEC_HISTORY:  # R3 Codex 251: top up with live keys of the same partition
                trng = random.Random(4000 + len(names))
                tries = 0
                while len(hist) < N_EXEC_HISTORY and tries < 200_000:
                    hk = live[trng.randrange(len(live))]
                    tries += 1
                    if r.rel(hk) == rel and hk not in hist and hk != kb:
                        hist.append(hk)
            hist_n.append(len(hist))
            for hk in hist:
                conn.execute(f"EXECUTE {names[rel]}('{uuid.UUID(bytes=hk)}')").fetchall()
        g0, c0 = stmt_counts(conn, names[rel])
        res = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) EXECUTE {names[rel]}('{uuid.UUID(bytes=kb)}')"
                           ).fetchone()[0][0]
        g1, c1 = stmt_counts(conn, names[rel])
        top = res["Plan"]
        plan_ms.append(res.get("Planning Time", 0.0))
        exec_ms.append(res["Execution Time"])
        bufs.append(top.get("Shared Hit Blocks", 0) + top.get("Shared Read Blocks", 0))
        removed.append(has_key(top, "Subplans Removed"))
        by_count.append("generic" if g1 - g0 == 1 and c1 == c0 else "custom" if c1 - c0 == 1 and g1 == g0 else "?")
        by_dollar.append(any("$1" in c for c in index_conds(top)))
    conn.close()
    return {"n": len(exec_ms), "statements": len(names), "history_min": min(hist_n), "history_mean": float(np.mean(hist_n)),
            "history_ok": min(hist_n) >= N_EXEC_HISTORY,
            "planning_ms_mean": float(np.mean(plan_ms)), "execution_ms_mean": float(np.mean(exec_ms)),
            "buffers_mean": float(np.mean(bufs)), "subplans_removed_field_share": float(np.mean(removed)),
            "plan_kind_by_counters": dict(Counter(by_count)), "dollar1_in_index_cond_share": float(np.mean(by_dollar))}


def diag_explain_literal(ctx: Ctx, route: str, keys: list[bytes]) -> dict:
    """(д5): literal EXPLAIN (ANALYZE, BUFFERS); the first 200 are warm-up (R6 Codex 249)."""
    conn = diag_conn("auto")
    r = ctx.reader(route, conn)
    plan_ms, exec_ms, bufs, pbufs = [], [], [], []
    for i, kb in enumerate(keys[:DIAG_WARM + N_EXPLAIN_LIT]):
        sql = G.literal(f"SELECT {COLS} FROM {r.rel(kb)} WHERE k = %s", (uuid.UUID(bytes=kb),))
        res = conn.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql).fetchone()[0][0]
        if i < DIAG_WARM:
            continue
        top = res["Plan"]
        pl = res.get("Planning", {})
        plan_ms.append(res.get("Planning Time", 0.0))
        exec_ms.append(res["Execution Time"])
        bufs.append(top.get("Shared Hit Blocks", 0) + top.get("Shared Read Blocks", 0))
        pbufs.append(pl.get("Shared Hit Blocks", 0) + pl.get("Shared Read Blocks", 0))
    conn.close()
    return {"n": len(exec_ms), "planning_ms_mean": float(np.mean(plan_ms)), "execution_ms_mean": float(np.mean(exec_ms)),
            "buffers_mean": float(np.mean(bufs)), "planning_buffers_mean": float(np.mean(pbufs))}


LOCKS_Q = ("SELECT l.relation::bigint, l.mode, l.granted, c.relkind::text, coalesce(t.relname, c.relname)::text "
           "FROM pg_locks l LEFT JOIN pg_class c ON c.oid = l.relation LEFT JOIN pg_index i ON i.indexrelid = l.relation "
           "LEFT JOIN pg_class t ON t.oid = i.indrelid WHERE l.pid = pg_backend_pid() AND l.locktype = 'relation'")


def lock_category(ctx: Ctx, relkind: str | None, table_name: str | None) -> str:
    kind = "index" if relkind in ("i", "I") else "table"
    if table_name == ctx.tp:
        return f"parent_{kind}"
    if table_name and re.fullmatch(rf"{re.escape(ctx.tp)}_p\d+", table_name):
        return f"leaf_{kind}"
    if table_name == ctx.ts:
        return f"single_{kind}"
    return "other"


def diag_locks(ctx: Ctx, route: str, mode: str, keys: list[bytes]) -> dict:
    """(д6): relation locks the lookup adds in its own transaction, first and second execution of the same prepared
    statement: (relation, mode, granted) snapshots before and after; distinct relations by category (parent, leaves,
    their indexes, other). Composition only, not the time of taking them (R5 Codex 249)."""
    conn = diag_conn(mode)
    r = ctx.reader(route, conn)
    out = []
    for kb in (keys[0], keys[0]):  # the same statement and the same key both times
        with conn.transaction():
            before = {tuple(x) for x in conn.execute(LOCKS_Q, prepare=False).fetchall()}
            r.lookup(kb)
            after = {tuple(x) for x in conn.execute(LOCKS_Q, prepare=False).fetchall()}
        added = after - before
        rels = {}
        for rel, _mode, _g, relkind, tname in added:
            rels[rel] = lock_category(ctx, relkind, tname)
        out.append({"entries_added": len(added), "relations_added": len(rels), "by_category": dict(Counter(rels.values())),
                    "not_granted": sum(1 for x in added if not x[2])})
    conn.close()
    return {"first": out[0], "second": out[1]}


def diagnostics(ctx: Ctx, cp_no: int, model: Model) -> dict:
    keys = random.Random(31 + cp_no).sample(model.live, min(DIAG_N + DIAG_WARM, len(model.live)))
    items = [("d1", "s", "auto"), ("d1", "par", "auto"), ("d1", "dir", "auto"),
             ("d2", "par", "force_generic_plan"), ("d2", "par", "force_custom_plan"), ("d3", "pairs", "auto"),
             ("d4", "par", "auto"), ("d4", "par", "force_generic_plan"), ("d4", "par", "force_custom_plan"),
             ("d4", "dir", "auto"), ("d4", "s", "auto"), ("d5", "s", "auto"), ("d5", "par", "auto"),
             ("d5", "dir", "auto"), ("d6", "par", "force_generic_plan"), ("d6", "par", "force_custom_plan"),
             ("d6", "dir", "auto"), ("d6", "s", "auto")]
    random.Random(900 + cp_no).shuffle(items)
    out = {"order": [":".join(x) for x in items]}
    for kind, route, mode in items:
        if kind in ("d1", "d2"):
            res = diag_latency(ctx, route, mode, keys)
        elif kind == "d3":
            res = diag_pairs(ctx, keys)
        elif kind == "d4":
            res = diag_explain_execute(ctx, route, mode, keys, model.live)
        elif kind == "d5":
            res = diag_explain_literal(ctx, route, keys)
        else:
            res = diag_locks(ctx, route, mode, keys)
        out[f"{kind}:{route}:{mode}"] = res
    return out


# ------------------------------------------------------------------ checkpoint C7 (§3)
def checkpoint7(ctx: Ctx, cp_no: int, op_no: int, model: Model, ever: np.ndarray, a: A6.A6System, work: dict,
                diag: bool = True) -> dict:
    rng = random.Random(21 + cp_no)
    live = rng.sample(model.live, min(SAMPLE, len(model.live)))
    dele = sorted(model.deleted)
    random.Random(41 + cp_no).shuffle(dele)
    dele = dele[:SAMPLE]
    absent = absent_keys(ever, SAMPLE, 61 + cp_no)
    rec = {"cp": cp_no, "after_op": op_no, "samples": {"live": len(live), "deleted": len(dele), "absent": len(absent)},
           "routes": {}}

    def check(lookup) -> tuple[dict, list[float]]:
        err = {"live": 0, "deleted": 0, "absent": 0}
        times = []
        for kb in live:
            t = time.perf_counter_ns()
            row = lookup(kb)
            times.append((time.perf_counter_ns() - t) / 1000)
            s_ = model.state[kb]
            if row is None or row[1] != s_[0] or sha16(row[3]) != s_[1]:
                err["live"] += 1
        err["deleted"] = sum(1 for kb in dele if lookup(kb) is not None)
        err["absent"] = sum(1 for kb in absent if lookup(kb) is not None)
        return err, times

    for route in ("s", "par", "dir"):  # separate new connections (R5)
        r = ctx.reader(route)
        err, times = check(r.lookup)
        r.conn.close()
        rec["routes"][route] = {"errors": err, "client_us": G.pct(times)}
    err, times = check(a.lookup)
    rec["routes"]["a7"] = {"errors": err, "client_us": G.pct(times)}
    rows = {r: prepared_stats(c.conn) for r, c in work.items()}
    rec["prepared"] = {r: plan_counts(v) for r, v in rows.items()}
    rec["prepared_rule_violations"] = prepared_rule(ctx.tp, rows["par"], rows["dir"])
    rec["errors_total"] = sum(sum(v["errors"].values()) for v in rec["routes"].values())
    if diag:
        rec["diag"] = diagnostics(ctx, cp_no, model)
    log("C7", cp_no, op_no, {k: sum(v["errors"].values()) for k, v in rec["routes"].items()},
        {k: round(v["client_us"]["p50"], 1) for k, v in rec["routes"].items()},
        "prep violations", len(rec["prepared_rule_violations"]))
    return rec


# ------------------------------------------------------------------ summary (§4); pure, checked on fixed numbers
def div(a, b):
    """a / b, or None when either is undefined or b is zero (a route absent from the selection, R1 Codex 249)."""
    return None if a is None or b is None or b == 0 else a / b


def summarize(batches: list[dict], vac: dict, maint_s: float, block_batches: int = BLOCK_BATCHES) -> dict:
    def tsum(sel, key):
        return sum(x[key] for x in sel)

    def ops(sel):
        return sum(x["ops"] for x in sel)

    def per_op(sel, key):
        return div(tsum(sel, key) * 1e6, ops(sel))

    def ratios(sel_all):
        P = [x for x in sel_all if x["route"] == "par"]
        D = [x for x in sel_all if x["route"] == "dir"]
        Ts, Tpar, Tdir, Ta = per_op(sel_all, "t_s"), per_op(P, "t_p"), per_op(D, "t_p"), per_op(sel_all, "t_a7")
        return {"T_s": Ts, "T_par": Tpar, "T_dir": Tdir, "T_a7": Ta, "K_par": div(Tpar, Ts), "K_dir": div(Tdir, Ts),
                "K_route": div(Tpar, Tdir), "K_par_m": div(tsum(P, "t_p"), tsum(P, "t_s")),
                "K_dir_m": div(tsum(D, "t_p"), tsum(D, "t_s")), "a7_s": div(Ta, Ts),
                "a7_s_on_P": div(tsum(P, "t_a7"), tsum(P, "t_s")), "a7_s_on_D": div(tsum(D, "t_a7"), tsum(D, "t_s"))}

    n = ops(batches)
    whole = ratios(batches)
    F_s = div((tsum(batches, "t_s") + vac["s"]) * 1e6, n)
    F_pmix = div((tsum(batches, "t_p") + vac["p"]) * 1e6, n)
    vp = div(vac["p"] * 1e6, n)
    F_par = None if whole["T_par"] is None else whole["T_par"] + vp
    F_dir = None if whole["T_dir"] is None else whole["T_dir"] + vp
    F_a7 = div((tsum(batches, "t_a7") + maint_s + vac["a7"]) * 1e6, n)
    full = {"F_s": F_s, "F_p_mix": F_pmix, "F_par_hat": F_par, "F_dir_hat": F_dir, "F_a7": F_a7,
            "F_p_mix/F_s": div(F_pmix, F_s), "F_par_hat/F_s": div(F_par, F_s), "F_dir_hat/F_s": div(F_dir, F_s),
            "F_a7/F_s": div(F_a7, F_s), "F_a7/F_par_hat": div(F_a7, F_par), "F_a7/F_dir_hat": div(F_a7, F_dir)}
    blocks = [ratios(batches[i:i + block_batches]) for i in range(0, len(batches), block_batches)]
    bstats = {}
    for k in ("K_par", "K_dir", "K_route", "K_par_m", "K_dir_m", "a7_s", "a7_s_on_P", "a7_s_on_D"):
        v = [b[k] for b in blocks]
        d = [x for x in v if x is not None]
        bstats[k] = {"defined": len(d), "values": v}
        if d:
            bstats[k].update({"min": min(d), "max": max(d), "cv": float(np.std(d) / np.mean(d)),
                              "sign": ("all>1" if all(x > 1 for x in d) else "all<1" if all(x < 1 for x in d)
                                       else "mixed") if len(d) == len(v) else "undefined blocks"})
    trans = {}
    for cur in ("par", "dir"):
        for prev in ("par", "dir"):
            sel = [x for x in batches if x["route"] == cur and x["prev"] == prev]
            trans[f"{prev}->{cur}"] = {"batches": len(sel), "T_us": per_op(sel, "t_p")}
    tblocks = []
    for i in range(0, len(batches), block_batches):
        blk = batches[i:i + block_batches]
        tblocks.append({f"{prev}->{cur}": per_op([x for x in blk if x["route"] == cur and x["prev"] == prev], "t_p")
                        for cur in ("par", "dir") for prev in ("par", "dir")})
    mix = {r: {k: sum(x[f"n{k}"] for x in batches if x["route"] == r) for k in "IUD"} for r in ("par", "dir")}
    return {"ops": n, "whole": whole, "full": full, "blocks": bstats, "transitions": trans,
            "transitions_by_block": tblocks, "mix": mix}


def summarize_selftest() -> dict:
    """Fixed numbers: 12 batches of 1000 ops alternating par/dir; t_s 0.1 on par batches and 0.2 on dir batches;
    t_p 0.2 (par) / 0.15 (dir); t_a7 0.3; maintenance 1.2 s; VACUUM s 1, p 2, a7 3; blocks of 6."""
    batches = []
    for b in range(12):
        r = "par" if b % 2 == 0 else "dir"
        batches.append({"ops": 1000, "route": r, "prev": None if b == 0 else ("dir" if r == "par" else "par"),
                        "t_s": 0.1 if r == "par" else 0.2, "t_p": 0.2 if r == "par" else 0.15, "t_a7": 0.3,
                        "nI": 500, "nU": 300, "nD": 200})
    s = summarize(batches, {"s": 1.0, "p": 2.0, "a7": 3.0}, 1.2, 6)
    want = {("whole", "T_s"): 150.0, ("whole", "T_par"): 200.0, ("whole", "T_dir"): 150.0, ("whole", "T_a7"): 300.0,
            ("whole", "K_par"): 200 / 150, ("whole", "K_dir"): 1.0, ("whole", "K_route"): 200 / 150,
            ("whole", "K_par_m"): 2.0, ("whole", "K_dir_m"): 0.75, ("whole", "a7_s_on_P"): 3.0,
            ("whole", "a7_s_on_D"): 1.5, ("full", "F_s"): 2.8 / 12000 * 1e6, ("full", "F_p_mix"): 4.1 / 12000 * 1e6,
            ("full", "F_par_hat"): 200 + 2 / 12000 * 1e6, ("full", "F_dir_hat"): 150 + 2 / 12000 * 1e6,
            ("full", "F_a7"): 7.8 / 12000 * 1e6}
    bad = {f"{a}.{b}": (s[a][b], v) for (a, b), v in want.items() if abs(s[a][b] - v) > 1e-9 * max(1, abs(v))}
    if len(s["blocks"]["K_par"]["values"]) != 2 or s["blocks"]["K_par_m"]["sign"] != "all>1" \
            or s["blocks"]["K_dir_m"]["sign"] != "all<1":
        bad["blocks"] = s["blocks"]["K_par"]
    if s["transitions"]["dir->par"]["batches"] != 5 or abs(s["transitions"]["dir->par"]["T_us"] - 200) > 1e-9 \
            or s["transitions"]["par->par"]["batches"] != 0 or s["transitions"]["par->dir"]["batches"] != 6:
        bad["transitions"] = s["transitions"]
    if s["mix"]["par"] != {"I": 3000, "U": 1800, "D": 1200} or s["ops"] != 12000:
        bad["mix"] = s["mix"]
    # R1 Codex 249: a set with par batches only (as under M3) gives undefined dir ratios, not an exception
    only_par = [dict(x, route="par", prev="par") for x in batches]
    try:
        s2 = summarize(only_par, {"s": 1.0, "p": 2.0, "a7": 3.0}, 1.2, 6)
        if s2["whole"]["T_dir"] is not None or s2["whole"]["K_route"] is not None \
                or s2["full"]["F_dir_hat"] is not None or s2["blocks"]["K_dir"]["defined"] != 0 \
                or abs(s2["whole"]["T_par"] - 175.0) > 1e-9:
            bad["only_par"] = s2["whole"]
    except Exception as e:  # noqa: BLE001
        bad["only_par"] = f"{type(e).__name__}: {e}"
    return {"ok": not bad, "bad": bad}


# ------------------------------------------------------------------ the run loop (§2.3, §3)
class Stop(Exception):
    pass


def plan_increments(cps: list[dict]) -> list[dict]:
    """R7: generic / custom plan executions of the working B7 connections between neighbouring C7 (operations only;
    C7 and the diagnostics use other connections)."""
    out = []
    for c0, c1 in zip(cps, cps[1:]):
        row = {"from_op": c0["after_op"], "to_op": c1["after_op"]}
        for route in c1["prepared"]:
            row[route] = {}
            for kind, v in c1["prepared"][route].items():
                v0 = c0["prepared"].get(route, {}).get(kind, {"generic": 0, "custom": 0})
                row[route][kind] = {"generic": v["generic"] - v0["generic"], "custom": v["custom"] - v0["custom"]}
        out.append(row)
    return out


MIX = {"I": 0.5, "U": 0.3, "D": 0.2}  # 250 000 / 150 000 / 100 000 of the stream


def standardized_internal(kinds: dict, summ: dict) -> dict:
    """R7: standardization of the MEASURED INTERNAL time of an operation (routing + lookup + write; without the
    transaction commit and the batch remainder) to the stream mix; and K_dir with the mean g(k) time subtracted
    (a sensitivity calculation, not a measurement of the physical cost)."""
    std = {}
    for r, ks in kinds.items():
        if all(ks[k]["op_us"] is not None for k in "IUD"):
            std[r] = sum(MIX[k] * ks[k]["op_us"] for k in "IUD")
    out = {"op_us_standardized": std}
    if "s" in std and "par" in std and "dir" in std:
        out["K_par_internal"] = std["par"] / std["s"]
        out["K_dir_internal"] = std["dir"] / std["s"]
    g = kinds.get("dir", {}).get("g", {})
    w = summ["whole"]
    ops_dir = sum(kinds["dir"][k]["ops"] for k in "IUD") if "dir" in kinds else 0
    if g.get("calls") and ops_dir and w["T_dir"] is not None and w["T_s"]:
        g_per_op = g["us_per_call"] * g["calls"] / ops_dir
        out["g_us_per_op_dir"] = g_per_op
        out["K_dir_minus_g"] = (w["T_dir"] - g_per_op) / w["T_s"]
    return out


def run_loop(ctx: Ctx, a: A6.A6System, ops, model: Model, ever: np.ndarray, *, size: str, vacuums, cps,
             stop_on_error: bool, diag: bool, out: dict) -> tuple[dict, dict]:
    """Fills `out` in place (R3 Codex 249): on any exception the caller still holds the batches, C7, events and the
    stage that failed."""
    conn = G.connect()
    ctx.loop_conn = conn
    nb = len(ops) // BATCH
    sched = schedule(nb, ctx.mutant)
    bs = ctx.reader("s")
    work = {"s": bs, "par": ctx.reader("par"), "dir": ctx.reader("dir")}
    ctx.work = work
    out.update({"schedule_seed": SCHED_SEED, "schedule": [[o, r] for o, r in sched], "batches": [], "events": [],
                "checkpoints": [], "vacuum": [], "after_vacuum_route": [], "after_cp_route": [], "timer_violations": [],
                "apply_errors": {"s": 0, "p": 0, "a7": 0}, "witness_errors": 0, "stage": {"at": "start"}})
    t_maint = 0.0
    vac = {"s": 0.0, "p": 0.0, "a7": 0.0}
    wrng = random.Random(77)
    used = []
    prev = None

    def stop(why: str) -> None:
        out["stopped"] = why
        raise Stop(why)

    if 0 in cps:
        out["stage"] = {"at": "c7", "op": 0}
        rec = checkpoint7(ctx, 0, 0, model, ever, a, work, diag)
        out["checkpoints"].append(rec)
        out["after_cp_route"].append([0, sched[0][1]])
        if stop_on_error and (rec["errors_total"] or rec["prepared_rule_violations"]):
            stop("C7 errors at op 0")
    for b in range(nb):
        start = b * BATCH
        batch = ops[start: start + BATCH]
        expected = ["ins" if k == "I" else "upd" if k == "U" else "del" for k, _, _ in batch]
        o, route = sched[b]
        used.append((o, route))
        bp = work[route]
        times, gots = {}, {}
        for name in ORDERS[o]:
            cl = bs if name == "s" else bp if name == "p" else a
            out["stage"] = {"at": "apply", "b": b, "op": start, "system": name, "route": route if name == "p" else None}
            t = time.perf_counter()
            gots[name] = cl.apply(batch)
            times[name] = time.perf_counter() - t
            out["current_batch"] = {"b": b, "order": o, "route": route, "times": dict(times), "complete": False}
        errs = {name: sum(1 for x, y in zip(g, expected) if x != y) for name, g in gots.items()}
        errs["s"] += bs.errors
        errs["p"] += bp.errors
        errs["a7"] += a.errors
        bs.errors = bp.errors = a.errors = 0
        for name, e in errs.items():
            out["apply_errors"][name] += e
        for i, (kind, kb, payload) in enumerate(batch):
            model.apply(start + i, kind, kb, payload)
        op_no = start + len(batch)
        for name, cl in (("s", bs), ("p", bp)):
            ts_ = cl.batch_timer_ns / 1e9
            if not (0 < ts_ <= times[name]):
                out["timer_violations"].append([b, name, ts_, times[name]])
        if stop_on_error and any(errs.values()):
            stop(f"apply errors at op {op_no}: {errs}")
        if size == "N10" and op_no == A6.FORCED_N10:
            a.lsm.round_pending = True
            log(size, "forced round requested at op", op_no)
        out["stage"] = {"at": "maintain", "b": b, "op": op_no}
        tm = time.perf_counter()
        evs = a.maintain()
        tmaint = time.perf_counter() - tm
        t_maint += tmaint
        for ev in evs:
            wk = A6.witness_keys(a, ev, wrng)
            ev["witness_errors"] = A6.check_witnesses(a, model, wk)
            ev["witnesses"] = len(wk)
            out["witness_errors"] += ev["witness_errors"]
            A6.drop_private(ev)
            ev["after_op"] = ev.get("after_op", op_no)
            out["events"].append(clean(ev))
            if ev["kind"] == "round_start":
                log(size, "round start at op", op_no)
        if stop_on_error and out["witness_errors"]:
            stop(f"witness errors at op {op_no}")
        cnt = Counter(k for k, _, _ in batch)
        out["batches"].append({"b": b, "op_no": op_no, "ops": len(batch), "order": o, "route": route, "prev": prev,
                               "t_s": times["s"], "t_p": times["p"], "t_a7": times["a7"], "t_maint": tmaint,
                               "timer_s": bs.batch_timer_ns / 1e9, "timer_p": bp.batch_timer_ns / 1e9,
                               "nI": cnt["I"], "nU": cnt["U"], "nD": cnt["D"],
                               "events": "+".join(e["kind"] for e in evs)})
        out["current_batch"]["complete"] = True
        prev = route
        if op_no in vacuums:
            out["stage"] = {"at": "vacuum", "op": op_no}
            for key, t_ in (("s", ctx.ts), ("p", ctx.tp), ("a7", ctx.ta)):
                tv = time.perf_counter()
                conn.execute(f"VACUUM {t_}")
                dv = time.perf_counter() - tv
                vac[key] += dv
                out["vacuum"].append([op_no, key, dv])
            out["after_vacuum_route"].append([op_no, sched[b + 1][1] if b + 1 < nb else None])
        if op_no in cps:
            out["stage"] = {"at": "c7", "op": op_no}
            rec = checkpoint7(ctx, len(out["checkpoints"]), op_no, model, ever, a, work, diag)
            out["checkpoints"].append(rec)
            out["after_cp_route"].append([op_no, sched[b + 1][1] if b + 1 < nb else None])
            if stop_on_error and (rec["errors_total"] or rec["prepared_rule_violations"]):
                stop(f"C7 errors at op {op_no}")
            if size in SIZES:
                log(size, "ops", op_no, {k: round(sum(x[k] for x in out["batches"]), 1) for k in ("t_s", "t_p", "t_a7")},
                    "maint", round(t_maint, 1))
    out["balance"] = schedule_balance(used)
    out["vacuum_s"] = vac
    out["maintenance_s"] = t_maint
    out["ops_time_s"] = {k: sum(x[k] for x in out["batches"]) for k in ("t_s", "t_p", "t_a7")}
    out["kinds"] = {r: c.kind_summary() for r, c in work.items()}
    out["summary"] = summarize(out["batches"], vac, t_maint)
    out["plan_increments"] = plan_increments(out["checkpoints"])
    out["standardized_internal"] = standardized_internal(out["kinds"], out["summary"])
    out["stage"] = {"at": "done"}
    conn.close()
    return out, work


# ------------------------------------------------------------------ selfcheck (§3.1)
def scenario_two_routes(ctx: Ctx, work: dict, model: Model) -> dict:
    """INSERT / UPDATE / DELETE through one route, checked through the other, both directions, all partitions; s7
    unchanged."""
    conn = G.connect()
    q = f"SELECT count(*), coalesce(sum(hashtext(k::text || v::text)::bigint), 0) FROM {ctx.ts}"
    before = tuple(conn.execute(q).fetchone())
    rng = random.Random(5150)
    res = []
    par, dr = work["par"], work["dir"]
    for j in range(ctx.nparts):
        ks = []
        while len(ks) < 2:
            kb = uuid.UUID(bytes=rng.randbytes(16), version=4).bytes
            if PH.part_of(kb, ctx.nparts) == j and kb not in model.state and kb not in model.deleted:
                ks.append(kb)
        for kb, (w, r) in zip(ks, ((par, dr), (dr, par))):
            steps = {}
            try:
                g1 = w.apply([("I", kb, "a7scn1")])
                row = r.lookup(kb)
                steps["insert"] = g1 == ["ins"] and row is not None and row[1] == 1
                g2 = r.apply([("U", kb, "a7scn2")])
                row = w.lookup(kb)
                steps["update"] = g2 == ["upd"] and row is not None and row[1] == 2 and row[3] == "a7scn2"
                g3 = w.apply([("D", kb, None)])
                steps["delete"] = g3 == ["del"] and r.lookup(kb) is None
            except psycopg.Error as e:
                steps["exception"] = type(e).__name__
            res.append({"part": j, "writer": w.route, "ok": all(steps.get(k) for k in ("insert", "update", "delete")),
                        **steps})
    after = tuple(conn.execute(q).fetchone())
    conn.close()
    rows = {r: prepared_stats(c.conn) for r, c in work.items()}
    viol = prepared_rule(ctx.tp, rows["par"], rows["dir"])
    return {"cases": len(res), "failed": [x for x in res if not x["ok"]][:10], "s7_unchanged": before == after,
            "prepared_rule_violations": viol, "ok": all(x["ok"] for x in res) and before == after}


def loop_digest(res: dict) -> dict:
    """What the selfcheck keeps of a (possibly interrupted) loop; absent parts stay absent (R2 Codex 249)."""
    cps = res.get("checkpoints", [])
    return {"finished": res.get("stage", {}).get("at") == "done", "stage": res.get("stage"),
            "batches": len(res.get("batches", [])), "apply_errors": res.get("apply_errors"),
            "witness_errors": res.get("witness_errors"), "balance": res.get("balance"),
            "timer_violations": res.get("timer_violations"), "after_vacuum_route": res.get("after_vacuum_route"),
            "after_cp_route": res.get("after_cp_route"), "vacuum": res.get("vacuum"),
            "c7_errors": [c["errors_total"] for c in cps],
            "c7_errors_by_route": [{k: sum(v["errors"].values()) for k, v in c["routes"].items()} for c in cps],
            "c7_samples": [c["samples"] for c in cps],
            "prepared_rule_violations": sum(len(c["prepared_rule_violations"]) for c in cps),
            "event_kinds": dict(Counter(e["kind"] for e in res.get("events", []))),
            "summary_whole": res.get("summary", {}).get("whole"), "kinds": res.get("kinds"),
            "standardized_internal": res.get("standardized_internal"),
            "diag_keys": sorted(cps[-1]["diag"]) if cps and "diag" in cps[-1] else None}


def close_all(ctx: Ctx, a: A6.A6System | None, conn) -> list[str]:
    """Close every resource independently and drop the tables; errors are returned, not raised (R2 Codex 251)."""
    errs: list[str] = []

    def attempt(what: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errs.append(f"{what}: {type(e).__name__}: {str(e)[:150]}")

    for r, c in getattr(ctx, "work", {}).items():
        attempt(f"close {r}", c.conn.close)
    if getattr(ctx, "loop_conn", None) is not None:
        attempt("close loop connection", ctx.loop_conn.close)
    if a is not None:
        attempt("close address", a.lsm.close)
        attempt("close a7 connection", a.conn.close)
    for t in (ctx.ts, ctx.tp, ctx.ta):
        attempt(f"drop {t}", lambda t=t: conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE"))
    attempt("close main connection", conn.close)
    return errs


def run_small(mutant: str | None) -> dict:
    tag = "small"
    n, bits, t_run = 100_000, 3, 3_000
    ref = DATA / "ref_a7_small"
    d = DATA / "a7_small"
    ctx = Ctx(tag, 1 << bits, mutant)
    ctx.work = {}
    out: dict = {"mutant": mutant, "rows_expected": n}
    res: dict = {}
    conn = G.connect()
    a = None
    try:
        info, a = load_three(tag, n, ref, bits, d, t_run)
        out["load"] = info
        keys = [bytes(k) for k in np.load(f"{ref}.keys")]
        digs = [bytes(x) for x in np.load(f"{ref}.dig")]
        out["ref_lengths"] = [len(keys), len(digs)]
        extra = load_extra(DATA / "dyn.copy")
        ops = make_ops(keys, extra, 10_000, 6_000, 4_000, 13)
        ever = np.concatenate([np.load(f"{ref}.keys"), np.load(DATA / "dyn.keys")])
        model = Model(keys, digs)
        try:
            run_loop(ctx, a, ops, model, ever, size="small", vacuums=(10_000,), cps=(0, 10_000, 20_000),
                     stop_on_error=False, diag=mutant is None, out=res)
            out["scenario"] = scenario_two_routes(ctx, ctx.work, model)
        except psycopg.Error as e:
            out["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
            out["exception_type"] = type(e).__name__
            out["exception_stage"] = res.get("stage")
        out["loop"] = loop_digest(res)
    finally:
        out["cleanup_errors"] = close_all(ctx, a, conn)
    return out


def rows_ok(r: dict) -> bool:
    rows = r.get("load", {}).get("rows") or {}
    n = r.get("rows_expected")
    return bool(rows) and all(v == n for v in rows.values()) and r.get("ref_lengths") == [n, n]


def small_ok(r: dict) -> dict:
    lp = r.get("loop") or {}
    sc = r.get("scenario") or {}
    bal = lp.get("balance")
    ek = lp.get("event_kinds") or {}
    samples = lp.get("c7_samples") or [{}]
    return {
        "no_exception": "exception" not in r,
        "loop_finished": bool(lp.get("finished")),
        "apply_errors_0": lp.get("apply_errors") is not None and not any(lp["apply_errors"].values()),
        "witness_0": lp.get("witness_errors") == 0,
        "c7_errors_0": len(lp.get("c7_errors") or []) == 3 and not any(lp["c7_errors"]),
        "deleted_sample_last_c7": samples[-1].get("deleted", 0) > 0,
        "p1_balance": bal is not None and bal["ok"] is True,
        "p2_prepared_rule": lp.get("finished") is True and lp["prepared_rule_violations"] == 0
        and sc.get("prepared_rule_violations") == [],
        "p3_timers": lp.get("finished") is True and lp["timer_violations"] == [],
        "a7_seal_and_round_step": ek.get("round_step", 0) > 0 and ek.get("seal", 0) > 0,
        "scenario": sc.get("cases", 0) == 16 and sc.get("ok") is True,
        "rows": rows_ok(r),
    }


def caught(m: str, r: dict) -> bool:
    """Positive evidence of the designated check only; a check that did not run is not a catch (R2 Codex 249)."""
    lp = r.get("loop") or {}
    sc = r.get("scenario")
    if m == "M1":
        return lp.get("prepared_rule_violations", 0) > 0 or bool(sc and sc.get("prepared_rule_violations"))
    if m == "M2":
        st_ = r.get("exception_stage") or {}
        exc = (r.get("exception_type") == "CheckViolation" and st_.get("at") == "apply"
               and st_.get("system") == "p" and st_.get("route") == "dir")
        return exc or (lp.get("apply_errors") or {}).get("p", 0) > 0 or any(lp.get("c7_errors") or [])
    if m in ("M3", "M4"):
        b = lp.get("balance")
        return b is not None and b["ok"] is False
    if m == "M5":
        return bool(sc and sc.get("cases", 0) > 0 and sc.get("ok") is False) or any(lp.get("c7_errors") or [])
    raise ValueError(m)


def caught_selftest() -> dict:
    """A result with only a foreign exception (no designated check ran) must give caught = False for every mutant."""
    foreign = {"exception_type": "OperationalError", "exception_stage": {"at": "apply", "system": "s", "route": None},
               "loop": {"finished": False, "batches": 3, "apply_errors": {"s": 0, "p": 0, "a7": 0}, "balance": None,
                        "c7_errors": [0], "prepared_rule_violations": 0}}
    wrong_stage = {"exception_type": "CheckViolation", "exception_stage": {"at": "apply", "system": "a7", "route": None},
                   "loop": {"finished": False, "apply_errors": {"s": 0, "p": 0, "a7": 0}, "c7_errors": [0]}}
    bad = {m: True for m in MUTANTS if caught(m, foreign)}
    if caught("M2", wrong_stage):
        bad["M2_wrong_stage"] = True
    return {"ok": not bad, "bad": bad}


def cmd_selfcheck() -> None:
    check_protocol()
    A6.require_selfcheck()
    out = {"impl_sha256": impl_hashes(), "summarize_selftest": summarize_selftest(), "caught_selftest": caught_selftest()}
    log("summarize selftest", out["summarize_selftest"], "caught selftest", out["caught_selftest"])
    r = run_small(None)
    out["ok_run"] = r
    out["ok_checks"] = small_ok(r)
    log("ok run", out["ok_checks"])
    for m in MUTANTS:
        rm = run_small(m)
        out[f"mutant_{m}"] = {"caught": caught(m, rm), "checks": small_ok(rm), "exception": rm.get("exception"),
                              "exception_stage": rm.get("exception_stage"), "loop": rm.get("loop"),
                              "scenario": rm.get("scenario")}
        log("mutant", m, "caught" if out[f"mutant_{m}"]["caught"] else "NOT CAUGHT", rm.get("exception"))
    out["all_ok"] = (out["summarize_selftest"]["ok"] and out["caught_selftest"]["ok"] and all(out["ok_checks"].values())
                     and all(out[f"mutant_{m}"]["caught"] for m in MUTANTS))
    G.write_json(RES / "a7_selfcheck.json", out)
    log("selfcheck all_ok", out["all_ok"])
    if not out["all_ok"]:
        raise SystemExit(1)


def require_selfcheck() -> None:
    f = RES / "a7_selfcheck.json"
    if not f.exists():
        raise SystemExit("no a7_selfcheck.json")
    sc = json.loads(f.read_text(encoding="utf-8"))
    if not sc.get("all_ok") or sc.get("impl_sha256") != impl_hashes():
        raise SystemExit("A7 selfcheck failed or the implementation changed since the selfcheck")


# ------------------------------------------------------------------ dynamics N1 / N10
def cmd_dynamics(size: str, rep: str) -> None:
    check_protocol()
    A6.require_selfcheck()
    require_selfcheck()
    n = SIZES[size]
    tag = size.lower()
    name = f"a7_{tag}_{rep}"
    conn = G.connect()
    out: dict = {"size": size, "rep": rep, "rows": n, "impl_sha256": impl_hashes(),
                 "settings": {k: conn.execute(f"SHOW {k}").fetchone()[0] for k in SETTINGS}, "stage": "start"}
    free = shutil.disk_usage(G.require_data_dir()).free
    out["free_bytes_before"] = free
    if size == "N10" and free < MIN_FREE_N10:
        out["status"] = "НЕ ВЫПОЛНЕНО: мало места на томе данных PostgreSQL"
        G.write_json(RES / f"{name}.json", out)
        raise SystemExit(f"free {free / 1e9:.1f} GB < {MIN_FREE_N10 / 1e9:.0f} GB on the volume of {G.PG_DATA}")
    bits = L.bits_for(n, L.SIGMA)
    ctx = Ctx(tag, 1 << bits, None)
    ctx.work = {}
    ref = DATA / f"ref_a7_{tag}"
    loop: dict = {}
    a = None
    try:
        out["stage"] = "load"
        log(size, rep, "load three tables, partitions", 1 << bits)
        info, a = load_three(tag, n, ref, bits, DATA / f"a7_{tag}", A6.T_RUN)
        out["load"] = info
        log(size, "load", {k: v for k, v in info.items() if k != "pages"}, info["pages"])
        keys = [bytes(k) for k in np.load(f"{ref}.keys")]
        digs = [bytes(x) for x in np.load(f"{ref}.dig")]
        out["ref_lengths"] = [len(keys), len(digs)]
        if any(v != n for v in info["rows"].values()) or out["ref_lengths"] != [n, n]:
            raise Stop("row counts or reference lengths differ from N: НЕ УСТАНОВЛЕНО")
        extra = load_extra(DATA / "dyn.copy")
        ops = make_ops(keys, extra, 250_000, 150_000, 100_000, 13)
        out["ops_sha256"] = ops_sha256(ops)
        out["stream_matches_a4"] = out["ops_sha256"].startswith(A6.STREAM_SHA[size])
        if not out["stream_matches_a4"]:
            raise Stop("stream differs from A4: НЕ УСТАНОВЛЕНО")
        out["stage"] = "function_check"
        ins_keys = np.frombuffer(b"".join(kb for kind, kb, _ in ops if kind == "I"), np.uint8).reshape(-1, 16)
        fc = A6.function_check(conn, ctx.ta, a.lsm, ins_keys, None)
        out["function_check"] = fc
        log(size, "function check", {k: fc[k] for k in ("rows", "bad_numpy", "bad_scalar", "bad_insert_keys", "ok")})
        if not fc["ok"]:
            raise Stop("partition function check failed: НЕ УСТАНОВЛЕНО")
        out["sizes_start"] = sizes_now(conn, tag, a)
        out["heights"] = heights(conn, tag)
        out["partitions_start"] = {"p7": A6.part_sizes(conn, ctx.tp), "a7": A6.part_sizes(conn, ctx.ta)}
        ever = np.concatenate([np.load(f"{ref}.keys"), np.load(DATA / "dyn.keys")])
        model = Model(keys, digs)
        del keys, digs
        out["stage"] = "loop"
        run_loop(ctx, a, ops, model, ever, size=size, vacuums=VACUUMS, cps=CP_AT, stop_on_error=True, diag=True,
                 out=loop)
        out.update(loop)
        out["stage"] = "end"
        out["sizes_end"] = sizes_now(conn, tag, a)
        out["partitions_end"] = {"p7": A6.part_sizes(conn, ctx.tp), "a7": A6.part_sizes(conn, ctx.ta)}
        out["rss"] = rss_mb()
        out["status"] = "ok"
    except BaseException as e:  # R3 Codex 249, R1 Codex 251: keep what the loop gathered, the stage and the cause
        out.update({k: v for k, v in loop.items() if k not in out})
        out["failure_stage"] = loop.get("stage")
        out["current_batch"] = loop.get("current_batch")
        out["status"] = f"ОСТАНОВЛЕН: {type(e).__name__}: {str(e)[:300]}"
        G.write_json(RES / f"{name}_FAILED.json", out)
        out["cleanup_errors"] = close_all(ctx, a, conn)
        if out["cleanup_errors"]:
            G.write_json(RES / f"{name}_FAILED.json", out)
        log(size, rep, "FAILED", out["status"], "stage", out.get("stage"), out["failure_stage"])
        raise SystemExit(2)
    G.write_json(RES / f"{name}.json", out)  # R2 Codex 251: the result is on disk before any cleanup
    out["cleanup_errors"] = close_all(ctx, a, conn)
    if out["cleanup_errors"]:
        G.write_json(RES / f"{name}.json", out)
        log(size, rep, "cleanup errors", out["cleanup_errors"])
    w = out["summary"]["whole"]
    f = out["summary"]["full"]
    log(size, rep, "done", {k: round(w[k], 3) for k in ("K_par", "K_dir", "K_route", "K_par_m", "K_dir_m", "a7_s")},
        {k: round(f[k], 3) for k in ("F_a7/F_s", "F_p_mix/F_s", "F_a7/F_par_hat")})


if __name__ == "__main__":
    G.check_cluster()  # SQL and RCM_PG_DATA address the same cluster (free-space check)
    if sys.argv[1] == "selfcheck":
        cmd_selfcheck()
    elif sys.argv[1] == "dynamics":
        cmd_dynamics(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
