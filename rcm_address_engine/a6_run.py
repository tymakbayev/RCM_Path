"""A6 runs (protocol A6_protocol.md edition 2): selfcheck (offline test + scenarios + mutants), dynamics N1|N10,
replay N1|N10, open-probe. Run with .venv-pg:
  a6_run.py selfcheck | dynamics N1 | dynamics N10 | replay N1 | replay N10"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import time
import tracemalloc
import uuid
from pathlib import Path

import numpy as np
import psycopg

import a2_pg as G
import a4_lsm as L4
import a6_lsm as L
import a6_pghash as PH
from a2_dyn import BATCH, N_EXPLAIN, OPS, SAMPLE, Model, load_extra, make_ops, ops_sha256
from a2_runs import DATA, RES, absent_keys, log, sha16
from a4_run import A4System, Control, moving_update, wide_payload
from a5_run import clean, rss_mb

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT.parent / "docs" / "experiments" / "A6_protocol.md"
PY = Path(sys.executable)
VACUUMS = (240_000, 480_000)
CP_EVERY = 50_000
F_MAJOR = 0.1
T_RUN = 30_000
FORCED_N10 = 150_000
STREAM_SHA = {"N1": "1f685ebe", "N10": "1bdeb7fb"}
COLS = "ctid, v, seq, payload"
# code = protocol letter: a..d = (а)..(г) A4; e=(д) f2=(е′) g=(ж) z=(з) i=(и) k=(к) k2=(к′) l=(л) m n o p r
MUTANTS = ("a", "b", "c", "d", "e", "f2", "g", "z", "i", "k", "k2", "l", "m", "n", "o", "p", "r")
MUST_CATCH = tuple(m for m in MUTANTS if m != "k")  # (k) is equivalent (A5 addendum §1)
OFFLINE_MUTANTS = ("e", "f2", "m", "n", "o", "p", "r", "b", "g", "i", "z")


def check_protocol() -> None:
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != (RES / "a6_protocol.sha256").read_text().split()[0]:
        raise SystemExit(f"A6 protocol changed: {sha}")


def page_of_ctid(ctid: str) -> int:
    return int(ctid[1:ctid.index(",")])


# ------------------------------------------------------------------ partitioned tables (§2.2, §3)
def create_parted(conn, table: str, nparts: int) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    conn.execute(f"CREATE TABLE {table} (seq bigint NOT NULL, k uuid NOT NULL, v integer NOT NULL, payload text NOT NULL)"
                 " PARTITION BY HASH (k)")
    for j in range(nparts):
        conn.execute(f"CREATE TABLE {table}_p{j} PARTITION OF {table} FOR VALUES WITH (MODULUS {nparts}, REMAINDER {j})"
                     " WITH (autovacuum_enabled = false)")


def load_parted(conn, table: str, n: int, ref: Path, nparts: int, workers: int = 6) -> dict:
    """COPY stream positions [0, n) from a2_gen into the partitioned parent (the server routes rows)."""
    t0 = time.perf_counter()
    create_parted(conn, table, nparts)
    proc = subprocess.Popen([str(G.GEN_PY), str(G.GEN), "0", str(n), "--ref", str(ref), "--workers", str(workers)],
                            stdout=subprocess.PIPE, cwd=str(ROOT))
    with conn.cursor() as cur, cur.copy(f"COPY {table} (seq, k, v, payload) FROM STDIN") as cp:
        while chunk := proc.stdout.read(1 << 20):
            cp.write(chunk)
    if proc.wait() != 0:
        raise RuntimeError("a2_gen failed")
    t1 = time.perf_counter()
    conn.execute(f"VACUUM (ANALYZE) {table}")
    return {"copy_s": t1 - t0, "vacuum_s": time.perf_counter() - t1}


def remainders(conn, table: str) -> dict[str, int]:
    """Leaf partition name -> its remainder (from the partition bound), R2."""
    out = {}
    for name, bound in conn.execute(
            "SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
            "WHERE i.inhparent = %s::regclass", (table,)).fetchall():
        out[name] = int(re.search(r"remainder (\d+)", bound).group(1))
    return out


def read_part(conn, rel: str) -> tuple[np.ndarray, np.ndarray]:
    """All rows of one partition: binary COPY of k and the ctid page (§2.5)."""
    sql = f"COPY (SELECT k, (ctid::text::point)[0]::int4 FROM {rel}) TO STDOUT (FORMAT binary)"
    buf = bytearray()
    with conn.cursor() as cur, cur.copy(sql) as cp:
        for data in cp:
            buf += data
    dt = np.dtype([("n", ">i2"), ("l1", ">i4"), ("k", "V16"), ("l2", ">i4"), ("p", ">i4")])
    n = max(0, (len(buf) - 19 - 2) // dt.itemsize)
    arr = np.frombuffer(bytes(buf), dt, count=n, offset=19) if n else np.zeros(0, dt)
    keys = np.frombuffer(arr["k"].tobytes(), np.uint8).reshape(n, 16) if n else np.zeros((0, 16), np.uint8)
    return keys, arr["p"].astype(np.uint32)


def part_sizes(conn, table: str) -> list[list[int]]:
    """[partition, pages, exact live rows] (count(*), outside measured work)."""
    rows = conn.execute("SELECT c.relname, pg_relation_size(c.oid) / 8192 FROM pg_inherits i JOIN pg_class c "
                        "ON c.oid = i.inhrelid WHERE i.inhparent = %s::regclass", (table,)).fetchall()
    cnt = dict(conn.execute(f"SELECT tableoid::regclass::text, count(*) FROM {table} GROUP BY 1").fetchall())
    out = [[int(r[0].rsplit("_p", 1)[1]), int(r[1]), int(cnt.get(r[0], 0))] for r in rows]
    return sorted(out)


def parted_bytes(conn, table: str) -> dict:
    main, tot = conn.execute("SELECT sum(pg_relation_size(relid)), sum(pg_table_size(relid)) FROM pg_partition_tree(%s) "
                             "WHERE isleaf", (table,)).fetchone()
    return {"main_bytes": int(main), "table_bytes": int(tot)}


def function_check(conn, table: str, lsm: L.AddressLSM6, extra_keys: np.ndarray, scalar_sample: int | None) -> dict:
    """§2.3 (R2): g(k) (numpy and scalar) against the remainder of the partition holding the row, for all rows; and
    against satisfies_hash_partition for the stream's insert keys."""
    t0 = time.perf_counter()
    rem = remainders(conn, table)
    S = lsm.n_parts
    bad_np = bad_sc = rows = checked_sc = 0
    for rel, r in sorted(rem.items(), key=lambda x: x[1]):
        keys, _ = read_part(conn, rel)
        rows += len(keys)
        bad_np += int(np.sum(lsm.part_np(keys) != r))
        sel = keys if scalar_sample is None else keys[:scalar_sample]
        bad_sc += sum(1 for k in sel if lsm._part_uncached(bytes(k)) != r)
        checked_sc += len(sel)
    oid = conn.execute("SELECT %s::regclass::oid", (table,)).fetchone()[0]
    bad_ins = 0
    gx = lsm.part_np(extra_keys)
    for i in range(0, len(extra_keys), 50_000):
        ks = [uuid.UUID(bytes=bytes(k)) for k in extra_keys[i:i + 50_000]]
        bad_ins += conn.execute("SELECT count(*) FROM unnest(%s::uuid[], %s::int[]) AS t(k, r) "
                                "WHERE NOT satisfies_hash_partition(%s, %s, r, k)",
                                (ks, [int(x) for x in gx[i:i + 50_000]], oid, S)).fetchone()[0]
    t = time.perf_counter()
    sc_ins = sum(1 for k, g in zip(extra_keys, gx.tolist()) if lsm._part_uncached(bytes(k)) != g)
    us = (time.perf_counter() - t) / max(1, len(extra_keys)) * 1e6
    out = {"rows": rows, "bad_numpy": bad_np, "scalar_checked": checked_sc, "bad_scalar": bad_sc,
           "insert_keys": int(len(extra_keys)), "bad_insert_keys": int(bad_ins), "scalar_us_per_key": us,
           "insert_scalar_checked": int(len(extra_keys)), "bad_insert_scalar_vs_numpy": sc_ins,
           "s": time.perf_counter() - t0}
    out["ok"] = (bad_np == 0 and bad_sc == 0 and bad_ins == 0 and sc_ins == 0 and checked_sc == rows
                 if scalar_sample is None else bad_np == 0 and bad_sc == 0 and bad_ins == 0 and sc_ins == 0)
    return out


def setup_control(conn, ctable: str, atable: str, nparts: int, n: int) -> dict:
    """§3 (R7): same partitioning, filled on the server partition by partition in ctid order; layout checked by key."""
    t0 = time.perf_counter()
    create_parted(conn, ctable, nparts)
    for j in range(nparts):
        conn.execute(f"INSERT INTO {ctable}_p{j} SELECT * FROM {atable}_p{j} ORDER BY ctid")
    t1 = time.perf_counter()
    same = total = 0
    for j in range(nparts):
        ka, pa = read_part(conn, f"{atable}_p{j}")
        kc, pc = read_part(conn, f"{ctable}_p{j}")
        da = dict(zip(map(bytes, ka), pa.tolist()))
        same += sum(1 for k, p in zip(map(bytes, kc), pc.tolist()) if da.get(k) == p)
        total += len(ka)
    conn.execute(f"DROP SEQUENCE IF EXISTS {ctable}_seq")
    conn.execute(f"CREATE SEQUENCE {ctable}_seq START {n + 1}")
    conn.execute(f"CREATE UNIQUE INDEX {ctable}_bt ON {ctable} USING btree (k)")
    conn.execute(f"VACUUM (ANALYZE) {ctable}")
    return {"fill_s": t1 - t0, "layout_same": same, "layout_total": total, "layout_share": same / max(1, total),
            "pages_a": part_sizes(conn, atable), "pages_c": part_sizes(conn, ctable), "s": time.perf_counter() - t0}


def btree_heights(conn, ctable: str) -> list[int]:
    out = []
    for (rel,) in conn.execute("SELECT relid::regclass::text FROM pg_partition_tree(%s) WHERE isleaf",
                               (f"{ctable}_bt",)).fetchall():
        out.append(int(conn.execute("SELECT level FROM bt_metap(%s)", (rel,)).fetchone()[0]) + 1)
    return out


# ------------------------------------------------------------------ A6 system on PostgreSQL
class A6System(A4System):
    """A4System with every statement sent to the partition g(k) (§2.4)."""

    def __init__(self, name: str, table: str, seq: str, d: Path, lsm: L.AddressLSM6, conn=None) -> None:
        # the A4System attributes, without its unconditional connect (R11: the open probe passes its connection)
        self.name, self.table, self.seq, self.dir = name, table, seq, d
        self.kind = "A6"
        self.conn = conn if conn is not None else G.connect()
        self.conn.prepared_max = 1_000_000
        self.delta = None
        self.errors = 0
        self.reader_mode = False
        self.reader_refresh = True
        self.last_candidates = 0
        self.cand_hist: dict[int, int] = {}
        self.last_pages: dict[bytes, int] = {}
        self._q: dict[tuple, str] = {}
        self.writes_since_read: dict[int, int] = {}
        self.wrng = random.Random(77)
        self.lsm = None
        self.attach(lsm)

    def attach(self, lsm: L.AddressLSM6, recount: bool = False) -> None:
        self.lsm = lsm
        if lsm is not None:
            lsm.step_hook = self.step_hook
            if recount:  # Codex 241 R2: exact live count after a writer reopen, before any maintenance
                lsm.n_live = int(self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])

    def step_hook(self, s: int, keys: np.ndarray, pages: np.ndarray) -> list[bytes]:
        """R1: witnesses of a round step chosen while its pairs exist: keys on the first and the last page of the
        partition and 100 random keys of it (the tail keys of the partition are added by the stand)."""
        if not len(keys):
            return []
        first = np.flatnonzero(pages == pages.min())[:25]
        last = np.flatnonzero(pages == pages.max())[:25]
        idx = list(dict.fromkeys(first.tolist() + last.tolist()))
        rest = [i for i in self.wrng.sample(range(len(keys)), min(len(keys), 150 + len(idx))) if i not in set(idx)]
        idx += rest[:max(0, 150 - len(idx))]
        return [bytes(keys[i]) for i in idx]

    def reopen(self) -> None:
        self.conn.close()
        self.conn = G.connect()
        self.conn.prepared_max = 1_000_000

    def rel(self, j: int) -> str:
        return f"{self.table}_p{j}"

    def query(self, kb: bytes, writer: bool = False):
        if self.reader_refresh and self.reader_mode:
            self.lsm.tail.refresh()
        tg = self.lsm.targets(kb, writer)
        self.last_candidates = len(tg)
        if not tg:
            return None
        key = tuple(j for j, _ in tg)
        if key not in self._q:
            self._q[key] = " UNION ALL ".join(
                f"(SELECT {COLS} FROM {self.rel(j)} WHERE ctid >= %s::tid AND ctid < %s::tid AND k = %s)" for j in key) \
                + " LIMIT 1"
        k = uuid.UUID(bytes=kb)
        params = []
        for _, p in tg:
            params += [f"({p},0)", f"({p + 1},0)", k]
        return self._q[key], tuple(params)

    def lookup(self, kb: bytes, writer: bool = False):
        q = self.query(kb, writer)
        self.cand_hist[self.last_candidates] = self.cand_hist.get(self.last_candidates, 0) + 1
        if q is None:
            return None
        rows = self.conn.execute(q[0], q[1], prepare=True).fetchall()
        return rows[0] if rows else None

    def apply(self, ops, *, rollback: bool = False) -> list[str]:
        """One writer batch (A4 §2.7) with statements on the partition g(k)."""
        out = []
        lsm = self.lsm
        lsm.batch_no += 1
        self.last_pages = {}
        touched: dict[int, int] = {}
        try:
            with self.conn.transaction():
                for kind, kb, payload in ops:
                    k = uuid.UUID(bytes=kb)
                    rel = self.rel(lsm.part(kb))
                    row = self.lookup(kb, writer=True)
                    if kind == "I":
                        if row is not None:
                            out.append("dup")
                            continue
                        r = self.conn.execute(f"INSERT INTO {rel} (seq, k, v, payload) VALUES (nextval('{self.seq}'), %s, 1, %s)"
                                              " RETURNING ctid", (k, payload), prepare=True).fetchone()
                        lsm.opseq += 1
                        lsm.tail.append(L4.I_, kb, page_of_ctid(r[0]), lsm.opseq)
                        touched[lsm.part(kb)] = touched.get(lsm.part(kb), 0) + 1
                        out.append("ins")
                        continue
                    if row is None:
                        out.append("none")
                        continue
                    if kind == "U":
                        r = self.conn.execute(f"UPDATE {rel} SET payload = %s, v = v + 1, seq = nextval('{self.seq}') "
                                              "WHERE ctid = %s::tid AND k = %s RETURNING ctid",
                                              (payload, row[0], k), prepare=True).fetchone()
                        if r is None:
                            self.errors += 1
                            out.append("lost")
                            continue
                        lsm.opseq += 1
                        lsm.tail.append(L4.U_, kb, page_of_ctid(r[0]), lsm.opseq)
                        self.last_pages[kb] = page_of_ctid(r[0])
                        touched[lsm.part(kb)] = touched.get(lsm.part(kb), 0) + 1
                        out.append("upd")
                    else:
                        cur = self.conn.execute(f"DELETE FROM {rel} WHERE ctid = %s::tid AND k = %s", (row[0], k),
                                                prepare=True)
                        if cur.rowcount != 1:
                            self.errors += 1
                        lsm.opseq += 1
                        lsm.tail.append(L4.D_, kb, 0, lsm.opseq)
                        out.append("del")
                if rollback:
                    raise psycopg.Rollback()
        except Exception:
            lsm.tail.rollback(lsm.batch_no)
            raise
        if rollback:
            lsm.tail.rollback(lsm.batch_no)
        else:
            lsm.ops_total += lsm.tail.commit(lsm.batch_no)
            for j, c in touched.items():
                self.writes_since_read[j] = self.writes_since_read.get(j, 0) + c
            if lsm.n_live is not None:  # R9: exact live count, committed INSERT / DELETE only
                lsm.n_live += out.count("ins") - out.count("del")
        return out

    def read_fn(self, s: int):
        rel = self.rel(s)
        keys, pages = read_part(self.conn, rel)
        rb = int(self.conn.execute("SELECT pg_relation_size(%s)", (rel,)).fetchone()[0])
        info = {"read_bytes": rb, "pages": rb // 8192, "writes_since_read": self.writes_since_read.get(s, 0)}
        self.writes_since_read[s] = 0
        return keys, pages, info

    def maintain(self, live_rows: int | None = None) -> list[dict]:
        return self.lsm.boundary(self.read_fn)

    def final_round(self) -> list[dict]:
        return self.lsm.run_round_now(self.read_fn)

    def sizes(self) -> dict:
        b = self.lsm.nbytes()
        return {"address_main": b, "address_total": b, "table_bytes": parted_bytes(self.conn, self.table)["table_bytes"],
                "levels": self.lsm.levels(), "runs": [[r.id, r.prio, r.tier, r.n, r.pre] for r in self.lsm.runs],
                "tail_committed_keys": len(self.lsm.tail.committed), "job": None if self.lsm.job is None else
                [self.lsm.job.tier, self.lsm.job.done, 1 << self.lsm.job.bits],
                "round": None if self.lsm.round is None else self.lsm.round["s"]}


class Control6(Control):
    """Main control (§3): the same partitioned table with a local unique B-tree; lookups and writes through the
    parent by key (PostgreSQL prunes partitions)."""

    def __init__(self, name: str, table: str, seq: str) -> None:
        super().__init__(name, table, seq)
        self.conn.prepared_max = 1_000_000
        self.nparts = 1

    def reopen(self) -> None:
        super().reopen()
        self.conn.prepared_max = 1_000_000

    def sizes(self) -> dict:
        main, tot = self.conn.execute("SELECT sum(pg_relation_size(relid)), sum(pg_total_relation_size(relid)) "
                                      "FROM pg_partition_tree(%s) WHERE isleaf", (f"{self.table}_bt",)).fetchone()
        return {"address_main": int(main), "address_total": int(tot),
                "table_bytes": parted_bytes(self.conn, self.table)["table_bytes"]}


def setup_a6(conn, table: str, n: int, ref: Path, d: Path, nparts: int | None = None, **kw) -> A6System:
    bits = kw.pop("s_base_bits", None)
    if bits is None:
        bits = L.bits_for(n, kw.get("sigma", L.SIGMA))
    load_parted(conn, table, n, ref, 1 << bits)
    conn.execute(f"DROP SEQUENCE IF EXISTS {table}_seq")
    conn.execute(f"CREATE SEQUENCE {table}_seq START {n + 1}")
    ks, ps = [], []
    for j in range(1 << bits):
        k, p = read_part(conn, f"{table}_p{j}")
        ks.append(k)
        ps.append(p)
    lsm = L.AddressLSM6.create(d, np.concatenate(ks), np.concatenate(ps), n, s_base_bits=bits, **kw)
    lsm.n_live = int(sum(len(k) for k in ks))
    return A6System(table, table, f"{table}_seq", d, lsm)


# ------------------------------------------------------------------ checkpoints with per-partition statistics
STATIO = ("SELECT sum(heap_blks_read + heap_blks_hit), sum(coalesce(idx_blks_read + idx_blks_hit, 0)), "
          "sum(coalesce(toast_blks_read + toast_blks_hit, 0)) FROM pg_statio_user_tables "
          "WHERE relid IN (SELECT relid FROM pg_partition_tree(%s) WHERE isleaf)")


def statio_parts(client_conn, reader_conn, table: str, run_batch) -> dict:
    def read():
        reader_conn.execute("SELECT pg_stat_clear_snapshot()")
        return np.array(reader_conn.execute(STATIO, (table,)).fetchone(), dtype=np.int64)
    client_conn.execute("SELECT pg_stat_force_next_flush()")
    time.sleep(0.2)
    before = read()
    n = run_batch()
    client_conn.execute("SELECT pg_stat_force_next_flush()")
    after, waited = read(), 0.0
    while waited < 5.0:
        time.sleep(0.2)
        waited += 0.2
        again = read()
        if np.array_equal(again, after):
            break
        after = again
    d = after - before
    return {"queries": n, "heap_per_q": float(d[0]) / n, "index_per_q": float(d[1]) / n, "toast_per_q": float(d[2]) / n}


def direct_control(conn, ctable: str, nparts: int, keys: list[bytes]) -> dict:
    """Diagnostic control (§3, R7): lookup in partition g(k) by its local B-tree; g(k) cost included."""
    times, warm, bufs, plan = [], [], [], []
    cache: dict[bytes, int] = {}
    for kb in keys:  # cold: g(k) computed for every lookup
        t = time.perf_counter_ns()
        j = PH.part_of(kb, nparts)
        cache[kb] = j
        conn.execute(f"SELECT {COLS} FROM {ctable}_p{j} WHERE k = %s", (uuid.UUID(bytes=kb),), prepare=True).fetchall()
        times.append((time.perf_counter_ns() - t) / 1000)
    for kb in keys:  # warm: g(k) from a cache, as the A6 client's LRU
        t = time.perf_counter_ns()
        j = cache[kb]
        conn.execute(f"SELECT {COLS} FROM {ctable}_p{j} WHERE k = %s", (uuid.UUID(bytes=kb),), prepare=True).fetchall()
        warm.append((time.perf_counter_ns() - t) / 1000)
    for kb in keys[:500]:
        j = PH.part_of(kb, nparts)
        e = G.explain(conn, f"SELECT {COLS} FROM {ctable}_p{j} WHERE k = %s", (uuid.UUID(bytes=kb),))
        bufs.append(e["buffers"])
        plan.append(e["planning_buffers"])
    return {"client_us_cold_g": G.pct(times), "client_us_warm_g": G.pct(warm), "buffers": float(np.mean(bufs)),
            "planning_buffers": float(np.mean(plan))}


def explain_generic(conn, cache: dict, sql: str, params: tuple) -> int:
    """Executor buffers of the GENERIC plan of the statement as the client prepares it (R12)."""
    if sql not in cache:
        name = f"g6_{len(cache)}"
        i = 0
        out = []
        for part in sql.split("%s"):
            out.append(part)
            i += 1
            out.append(f"${i}")
        out = out[:-1]
        conn.execute(f"PREPARE {name} AS " + "".join(out))
        cache[sql] = name
    lits = ", ".join(f"'{x}'" for x in params)
    res = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) EXECUTE {cache[sql]}({lits})").fetchone()[0]
    top = res[0]["Plan"]
    return top.get("Shared Hit Blocks", 0) + top.get("Shared Read Blocks", 0)


def checkpoint6(cp_no: int, name: str, op_no: int, systems, model: Model, ever_keys: np.ndarray) -> dict:
    rng = random.Random(21 + cp_no)
    live = rng.sample(model.live, min(SAMPLE, len(model.live)))
    dele = sorted(model.deleted)
    random.Random(41 + cp_no).shuffle(dele)
    dele = dele[:SAMPLE]
    absent = absent_keys(ever_keys, SAMPLE, 61 + cp_no)
    rec = {"cp": cp_no, "name": name, "after_op": op_no, "live_keys": len(model.state), "deleted_keys": len(model.deleted),
           "systems": {}}
    for s in systems:
        err = {"live": 0, "deleted": 0, "absent": 0}
        times = []
        for kb in live:
            t = time.perf_counter_ns()
            row = s.lookup(kb)
            times.append((time.perf_counter_ns() - t) / 1000)
            st = model.state[kb]
            if row is None or row[1] != st[0] or sha16(row[3]) != st[1]:
                err["live"] += 1
        err["deleted"] = sum(1 for kb in dele if s.lookup(kb) is not None)
        err["absent"] = sum(1 for kb in absent if s.lookup(kb) is not None)
        by_cat: dict[str, list[float]] = {}
        plan = []
        for kb in live[:N_EXPLAIN]:
            q = s.query(kb) if s.kind == "A6" else s.client.query(kb)
            if q is None:
                continue  # no candidate: already counted as a live-key error above
            e = G.explain(s.conn, *q)
            by_cat.setdefault(model.state[kb][2], []).append(e["buffers"])
            plan.append(e["planning_buffers"])
        cat_means = {c: float(np.mean(v)) for c, v in by_cat.items()}
        gconn = G.connect()
        gconn.execute("SET plan_cache_mode = force_generic_plan")
        gcache: dict = {}
        generic = []
        for kb in live[:200]:
            q = s.query(kb) if s.kind == "A6" else s.client.query(kb)
            if q is not None:
                generic.append(explain_generic(gconn, gcache, *q))
        gconn.close()

        def mean_buf(keys):
            vals = []
            for kb in keys[:500]:
                q = s.query(kb) if s.kind == "A6" else s.client.query(kb)
                vals.append(0 if q is None else G.explain(s.conn, *q)["buffers"])
            return float(np.mean(vals)) if vals else 0.0
        reader = G.connect()
        pk = live[:N_EXPLAIN]
        st = statio_parts(s.conn, reader, s.table, lambda: [s.lookup(kb) for kb in pk] and len(pk))
        reader.close()
        rec["systems"][s.name] = {
            "errors": err, "client_us": G.pct(times), "buffers_by_category": cat_means,
            "buffers_equal_mix": float(np.mean(list(cat_means.values()))),
            "buffers_raw_mix": float(np.mean([b for v in by_cat.values() for b in v])),
            "planning_buffers": float(np.mean(plan)), "buffers_generic_plan_200": float(np.mean(generic)) if generic else None,
            "category_counts": {c: len(v) for c, v in by_cat.items()},
            "buffers_deleted": mean_buf(dele), "buffers_absent": mean_buf(absent), "statio_live": st, "sizes": s.sizes()}
        sz = rec["systems"][s.name]["sizes"]
        rec["systems"][s.name]["address_main_per_live_key"] = sz["address_main"] / len(model.state)
        s.errors += sum(err.values())
        if s.kind == "C":
            rec["systems"][s.name]["direct_diagnostic"] = direct_control(s.conn, s.table, s.nparts, live[:N_EXPLAIN])
    log("checkpoint", cp_no, name, op_no, {k: (v["errors"], round(v["buffers_equal_mix"], 3),
                                              round(v["address_main_per_live_key"], 3)) for k, v in rec["systems"].items()})
    return rec


# ------------------------------------------------------------------ witnesses (§3)
def state_ok(s, model: Model, kb: bytes) -> bool:
    row = s.lookup(kb)
    st = model.state.get(kb)
    return (row is None) if st is None else (row is not None and row[1] == st[0] and sha16(row[3]) == st[1])


def witness_keys(s: A6System, ev: dict, rng: random.Random, n: int = 200) -> list[bytes]:
    """Touched keys of the step; a round step: keys of its partition, keys of the partition changed after W (tail),
    keys on the first and the last page of the partition."""
    k = ev.get("_keys")
    out: list[bytes] = []
    if ev["kind"] == "round_step":
        out = list(ev.get("_wit") or [])
        j = ev["part"]
        tail = [kb for kb in s.lsm.tail.committed if s.lsm.part(kb) == j]
        out += rng.sample(tail, min(25, len(tail)))
        for r in s.lsm.runs:  # post-W runs (sealed after W): their keys of partition j
            if r.pre or not r.n:
                continue
            ks = r.arrays(s.lsm.hz)[0]
            sel = np.flatnonzero(s.lsm.part_np(ks) == j)
            out += [bytes(ks[i]) for i in sel[:25]]
        out = list(dict.fromkeys(out))[:n]
    elif k is not None and len(k):
        idx = rng.sample(range(len(k)), min(n, len(k)))
        out = [bytes(k[i]) for i in idx]
    if ev["kind"] == "round_start" and "seal" in ev:
        out = witness_keys(s, ev["seal"], rng, n)
    return out


def check_witnesses(s: A6System, model: Model, keys: list[bytes]) -> int:
    return sum(1 for kb in keys if (kb in model.state or kb in model.deleted) and not state_ok(s, model, kb))


def drop_private(ev: dict) -> None:
    """R1: no key arrays survive the boundary in kept events (seal / merge keys, round-step witnesses)."""
    for k in [k for k in ev if k.startswith("_")]:
        del ev[k]
    if isinstance(ev.get("seal"), dict):
        drop_private(ev["seal"])


# ------------------------------------------------------------------ scenarios and mutants (§4)
def collision_check(mutant: str | None) -> dict:
    """A5 R4: artificial collisions of h (top 6 bits only); versions must be grouped by the full key."""
    import a4_lsm as _L4
    d = DATA / f"a6_coll_{mutant or 'ok'}"
    rng = np.random.default_rng(7)
    keys = rng.integers(0, 256, (2000, 16), dtype=np.uint8)
    pages = rng.integers(0, 500, 2000).astype(np.uint32)
    lsm = L.AddressLSM6.create(d, keys, pages, 2000, t_run=50, f_major=1e9, sigma=256, s_base_bits=2, mutant=mutant,
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
            lsm.tail.append(_L4.U_, kb, pg, lsm.opseq)
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
    return {"scenario": "A5 R4: collisions of h (6-bit h), merge by full key", "ok": bad_merged == 0 and bad_cand == 0,
            "changed": len(changed), "bad_merged": bad_merged, "bad_candidates": bad_cand}


def _scenario_body(mutant: str | None, res: list[dict]) -> None:
    conn = G.connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS pageinspect")
    d = DATA / f"a6_scn_{mutant or 'ok'}"
    kw = dict(t_run=50, f_major=1e9, sigma=256, s_base_bits=3, mutant=mutant)
    S = 8
    s = setup_a6(conn, "a6_scn", 10_000, DATA / "a6_scn", d, **kw)
    conn.execute("ALTER TABLE a6_scn ALTER COLUMN payload SET STORAGE PLAIN")  # no TOAST: wide rows really move
    keys = [bytes(k) for k in np.load(DATA / "a6_scn.keys")]
    model = Model(keys, [bytes(x) for x in np.load(DATA / "a6_scn.dig")])
    extra = load_extra(DATA / "dyn.copy")[-8000:]
    # ---- the partition function (§2.3): all rows + the fresh keys used below
    fc = function_check(conn, "a6_scn", s.lsm, np.frombuffer(b"".join(k for k, _ in extra), np.uint8).reshape(-1, 16), None)
    res.append({"scenario": "partition function: all rows and fresh keys", "ok": fc["ok"]} | fc)
    pay = iter(extra)
    mrng = random.Random(5)
    order = list(keys)
    random.Random(3).shuffle(order)
    fresh = iter(order)

    def untouched(n, part=None):
        out = []
        for k in fresh if part is None else [x for x in order if s.lsm.part(x) == part]:
            if model.state.get(k, [0, 0, ""])[2] == "base" and k not in used:
                out.append(k)
                used.add(k)
                if len(out) == n:
                    break
        return out
    used: set[bytes] = set()

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

    def wide(ks):
        return run("wide UPDATEs", [("U", k, wide_payload(mrng)) for k in ks], ["upd"] * len(ks), [], record=False)

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

    def fresh_in(part):
        for k, pl in pay:
            if PH.part_of(k, S) == part and k not in model.state:
                return k, pl
        raise RuntimeError("no fresh key")

    def keys_of_part(j):
        return [k for k in model.state if s.lsm.part(k) == j]

    def move_all(ks):
        """R4: each witness must really move (new page not among its old candidates, so the old base page no longer
        holds the visible row); keys that do not move are dropped from the witness set."""
        out = []
        for k in ks:
            moved, _ = moving_update(s, model, k, mrng, rollback=False)
            if moved:
                out.append(k)
        return out

    def require(label, cond, **info):
        res.append({"scenario": f"precondition: {label}", "ok": bool(cond)} | info)

    # ---- negative (R2): a direct INSERT into a wrong partition and an UPDATE of k leaving the leaf are rejected
    k0, pl0 = next(pay)
    wrong = (PH.part_of(k0, S) + 1) % S
    rej = []
    for sql, args in ((f"INSERT INTO a6_scn_p{wrong} (seq, k, v, payload) VALUES (0, %s, 1, %s)", (uuid.UUID(bytes=k0), pl0)),
                      (f"UPDATE a6_scn_p{PH.part_of(keys[0], S)} SET k = %s WHERE k = %s",
                       (uuid.UUID(bytes=next(x for x, _ in extra if PH.part_of(x, S) != PH.part_of(keys[0], S))),
                        uuid.UUID(bytes=keys[0])))):
        try:
            with conn.transaction():
                conn.execute(sql, args)
                raise psycopg.Rollback()
            rej.append(False)
        except psycopg.errors.CheckViolation:
            rej.append(True)
    res.append({"scenario": "R2: wrong-partition INSERT and UPDATE of k leaving the leaf are rejected", "ok": all(rej),
                "rejected": rej})
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
        conn.execute("VACUUM a6_scn")
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
    check_all("witness after a tier merge (3 versions on different pages)", [w], distinct_pages=len(set(pages_w)) == 4)
    require("(b) witness had 4 distinct pages", len(set(pages_w)) == 4)
    # ---- lookup during a multi-section tier merge (mutant (z))
    empty_tier(0)
    groups = []
    for _ in range(4):
        g = untouched(70)
        wide(g)
        groups += g
        s.lsm.seal()
    started = s.lsm.maybe_start_merge()
    sections = 1 << s.lsm.job.bits if started else 0
    s.lsm.merge_step()
    check_all("lookup during a tier merge (ready + not-ready sections)", groups, sections=sections)
    res[-1]["ok"] &= sections >= 2
    job_files = sorted(x.name for x in d.glob(f"r{s.lsm.job.id:06d}*")) if s.lsm.job else []
    rdr = L.AddressLSM6.open(d, **kw)
    rdr_ok = all(rdr.candidates(k, False) for k in groups[:50])
    rdr.close()
    still = sorted(x.name for x in d.glob(f"r{s.lsm.job.id:06d}*")) if s.lsm.job else []
    res.append({"scenario": "R2: reader opened and closed during an active merge deletes nothing", "ok":
                bool(job_files) and job_files == still and rdr_ok, "job_files": job_files})
    finish_merges()
    check_all("after the multi-section merge", groups)
    # ---- A5 R2: merge -> moved witness -> seal X -> finish merge -> next merges -> lookup; mutant (k')
    guard = 0
    while any(r.tier in (0, 1) for r in s.lsm.runs):
        filler_seal()
        finish_merges()
        guard += 1
        if guard > 300:
            raise RuntimeError("tiers do not empty")
    w2 = untouched(1)[0]
    era = []
    for i in range(4):
        wide(untouched(70))
        if i == 0:
            moving_update(s, model, w2, mrng, rollback=False)
            era.append(s.lsm.tail.committed.get(w2))
        s.lsm.seal()
    s.lsm.maybe_start_merge()
    s.lsm.merge_step()
    moving_update(s, model, w2, mrng, rollback=False)
    era.append(s.lsm.tail.committed.get(w2))
    s.lsm.seal()
    finish_merges()
    check_all("R2: after the old merge finished", [w2])
    for _ in range(3):
        filler_seal()
    finish_merges()
    check_all("R2: after the next merge", [w2])
    n_t2 = sum(1 for r in s.lsm.runs if r.tier == 2)
    while sum(1 for r in s.lsm.runs if r.tier == 2) == n_t2:
        filler_seal()
        finish_merges()
    check_all("R2: after the tier-2 merge", [w2], distinct=len(set(era)) == 2)
    require("(k') witness on 2 distinct pages", len(set(era)) == 2)
    # ---- A5 R4: artificial collisions of h (no SQL)
    res.append(collision_check(mutant))
    # ---- same page number in different partitions (R8, mutant (p)): two keys found through a run only
    empty_tier(0)
    same = None
    for j in range(S):
        for jj in range(j + 1, S):
            a_ = [k for k in keys_of_part(j) if model.state[k][2] == "base" and k not in used][:400]
            b_ = [k for k in keys_of_part(jj) if model.state[k][2] == "base" and k not in used][:400]
            pa = {s.lsm.candidates(k, False)[0]: k for k in a_}
            for k in b_:
                pg = s.lsm.candidates(k, False)[0]
                if pg in pa:
                    same = (pa[pg], k, pg)
                    break
            if same:
                break
        if same:
            break
    pw = None
    for cand in untouched(30):
        if not move_all([cand]):
            continue
        j = s.lsm.part(cand)
        pg = s.lsm.tail.committed.get(cand)
        other = conn.execute(f"SELECT k FROM a6_scn_p{(j + 1) % S} WHERE (ctid::text::point)[0]::int4 = %s LIMIT 1",
                             (pg,)).fetchone()
        if other is not None:
            pw = (cand, uuid.UUID(str(other[0])).bytes, pg)
            break
    require("(p) moved key whose current page number also holds a row of partition g+1", pw is not None)
    if pw:
        s.lsm.seal()  # the moved key's only current record is now in a run
        b = s.lsm.base[s.lsm.part(pw[0])][2]
        require("(p) the base does not point to the current page", b is None or b.page_of(pw[0]) != pw[2])
        check_all("same page number in two partitions, found through a run", [pw[0], pw[1]], page=pw[2],
                  parts=[s.lsm.part(pw[0]), s.lsm.part(pw[1])])
    # ---- the round by partitions (§2.5, §4)
    empty_tier(0)
    s.lsm.t_run = 10**9  # seals in this scenario are explicit: the run composition is part of the mutant witnesses
    p1 = move_all([k for j in range(S) for k in untouched(6, part=j)])  # pre-W moved + sealed, EVERY partition
    require("(e)/(f2) moved pre-W witnesses in every partition", {s.lsm.part(k) for k in p1} == set(range(S)),
            parts=sorted({s.lsm.part(k) for k in p1}))
    s.lsm.seal()
    p2 = untouched(20)
    wide(p2)  # pre-W, left in the tail at W
    empty_part = S - 2
    victims = [k for k in keys_of_part(empty_part)]
    for i in range(0, len(victims), 500):
        ch = victims[i:i + 500]
        run("empty a partition", [("D", k, None) for k in ch], ["del"] * len(ch), [], record=False)
    first_last = []
    for j in range(S):
        pages = conn.execute(f"SELECT min((ctid::text::point)[0]::int4), max((ctid::text::point)[0]::int4) FROM a6_scn_p{j}").fetchone()
        if pages[0] is None:
            continue
        first_last += [uuid.UUID(str(r[0])).bytes for r in conn.execute(
            f"SELECT k FROM a6_scn_p{j} WHERE (ctid::text::point)[0]::int4 IN (%s, %s)", pages).fetchall()]
    first_last = [k for k in first_last if model.state.get(k, [0, 0, ""])[2] == "base"][:300]
    s.lsm.round_pending = True
    ev0 = s.maintain()
    res.append({"scenario": "round started", "ok": s.lsm.round is not None and any(e["kind"] == "round_start" for e in ev0)})
    in_rebuilt, in_pending, ins_rebuilt, del_rebuilt, del_pending, post_sealed, post_tail = [], [], [], [], [], [], []
    step = 0
    mid_checked = False
    while s.lsm.round is not None:
        nxt = s.lsm.round["s"]
        done = list(range(nxt))
        if done:
            j = random.Random(step).choice(done)
            ks = untouched(3, part=j)
            wide(ks)
            in_rebuilt += ks  # moved inside an already rebuilt partition
            k_, pl_ = fresh_in(j)
            run("INSERT into a rebuilt partition", [("I", k_, pl_)], ["ins"], [], record=False)
            moving_update(s, model, k_, mrng, rollback=False)
            ins_rebuilt.append(k_)
            dl = untouched(1, part=j)
            run("DELETE in a rebuilt partition", [("D", k, None) for k in dl], ["del"] * len(dl), [], record=False)
            del_rebuilt += dl
        if nxt < S:
            ks = untouched(3, part=nxt)
            wide(ks)
            in_pending += ks  # moved inside the partition read by the next step
            dl = untouched(1, part=nxt)
            run("DELETE in a not yet rebuilt partition", [("D", k, None) for k in dl], ["del"] * len(dl), [], record=False)
            del_pending += dl
        if step == 3:
            conn.execute("VACUUM a6_scn")
            ins = [fresh_in(j) for j in range(S) if j != empty_part]
            run("INSERT after VACUUM during the round", [("I", k, pl) for k, pl in ins], ["ins"] * len(ins), [], record=False)
            ins_rebuilt += [k for k, _ in ins]
        if step in (1, 2):
            g = untouched(20)
            wide(g)
            post_sealed += g
            s.lsm.seal()
        if step == 4:
            g = untouched(10)
            wide(g)
            post_tail += g
        evs = s.maintain()
        st = [e for e in evs if e["kind"] == "round_step"]
        if st and st[0]["part"] == 0:
            # mutant (e): right after the first publication, pre-W moved keys of NOT yet rebuilt partitions
            check_all("after the first publication: pre-W moved keys of other partitions",
                      [k for k in p1 if s.lsm.part(k) != 0])
            # mutant (m): every live key of the two affected partitions
            check_all("after the first publication: every live key of partitions 0 and 1",
                      keys_of_part(0) + keys_of_part(1))
        if st and st[0]["part"] == empty_part:
            res.append({"scenario": "empty partition: empty section", "ok": st[0]["rows"] == 0, "rows": st[0]["rows"]})
            k_, pl_ = fresh_in(empty_part)
            run("INSERT into the emptied partition after its step", [("I", k_, pl_)], ["ins"], [k_])
            ins_rebuilt.append(k_)
        if st and st[0]["part"] == S // 2 and not mid_checked:
            mid_checked = True
            check_all("mid-round: pre-W changes (sections partly rebuilt)", p1 + p2)
        step += 1
    base_sample = untouched(1500)
    check_all("after the round: pre-W moved, sealed, all partitions (incl. the last)", p1)
    check_all("after the round: pre-W in the tail at W", p2)
    check_all("after the round: moved inside rebuilt partitions", in_rebuilt)
    check_all("after the round: moved inside partitions before their step", in_pending)
    check_all("after the round: inserted into rebuilt partitions (incl. after VACUUM)", ins_rebuilt)
    check_all("after the round: deleted during the round", del_rebuilt + del_pending)
    check_all("after the round: post-W sealed", post_sealed)
    check_all("after the round: post-W in the tail", post_tail)
    check_all("after the round: keys on the first and last page of partitions", first_last)
    check_all("after the round: untouched base keys", base_sample)
    check_all("after the round: every live key of the emptied partition", keys_of_part(empty_part))
    s5 = s.lsm.s5_violations
    res.append({"scenario": "S5: no stray files, step files accounted", "ok": not s5 and not s.lsm.stray_files(),
                "violations": s5[:3], "stray": s.lsm.stray_files()})
    s.lsm.t_run = 50
    # ---- restart in the middle of a round with an active tier merge: the round RESUMES (R1, mutants (o), (r))
    empty_tier(0)
    s.lsm.t_run = 10**9
    q1 = move_all([k for j in range(S) for k in untouched(5, part=j)])  # pre-W moved, sealed, every partition (r)
    require("(r) moved pre-W witnesses in every partition", {s.lsm.part(k) for k in q1} >= set(range(S)) - {empty_part},
            parts=sorted({s.lsm.part(k) for k in q1}), emptied=empty_part)
    s.lsm.seal()
    s.lsm.round_pending = True
    s.maintain()
    for _ in range(S // 2):
        s.maintain()  # partitions 0 .. S/2-1 rebuilt
    o_keys = move_all([k for j in range(S // 2) for k in untouched(4, part=j)])  # moved after their partition was read
    require("(o) moved witnesses in rebuilt partitions", len(o_keys) > 0, n=len(o_keys))
    s.lsm.seal()  # sealed between the old W and the restart (mutant (o))
    r3 = []
    for _ in range(3):
        g = untouched(100)
        wide(g)
        r3 += g
        s.lsm.seal()
    started = s.lsm.maybe_start_merge()
    if started:
        s.lsm.merge_step()
    state = {"job": s.lsm.job is not None, "round_next": s.lsm.round["s"] if s.lsm.round else None}
    s.lsm.close()
    s.attach(L.AddressLSM6.open(d, **kw, writer=True), recount=True)
    check_all("restart mid-round + mid-merge: right after reopening", q1 + o_keys + r3 + base_sample[:300],
              state_before=state, resumed=s.lsm.resumed, abandoned=sorted(s.lsm.abandoned))
    res[-1]["ok"] &= s.lsm.round is not None and state["job"]
    while s.lsm.round is not None or s.lsm.job is not None:
        s.maintain()
    check_all("restart: after the resumed round finished", q1 + o_keys + r3 + base_sample)
    check_all("restart: every live key after the resumed round", list(model.state))
    s.lsm.t_run = 50
    # ---- stop between the last publication and round_end (Codex 241 R1): the reopened round only finishes
    empty_tier(0)
    s.lsm.t_run = 10**9
    u1 = move_all([k for j in range(S) for k in untouched(3, part=j)])
    s.lsm.seal()
    s.lsm.round_pending = True
    s.maintain()
    while s.lsm.round is not None and s.lsm.round["s"] < S - 1:
        s.maintain()
    s.lsm.crash_before_end = True
    stopped = False
    try:
        s.maintain()
    except RuntimeError:
        stopped = True
    s.lsm.crash_before_end = False
    s.lsm.close()
    s.attach(L.AddressLSM6.open(d, **kw, writer=True), recount=True)
    state = {"stopped": stopped, "round_next": s.lsm.round["s"] if s.lsm.round else None}
    evs = s.maintain()
    kinds = [e["kind"] for e in evs]
    check_all("stop after the last publication: the reopened round only finishes", u1 + base_sample[:300],
              state=state, kinds=kinds)
    res[-1]["ok"] &= stopped and state["round_next"] == S and "round_end_only" in kinds and s.lsm.round is None
    res.append({"scenario": "n_live recount after a writer reopen", "ok": s.lsm.n_live == len(model.state),
                "n_live": s.lsm.n_live, "model": len(model.state)})
    s.lsm.t_run = 50
    # ---- reader in another instance: refresh vs mutant (c)
    reader = L.AddressLSM6.open(d, **kw)
    rs = A6System("reader", "a6_scn", "a6_scn_seq", d, reader)
    rs.reader_mode = True
    rs.reader_refresh = mutant != "c"
    nk = next(pay)
    run("writer INSERT after the reader opened", [("I", nk[0], nk[1])], ["ins"], [nk[0]])
    res.append({"scenario": "reader sees a key committed after it opened", "ok": rs.lookup(nk[0]) is not None})
    reader.close()
    rs.conn.close()
    # ---- refuse to open: a batch without an outcome marker, a torn record (mutant (d))
    s.lsm.batch_no += 1
    s.lsm.tail.append(L4.I_, bytes(16), 1, s.lsm.opseq + 1)
    s.lsm.tail.f.write(b"".join(s.lsm.tail._buf))
    s.lsm.tail.f.flush()
    s.lsm.tail._buf = []
    refused = False
    try:
        L.AddressLSM6.open(d, ignore_open_batch=(mutant == "d"), **kw).close()
    except L4.RefuseOpen:
        refused = True
    res.append({"scenario": "refuse to open with a batch without outcome", "ok": refused})
    with open(s.lsm.tail.path, "ab") as f:
        f.write(b"\x01\x02\x03")
    torn = False
    try:
        L.AddressLSM6.open(d, **kw).close()
    except L4.RefuseOpen:
        torn = True
    res.append({"scenario": "refuse to open with a torn record", "ok": torn})
    s.lsm.close()
    s.conn.close()
    conn.execute("DROP TABLE IF EXISTS a6_scn CASCADE")
    conn.close()


# the scenario that must fail for each mutant (R5): its assigned witness of §4
WITNESS = {
    "a": "R1: UPDATE moved to another page, then ROLLBACK",
    "b": "witness after a tier merge (3 versions on different pages)",
    "c": "reader sees a key committed after it opened",
    "d": "refuse to open with a batch without outcome",
    "e": "after the first publication: pre-W moved keys of other partitions",
    "f2": "after the round: pre-W moved, sealed, all partitions (incl. the last)",
    "g": "after the round: post-W sealed",
    "z": "lookup during a tier merge (ready + not-ready sections)",
    "i": "after the round: post-W in the tail",
    "k2": "R2: after the tier-2 merge",
    "l": "A5 R4: collisions of h (6-bit h), merge by full key",
    "m": "after the first publication: every live key of partitions 0 and 1",
    "n": "partition function: all rows and fresh keys",
    "o": "restart: after the resumed round finished",
    "p": "same page number in two partitions, found through a run",
    "r": "restart: after the resumed round finished",
}


def scenario_suite(mutant: str | None = None) -> dict:
    res: list[dict] = []
    exc = None
    try:
        _scenario_body(mutant, res)
    except Exception as e:  # noqa: BLE001 - recorded, never counted as a catch by itself (R5)
        exc = {"type": type(e).__name__, "msg": str(e)[:300], "after": res[-1]["scenario"] if res else None}
        try:
            c = G.connect()
            c.execute("DROP TABLE IF EXISTS a6_scn CASCADE")
            c.close()
        except Exception:  # noqa: BLE001
            pass
    failed = [r["scenario"] for r in res if not r["ok"]]
    return {"mutant": mutant, "all_ok": exc is None and not failed, "results": res, "failed": failed,
            "exception": exc}


IMPL_FILES = ("a6_lsm.py", "a6_run.py", "a6_pghash.py", "a6_selftest_lsm.py", "a5_lsm.py", "a5_run.py", "a4_lsm.py",
              "a4_run.py", "a2_core.py", "a2_pg.py", "a2_dyn.py", "a2_runs.py", "a2_gen.py", "a2_selftest_core.py", "a2.py",
              "corpus.py", "hoststats.py")  # the last two: portability commit (generator, host counters)
# preconditions a mutant's witness depends on (R4): an unmet precondition makes that mutant's result НЕ УСТАНОВЛЕНО
PRECOND = {
    "b": ["precondition: (b) witness had 4 distinct pages"],
    "k2": ["precondition: (k') witness on 2 distinct pages"],
    "e": ["precondition: (e)/(f2) moved pre-W witnesses in every partition"],
    "f2": ["precondition: (e)/(f2) moved pre-W witnesses in every partition"],
    "p": ["precondition: (p) moved key whose current page number also holds a row of partition g+1",
          "precondition: (p) the base does not point to the current page"],
    "r": ["precondition: (r) moved pre-W witnesses in every partition"],  # except the emptied one (no base keys left)
    "o": ["precondition: (o) moved witnesses in rebuilt partitions"],
}
# offline (no SQL) mutants caught by an exception: registered type (R5), seen in the first offline run before the
# SQL selfcheck: (g) merges a pre-W run that the round end then deletes -> the next merge step cannot read it
OFFLINE_EXC = {"g": "FileNotFoundError"}


def impl_hashes() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in IMPL_FILES}


def cmd_selfcheck() -> None:
    import a6_selftest_lsm as T
    out = {"impl_sha256": impl_hashes()}
    r = subprocess.run([str(PY), str(ROOT / "a2_selftest_core.py")], capture_output=True, text=True, cwd=str(ROOT))
    out["core_selftest"] = {"exit": r.returncode, "tail": r.stdout.strip().splitlines()[-1]}
    out["pghash_selftest"] = PH.selftest()
    out["offline_ok"] = T.run(None)
    out["offline_mutants"] = {}
    for m in OFFLINE_MUTANTS:
        try:
            rr = T.run(m)
            out["offline_mutants"][m] = {"caught": not rr["ok"] and (rr["bad_sampled"] > 0 or rr["bad_final"] > 0),
                                         "bad_sampled": rr["bad_sampled"], "bad_final": rr["bad_final"]}
        except Exception as e:  # noqa: BLE001
            out["offline_mutants"][m] = {"caught": OFFLINE_EXC.get(m) == type(e).__name__,
                                         "exception": f"{type(e).__name__}: {str(e)[:200]}"}
    log("offline", out["offline_ok"]["ok"], {m: v["caught"] for m, v in out["offline_mutants"].items()})
    out["ok"] = scenario_suite(None)
    log("scenarios ok-run:", out["ok"]["all_ok"], out["ok"]["failed"], out["ok"]["exception"])
    for m in MUTANTS:
        res = scenario_suite(m)
        present_ok = {r["scenario"] for r in res["results"] if r["ok"]}
        pre_bad = [x for x in PRECOND.get(m, []) if x not in present_ok]
        rec = {"witness": WITNESS.get(m), "witness_failed": WITNESS.get(m) in res["failed"],
               "preconditions_unmet": pre_bad, "failed": res["failed"], "exception": res["exception"]}
        rec["caught"] = rec["witness_failed"] and not pre_bad
        rec["outcome"] = ("caught by its witness" if rec["caught"] else
                          "НЕ УСТАНОВЛЕНО (precondition)" if pre_bad else "not caught")
        out[f"mutant_{m}"] = rec
        log("mutant", m, rec["outcome"], rec["exception"])
    ph = out["pghash_selftest"]
    out["mutant_k_note"] = "equivalent under one-merge-at-a-time (A5 addendum §1); reported, not required"
    out["all_ok"] = (out["core_selftest"]["exit"] == 0 and out["offline_ok"]["ok"] and out["ok"]["all_ok"]
                     and all(v["caught"] for v in out["offline_mutants"].values())
                     and ph["bad_uuid_hash_extended"] == 0 and ph["bad_numpy_vs_scalar"] == 0
                     and ph["bad_routing_S8"] == 0 and ph["bad_routing_S128"] == 0
                     and all(out[f"mutant_{m}"]["caught"] for m in MUST_CATCH))
    G.write_json(RES / "a6_selfcheck.json", out)
    log("selfcheck all_ok", out["all_ok"])
    if not out["all_ok"]:
        raise SystemExit(1)


def require_selfcheck() -> None:
    """R6: dynamics only after a passed selfcheck of exactly this implementation."""
    f = RES / "a6_selfcheck.json"
    if not f.exists():
        raise SystemExit("no a6_selfcheck.json")
    sc = json.loads(f.read_text(encoding="utf-8"))
    if not sc.get("all_ok") or sc.get("impl_sha256") != impl_hashes():
        raise SystemExit("selfcheck failed or implementation changed since the selfcheck")


class Stop(Exception):
    """R6: a correctness error stops the run (protocol §10)."""


# ------------------------------------------------------------------ dynamics (§3, §5)
def open_probe(d: Path, keys: list[bytes], table: str, runs: int = 21) -> dict:
    res = []
    for i in range(runs):
        r = subprocess.run([str(PY), str(ROOT / "a6_run.py"), "open-probe", str(d), table, keys[i].hex()],
                           capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        res.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return {k: G.pct([x[k] for x in res]) for k in res[0] if isinstance(res[0][k], (int, float))}


def cmd_open_probe(d: str, table: str, keyhex: str) -> None:
    t0 = time.perf_counter()
    conn = G.connect()
    t1 = time.perf_counter()
    lsm = L.AddressLSM6.open(Path(d))
    t2 = time.perf_counter()
    s = A6System("probe", table, f"{table}_seq", Path(d), lsm, conn=conn)
    row = s.lookup(bytes.fromhex(keyhex))
    t3 = time.perf_counter()
    print(json.dumps({"connect_ms": (t1 - t0) * 1e3, "open_ms": (t2 - t1) * 1e3, "first_answer_ms": (t3 - t2) * 1e3,
                      "found": row is not None, "levels": lsm.levels(), "files": len(list(Path(d).glob("*.adr"))),
                      "tail_records": lsm.tail.offset // L4.LOGREC.size}))


def round_summaries(events: list[dict], vac_ops: tuple[int, ...]) -> list[dict]:
    rounds, cur = [], None
    for e in events:
        if e["kind"] == "round_start":
            cur = {"start_op": e["after_op"], "start_s": e["s"], "read_s": 0.0, "build_s": 0.0, "read_bytes": 0,
                   "steps": 0, "max_step_s": 0.0, "max_read_s": 0.0, "max_build_s": 0.0, "max_rows": 0,
                   "forced": e.get("forced_round", False), "final": e.get("final", False),
                   "delta_bytes_at_start": e.get("published_bytes")}
        elif cur is not None and e["kind"] == "round_end_only":
            cur.update({k: v for k, v in e["round_end"].items() if k != "s"})
            cur["end_s"] = e["round_end"]["s"]
            cur["end_op"] = e["after_op"]
            cur["B_s"] = cur.get("steps_s", 0.0)
            cur["finished_after_restart"] = True
            rounds.append(cur)
            cur = None
        elif cur is not None and e["kind"] == "round_step":
            end_s = e["round_end"]["s"] if "round_end" in e else 0.0
            cur["read_s"] += e["s_read"]
            cur["build_s"] += e["s_build"]
            cur["steps_s"] = cur.get("steps_s", 0.0) + e["s"] - end_s
            cur["read_bytes"] += e.get("read_bytes", 0)
            cur["steps"] += 1
            cur["max_step_s"] = max(cur["max_step_s"], e["s"] - end_s)
            cur["max_read_s"] = max(cur["max_read_s"], e["s_read"])
            cur["max_build_s"] = max(cur["max_build_s"], e["s_build"])
            cur["max_rows"] = max(cur["max_rows"], e["rows"])
            if "round_end" in e:
                cur.update({k: v for k, v in e["round_end"].items() if k != "s"})
                cur["end_s"] = end_s
                cur["end_op"] = e["after_op"]
                cur["B_s"] = cur["steps_s"]
                cur["vacuum_in_window"] = [v for v in vac_ops if cur["start_op"] <= v <= cur["end_op"]]
                rounds.append(cur)
                cur = None
    return rounds


def fail(out: dict, tag: str, why: str) -> None:
    out["STOPPED"] = why
    G.write_json(RES / f"a6_dynamics_{tag}_STOPPED.json", out)
    log("STOP:", why)
    raise SystemExit(2)


def stop_on_cp(out: dict, tag: str) -> None:
    rec = out["checkpoints"][-1]
    bad = {k: v["errors"] for k, v in rec["systems"].items() if sum(v["errors"].values())}
    if bad:
        fail(out, tag, f"C4 errors at checkpoint {rec['cp']} {rec['name']}: {bad}")


def cmd_dynamics(size: str) -> None:
    import os
    smoke = int(os.environ.get("A6_SMOKE_OPS", "0"))  # a code smoke test only (not a measurement): fewer operations
    if not smoke:
        require_selfcheck()
    ops_run = smoke or OPS
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower() + ("_smoke" if smoke else "")
    conn = G.connect()
    ref = DATA / f"ref_a6_{size.lower()}"
    log(size, "load A6 table")
    t = time.perf_counter()
    a = setup_a6(conn, f"a6_{tag}", n, ref, DATA / f"a6_{tag}", t_run=T_RUN, f_major=F_MAJOR)
    load_s = time.perf_counter() - t
    S = a.lsm.n_parts
    keys = [bytes(k) for k in np.load(f"{ref}.keys")]
    digs = [bytes(x) for x in np.load(f"{ref}.dig")]
    extra = load_extra(DATA / "dyn.copy")
    ops = make_ops(keys, extra, 250_000, 150_000, 100_000, 13)
    ins_keys = np.frombuffer(b"".join(kb for kind, kb, _ in ops if kind == "I"), np.uint8).reshape(-1, 16)
    log(size, "partition function check")
    fc = function_check(conn, a.table, a.lsm, ins_keys, None)
    log(size, "function check", {k: fc[k] for k in ("rows", "bad_numpy", "bad_scalar", "bad_insert_keys", "ok")})
    if not fc["ok"]:
        raise SystemExit("partition function check failed: НЕ УСТАНОВЛЕНО")
    log(size, "fill control table on the server")
    ctl = setup_control(conn, f"c6_{tag}", a.table, S, n)
    log(size, "control layout share", round(ctl["layout_share"], 6))
    c = Control6(f"c6_{tag}", f"c6_{tag}", f"c6_{tag}_seq")
    c.nparts = S
    ever = np.concatenate([np.load(f"{ref}.keys"), np.load(DATA / "dyn.keys")])
    out = {"size": size, "rows": n, "ops_sha256": ops_sha256(ops), "partitions": S, "load_s": load_s,
           "function_check": fc, "control_setup": {k: v for k, v in ctl.items() if not k.startswith("pages")},
           "partitions_start": {"a6": ctl["pages_a"], "c6": ctl["pages_c"]}, "btree_heights": btree_heights(conn, c.table),
           "checkpoints": [], "batches": [], "events": [], "witness_errors": 0, "s3_batch_violations": [],
           "stray_file_boundaries": [], "vacuum": {}, "client_stops": {"vacuum_s": [], "pg_restart_s": None,
                                                                      "client_restart_mid_round_s": None}}
    out["stream_matches_a4"] = out["ops_sha256"].startswith(STREAM_SHA[size])
    log(size, "stream sha256", out["ops_sha256"], "matches A4:", out["stream_matches_a4"], "partitions", S,
        "btree heights", sorted(set(out["btree_heights"])))
    model = Model(keys, digs)
    del keys, digs
    systems = (a, c)
    cp = 0
    out["checkpoints"].append(checkpoint6(cp, "base", 0, systems, model, ever))
    stop_on_cp(out, tag)
    t_ops = {a.name: 0.0, c.name: 0.0}
    t_maint = 0.0
    t_vac = {a.name: 0.0, c.name: 0.0}
    wrng = random.Random(77)
    half = S // 2
    last_vac_batch = 0
    restarted_mid = False
    restart_mid = {}
    for start in range(0, ops_run, BATCH):
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
        batch_no = op_no // BATCH
        if a.errors or c.errors:
            fail(out, tag, f"apply errors at op {op_no}: A6 {a.errors}, control {c.errors}")
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
                log(size, "round start at op", op_no)
            if ev["kind"] == "round_step":
                ev["batches_since_vacuum"] = batch_no - last_vac_batch
            wk = witness_keys(a, ev, wrng)
            ev["witness_errors"] = check_witnesses(a, model, wk)
            ev["witnesses"] = len(wk)
            out["witness_errors"] += ev["witness_errors"]
            drop_private(ev)
            if ev["witness_errors"]:
                out["events"].append(clean(ev))
                fail(out, tag, f"witness errors at op {op_no}: {ev['kind']}")
            if ev["kind"] == "round_step" and ev["part"] == half - 1:
                mid = True
            if ev["kind"] == "round_step" and "round_end" in ev:
                ev["round_end"]["live_rows"] = len(model.state)
                log(size, "round end at op", op_no, {k: ev["round_end"][k] for k in ("rows", "max_rows", "steps", "window_ops")})
            out["events"].append(clean(ev))
        nb = a.lsm.nbytes()
        per = nb / len(model.state)
        stray = a.lsm.stray_files()
        if stray:
            out["stray_file_boundaries"].append([op_no, stray[:5]])
        out["batches"].append([op_no, ta, tmaint, nb, per, a.lsm.levels(), "+".join(e["kind"] for e in evs)])
        if per > 8.0:
            out["s3_batch_violations"].append([op_no, per])
        if op_no in VACUUMS:
            for s_ in systems:
                tv = time.perf_counter()
                conn.execute(f"VACUUM {s_.table}")
                dv = time.perf_counter() - tv
                t_vac[s_.name] += dv
                out["client_stops"]["vacuum_s"].append([op_no, s_.name, dv])
            last_vac_batch = batch_no
        if mid:
            cp += 1
            out["checkpoints"].append(checkpoint6(cp, "mid_round", op_no, systems, model, ever))
            stop_on_cp(out, tag)
            # N1: one client restart in the middle of the FIRST working round; the round resumes (§3)
            if size == "N1" and not restarted_mid:
                restarted_mid = True
                tr = time.perf_counter()
                before = {"round_next": a.lsm.round["s"] if a.lsm.round else None, "job": a.lsm.job is not None}
                a.lsm.close()
                a.reopen()
                a.attach(L.AddressLSM6.open(a.dir, t_run=T_RUN, f_major=F_MAJOR, writer=True), recount=True)
                a._q = {}
                restart_mid = before | {"resumed": a.lsm.resumed, "abandoned": a.lsm.abandoned,
                                        "s": time.perf_counter() - tr, "op": op_no}
                out["client_stops"]["client_restart_mid_round_s"] = restart_mid["s"]
                log(size, "client restart mid-round", restart_mid)
                cp += 1
                out["checkpoints"].append(checkpoint6(cp, "after_resume", op_no, systems, model, ever))
                stop_on_cp(out, tag)
        if op_no % CP_EVERY == 0:
            cp += 1
            rec = checkpoint6(cp, "every_50k", op_no, systems, model, ever)
            rec["candidates_hist"] = dict(a.cand_hist)
            a.cand_hist = {}
            rec["rss"] = rss_mb()
            out["checkpoints"].append(rec)
            stop_on_cp(out, tag)
            log(size, "ops", op_no, {k: round(v, 1) for k, v in t_ops.items()}, "maint", round(t_maint, 1),
                "levels", a.lsm.levels(), "max pause", round(max(b[2] for b in out["batches"]), 3))
    out["restart_mid_round"] = restart_mid
    # restart, reopen from disk, checkpoint, open probe
    log(size, "restart")
    tr = time.perf_counter()
    G.pg_ctl("restart")
    conn = G.connect()
    out["client_stops"]["pg_restart_s"] = time.perf_counter() - tr
    state_before = {"job": a.lsm.job is not None, "round": a.lsm.round is not None}
    a.lsm.close()
    t = time.perf_counter()
    a.reopen()
    a.attach(L.AddressLSM6.open(a.dir, t_run=T_RUN, f_major=F_MAJOR, writer=True), recount=True)
    out["reopen_s"] = time.perf_counter() - t
    out["restart_state"] = state_before | {"resumed": a.lsm.resumed, "abandoned": a.lsm.abandoned}
    a._q = {}
    c.reopen()
    out["client_stops"]["restart_total_s"] = time.perf_counter() - tr  # server + address open + recount + clients
    cp += 1
    out["checkpoints"].append(checkpoint6(cp, "after_restart", ops_run, systems, model, ever))
    stop_on_cp(out, tag)
    out["open_probe"] = open_probe(a.dir, model.live[:21], a.table)
    log(size, "open probe", out["open_probe"].get("open_ms"))
    # final round: steps back to back (§3); memory of each step measured with tracemalloc (§5, R4)
    published_before = a.lsm.nbytes()
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[0]
    a.lsm.trace_mem = True
    t = time.perf_counter()
    fevs = a.final_round()
    out["final_round_continuous_s"] = time.perf_counter() - t
    a.lsm.trace_mem = False
    for ev in fevs:  # witnesses of the final round's steps (checked after the round, outside its timing)
        wk = witness_keys(a, ev, wrng)
        ev["witness_errors"] = check_witnesses(a, model, wk)
        ev["witnesses"] = len(wk)
        drop_private(ev)
    tracemalloc.reset_peak()
    mem_after = tracemalloc.get_traced_memory()[0]
    out["final_round_memory"] = {"retained_after_round_bytes": mem_after - mem_before}
    tracemalloc.stop()
    for ev in fevs:
        if ev["kind"] == "round_start":
            ev["final"] = True
            ev["published_bytes"] = published_before
        if ev["kind"] == "round_step" and "round_end" in ev:
            ev["round_end"]["live_rows"] = len(model.state)
    out["final_events"] = [clean(e) for e in fevs]
    fsteps = [e for e in fevs if e["kind"] == "round_step"]
    out["step_memory"] = {"max_peak_bytes": max(e["mem_peak_bytes"] for e in fsteps),
                          "peak_by_step": [e["mem_peak_bytes"] for e in fsteps],
                          "max_pair_bytes": max(e["pair_bytes"] for e in fsteps),
                          "median_peak_bytes": float(np.median([e["mem_peak_bytes"] for e in fsteps]))}
    cp += 1
    out["checkpoints"].append(checkpoint6(cp, "after_final_round", ops_run, systems, model, ever))
    stop_on_cp(out, tag)
    final_wit = sum(e.get("witness_errors", 0) for e in out["final_events"])
    if final_wit:
        fail(out, tag, f"witness errors in the final round: {final_wit}")
    out["partitions_end"] = {"a6": part_sizes(conn, a.table), "c6": part_sizes(conn, c.table)}
    out["table_bytes"] = {"a6": parted_bytes(conn, a.table), "c6": parted_bytes(conn, c.table)}
    out["ops_time_s"] = t_ops
    out["maintenance_s"] = t_maint
    out["vacuum_s"] = t_vac
    out["errors"] = {s_.name: s_.errors for s_ in systems}
    out["full_cost_us_per_op"] = {a.name: (t_ops[a.name] + t_maint + t_vac[a.name]) / ops_run * 1e6,
                                  c.name: (t_ops[c.name] + t_vac[c.name]) / ops_run * 1e6}
    pauses = [b[2] for b in out["batches"]]
    out["boundary_pause_s"] = {"max": max(pauses), "p99": float(np.percentile(pauses, 99)),
                               "p95": float(np.percentile(pauses, 95)), "sum": sum(pauses),
                               "nonzero": sum(1 for x in pauses if x > 0.001),
                               "max_at_op": out["batches"][int(np.argmax(pauses))][0]}
    by_kind: dict[str, list[float]] = {}
    for e in out["events"]:
        by_kind.setdefault(e["kind"], []).append(e["s"])
    out["step_s_by_kind"] = {k: {"n": len(v), "max": max(v), "p99": float(np.percentile(v, 99)), "sum": sum(v)}
                             for k, v in by_kind.items()}
    steps = [e for e in out["events"] if e["kind"] == "round_step"]
    allsteps = steps + [e for e in out["final_events"] if e["kind"] == "round_step"]
    out["round_step_work"] = {"max_rows": max(e["rows"] for e in steps), "max_pair_bytes": max(e["pair_bytes"] for e in steps),
                              "max_read_s": max(e["s_read"] for e in steps), "max_build_s": max(e["s_build"] for e in steps),
                              "p50_read_s": float(np.median([e["s_read"] for e in steps]))}
    out["s5"] = {"steps_checked": len(allsteps),
                 "step_file_violations": [[e["after_op"], e["part"], e["s5_bad_files"]] for e in allsteps if e["s5_bad_files"]],
                 "stray_file_boundaries": len(out["stray_file_boundaries"])}
    bt = [b[1] + b[2] for b in out["batches"]]
    out["batch_time_s"] = {"p50": float(np.percentile(bt, 50)), "p95": float(np.percentile(bt, 95)),
                           "p99": float(np.percentile(bt, 99)), "max": max(bt)}
    cs = out["client_stops"]
    out["max_client_stop_s"] = {"address_pause": max(pauses),
                                "vacuum_max_single": max((v[2] for v in cs["vacuum_s"]), default=0.0),
                                "pg_restart": cs["pg_restart_s"], "restart_total": cs.get("restart_total_s"),
                                "client_restart_mid_round": cs["client_restart_mid_round_s"]}
    out["rounds"] = round_summaries(out["events"], VACUUMS)
    out["final_round"] = (round_summaries(out["final_events"], ()) or [None])[0]
    merges = [e for e in out["events"] if e["kind"] == "merge_step"]
    out["merge_work"] = {"steps": len(merges), "max_keys_in": max((e["keys_in"] for e in merges), default=0),
                         "max_keys_out": max((e["keys_out"] for e in merges), default=0),
                         "finished": sum(1 for e in merges if e.get("finished"))}
    out["peaks"] = {"published_bytes": a.lsm.peak_published}
    out["rss"] = rss_mb()
    a.lsm.close()
    for t_ in (a.table, c.table):
        conn.execute(f"DROP TABLE IF EXISTS {t_} CASCADE")
    G.write_json(RES / f"a6_dynamics_{tag}.json", out)
    log(size, "done errors", out["errors"], "witness", out["witness_errors"], "S3 batch violations",
        len(out["s3_batch_violations"]), "max pause", round(out["boundary_pause_s"]["max"], 3),
        "stray", len(out["stray_file_boundaries"]))


# ------------------------------------------------------------------ replay of a natural cycle without SQL (§5, R6)
def cmd_replay(size: str) -> None:
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower()
    keys_arr = np.load(DATA / f"ref_a4_{tag}.keys")
    n_pages = {"N1": 76_232, "N10": 762_888}[size]
    rng = np.random.default_rng(13)
    k_cycle = int(F_MAJOR * n)
    cap = n + k_cycle * 3
    d = DATA / f"a6_replay_{tag}"
    bits = L.bits_for(n, L.SIGMA)
    S = 1 << bits
    pp = max(1, n_pages // S)  # pages per partition
    tkeys = np.zeros((cap, 16), np.uint8)
    tkeys[:n] = keys_arr
    tpart = np.zeros(cap, np.int64)
    tpart[:n] = PH.part_of_np(keys_arr, S)
    tpages = np.zeros(cap, np.uint32)
    tpages[:n] = rng.integers(0, pp, size=n).astype(np.uint32)
    alive = np.zeros(cap, bool)
    alive[:n] = True
    live_pos = np.arange(n, dtype=np.int64)
    n_live, n_used = n, n
    lsm = L.AddressLSM6.create(d, keys_arr, tpages[:n], n, t_run=T_RUN, f_major=F_MAJOR)
    del keys_arr
    prng = random.Random(13)
    top = pp + 10_000
    due_marks = []
    pending_due = None

    def read_fn(s):
        idx = np.flatnonzero(alive[:n_used] & (tpart[:n_used] == s))
        return tkeys[idx], tpages[idx], {"read_bytes": 0}

    bounds_work, t_log = [], 0.0
    ops_done = 0
    w_marks = []
    while True:
        t = time.perf_counter()
        lsm.batch_no += 1
        for _ in range(BATCH):
            lsm.opseq += 1
            r = prng.random()
            if r < 0.5:
                kb = prng.randbytes(16)
                tkeys[n_used] = np.frombuffer(kb, np.uint8)
                tpart[n_used] = PH.part_of(kb, S)
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
                lsm.n_live += 1
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
                lsm.n_live -= 1
                lsm.tail.append(L4.D_, tkeys[i].tobytes(), 0, lsm.opseq)
        lsm.ops_total += lsm.tail.commit(lsm.batch_no)
        t_log += time.perf_counter() - t
        ops_done += BATCH
        due_at = ops_done if lsm.round_due() else None
        if due_at is not None and pending_due is None:
            pending_due = due_at
        t = time.perf_counter()
        evs = lsm.boundary(read_fn)
        wall = time.perf_counter() - t
        kinds = [e["kind"] for e in evs]
        step_s = sum(e["s"] - (e["round_end"]["s"] if "round_end" in e else 0.0) for e in evs if e["kind"] == "round_step")
        read_s = sum(e["s_read"] for e in evs if e["kind"] == "round_step")
        rs = [e for e in evs if e["kind"] == "round_start"]
        if rs:
            w_marks.append({"op": ops_done, "W": lsm.round["W"] if lsm.round else None})
            due_marks.append(pending_due)
            pending_due = None
        bounds_work.append({"op": ops_done, "work_s": wall - read_s, "other_s": wall - step_s, "kinds": kinds,
                            "due": due_at, "cycle": len(w_marks),
                            "seal_s": sum(e["s"] for e in evs if e["kind"] == "seal"),
                            "merge_s": sum(e["s"] for e in evs if e["kind"] == "merge_step"),
                            "round_start_s": sum(e["s"] for e in evs if e["kind"] == "round_start"),
                            "step_s": step_s,
                            "round_end_s": sum(e["round_end"]["s"] for e in evs if "round_end" in e),
                            "merge_read": sum(e.get("read_bytes", 0) for e in evs if e.get("finished")),
                            "merge_written": sum(e.get("written_bytes", 0) for e in evs if e.get("finished")),
                            "seal_written": sum(e["run"]["adr_bytes"] + e["run"]["log_bytes"] for e in evs
                                                if e["kind"] == "seal")
                            + sum(e["seal"]["run"]["adr_bytes"] + e["seal"]["run"]["log_bytes"] for e in evs
                                  if e["kind"] == "round_start" and "seal" in e),  # R8: the forced seal at W
                            "seals": kinds.count("seal") + sum(1 for e in evs if e["kind"] == "round_start" and "seal" in e)})
        if len(w_marks) == 2:
            break
        if n_used + BATCH >= cap:
            raise RuntimeError("replay capacity")
    # the measured cycle: from the boundary that started the round at W1 (inclusive) to the one at W2 (exclusive)
    cyc = [b for b in bounds_work if b["cycle"] == 1]
    w1, w2 = w_marks[0]["W"], w_marks[1]["W"]
    m_ops = w2 - w1
    first_s = sum(b["seal_s"] + b["merge_s"] + b["round_start_s"] + b["round_end_s"] for b in cyc)
    first_b = sum(b["seal_written"] + b["merge_read"] + b["merge_written"] for b in cyc)
    warm = [b for b in bounds_work if b["cycle"] == 0]
    multi = [b for b in cyc if len(b["kinds"]) >= 2]
    step_bounds = [b for b in cyc if "round_step" in b["kinds"]]
    out = {"size": size, "partitions": S, "cycle_ops_K": k_cycle, "ops_done": ops_done, "W1": w1, "W2": w2, "M": m_ops,
           "round_start_ops": [w["op"] for w in w_marks],
           "first_term_us_per_op": first_s / m_ops * 1e6, "first_term_bytes_per_op": first_b / m_ops,
           "seals_in_cycle": sum(b["seals"] for b in cyc),
           "merge_steps_in_cycle": sum(b["kinds"].count("merge_step") for b in cyc),
           "log_us_per_op": t_log / ops_done * 1e6,
           "max_boundary_work_s": max(b["work_s"] for b in cyc),
           "max_boundary_work_at": max(cyc, key=lambda b: b["work_s"])["op"],
           "warmup_max_boundary_work_s": max((b["work_s"] for b in warm), default=0.0),
           "w2_boundary_work_s": bounds_work[-1]["work_s"],
           "max_multi_kind_work_s": max((b["work_s"] for b in multi), default=0.0),
           "multi_kind_boundaries": len(multi),
           "max_other_work_at_step_boundaries_s": max((b["other_s"] for b in step_bounds), default=0.0),
           "max_by_kind": {k: max((b[k] for b in cyc), default=0.0) for k in
                           ("seal_s", "merge_s", "round_start_s", "step_s", "round_end_s")},
           "levels_end": lsm.levels(), "tiers_end": [r.tier for r in lsm.runs],
           "peak_published_bytes": lsm.peak_published, "rss": rss_mb(),
           "due_ops": due_marks, "start_delay_ops": [None if d_ is None else w["op"] - d_ for w, d_ in zip(w_marks, due_marks)],
           "n_live_end": lsm.n_live, "n0_end": lsm.n0}
    lsm.close()
    G.write_json(RES / f"a6_replay_{tag}.json", out)
    log("replay", size, {k: out[k] for k in ("M", "first_term_us_per_op", "first_term_bytes_per_op",
                                              "max_boundary_work_s", "max_multi_kind_work_s", "seals_in_cycle",
                                              "merge_steps_in_cycle")})


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
