"""A2 PostgreSQL layer: server control, COPY load from a2_gen, (k, page) export, lookup clients, EXPLAIN and pg_statio.

Protocol: A2_protocol.md §3-§7 (edition 2). Runs in rcm_address_engine/.venv-pg.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import psycopg
import psycopg.sql

import a2_core as C

ROOT = Path(__file__).resolve().parent
# Stand configuration from the environment (README, "Configure the stand"). The reference stand of the published
# runs: PostgreSQL 18.7 zip in C:\gams\pg18 (pg_ctl in pgsql\bin, cluster in data, log server.log), port 5433,
# superuser role "bench" with a password file; the record generator ran in a second venv (JARVIS project).
_PG_CTL_ON_PATH = shutil.which("pg_ctl")
PG_BIN = Path(os.environ.get("RCM_PG_BIN") or (Path(_PG_CTL_ON_PATH).parent if _PG_CTL_ON_PATH else "."))
PG_DATA = os.environ.get("RCM_PG_DATA") or os.environ.get("PGDATA") or ""
PG_LOG = os.environ.get("RCM_PG_LOG") or (str(Path(PG_DATA) / "rcm_server.log") if PG_DATA else "rcm_server.log")
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = int(os.environ.get("PGPORT", "5432"))
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PWFILE = os.environ.get("RCM_PG_PWFILE", "")  # optional; otherwise libpq uses PGPASSWORD or ~/.pgpass
DB = os.environ.get("RCM_DB", "a2")
GEN_PY = Path(sys.executable)
GEN = ROOT / "a2_gen.py"
DATA = ROOT / "data"
COLS = "ctid, v, seq, payload"


# ------------------------------------------------------------------ server
PG_CTL = PG_BIN / ("pg_ctl.exe" if os.name == "nt" else "pg_ctl")


def require_data_dir() -> str:
    """The cluster directory: the runs restart the server (cold cache) and measure the free space of its volume."""
    if not PG_DATA:
        raise RuntimeError("set RCM_PG_DATA (or PGDATA) to the data directory of a dedicated test cluster")
    return PG_DATA


def same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


_CLUSTER_CHECKED = False


def check_cluster() -> None:
    """The SQL connection (PGHOST/PGPORT) must reach the cluster that pg_ctl controls and whose volume is measured
    (RCM_PG_DATA); otherwise cold-start numbers and free-space checks would describe another server. Checked once per
    process, before any measurement: a2.Server on entry, A7/A8 at start-up."""
    global _CLUSTER_CHECKED
    if _CLUSTER_CHECKED:
        return
    with connect("postgres") as c:
        served = c.execute("SELECT current_setting('data_directory')").fetchone()[0]
    if not same_path(served, require_data_dir()):
        raise RuntimeError(f"PGHOST/PGPORT reach the cluster in {served!r}, but RCM_PG_DATA is {PG_DATA!r}: "
                           "point both at the same dedicated local test cluster")
    _CLUSTER_CHECKED = True


def pg_ctl(action: str) -> None:
    if action in ("stop", "restart"):
        check_cluster()  # no-op after the start-up check
    cmd = [str(PG_CTL), "-D", require_data_dir(), "-l", PG_LOG, "-w"]
    cmd += [action] + (["-m", "fast"] if action in ("stop", "restart") else [])
    # no pipes: the server inherits pg_ctl's handles, and a captured pipe would never close (hang seen in P0)
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"pg_ctl {action} failed (code {r.returncode}), see {PG_LOG}")


def pg_running() -> bool:
    return subprocess.run([str(PG_CTL), "-D", require_data_dir(), "status"], capture_output=True).returncode == 0


def connect(db: str = DB, autocommit: bool = True) -> psycopg.Connection:
    kw = {"host": PG_HOST, "port": PG_PORT, "user": PG_USER, "dbname": db, "autocommit": autocommit}
    if PG_PWFILE:
        kw["password"] = Path(PG_PWFILE).read_text().strip()
    return psycopg.connect(**kw)


def ensure_db() -> None:
    with connect("postgres") as c:
        if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,)).fetchone():
            c.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(DB)))  # RCM_DB is free text


def show_all(conn) -> dict:
    return {name: setting for name, setting in conn.execute("SELECT name, setting FROM pg_settings").fetchall()}


# ------------------------------------------------------------------ tables
def create_table(conn, table: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} (seq bigint NOT NULL, k uuid NOT NULL, v integer NOT NULL, payload text NOT NULL)"
                 " WITH (autovacuum_enabled = false)")


def load(conn, table: str, start: int, count: int, ref: Path, workers: int = 6) -> dict:
    """COPY stream positions [start, start+count) from a2_gen into a new empty table; VACUUM (ANALYZE)."""
    t0 = time.perf_counter()
    create_table(conn, table)
    proc = subprocess.Popen([str(GEN_PY), str(GEN), str(start), str(count), "--ref", str(ref), "--workers",
                             str(workers)], stdout=subprocess.PIPE, cwd=str(ROOT))
    with conn.cursor() as cur, cur.copy(f"COPY {table} (seq, k, v, payload) FROM STDIN") as cp:
        while chunk := proc.stdout.read(1 << 20):
            cp.write(chunk)
    if proc.wait() != 0:
        raise RuntimeError("a2_gen failed")
    t1 = time.perf_counter()
    conn.execute(f"VACUUM (ANALYZE) {table}")
    return {"copy_s": t1 - t0, "vacuum_s": time.perf_counter() - t1}


def relinfo(conn, table: str) -> dict:
    oid, relfilenode, toast = conn.execute(
        "SELECT oid, relfilenode, reltoastrelid FROM pg_class WHERE relname = %s", (table,)).fetchone()
    size = conn.execute("SELECT pg_relation_size(%s), pg_table_size(%s)", (table, table)).fetchone()
    toast_size = conn.execute("SELECT pg_relation_size(%s)", (toast,)).fetchone()[0] if toast else 0
    toast_rows = conn.execute(f"SELECT count(*) FROM pg_toast.{conn.execute('SELECT relname FROM pg_class WHERE oid = %s', (toast,)).fetchone()[0]}").fetchone()[0] if toast else 0
    return {"oid": int(oid), "relfilenode": int(relfilenode), "pages": int(size[0]) // 8192, "main_bytes": int(size[0]),
            "table_bytes": int(size[1]), "toast_bytes": int(toast_size), "toast_rows": int(toast_rows)}


def index_size(conn, index: str) -> dict:
    main, total = conn.execute("SELECT pg_relation_size(%s), pg_total_relation_size(%s)", (index, index)).fetchone()
    opts = conn.execute("SELECT reloptions FROM pg_class WHERE relname = %s", (index,)).fetchone()[0]
    return {"main_bytes": int(main), "total_bytes": int(total), "reloptions": opts}


def read_kp(conn, table: str, with_seq: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(keys uint8[n,16], heap page uint32[n], seq int64[n] or None) via binary COPY of k and ctid page."""
    cols = "k, (ctid::text::point)[0]::int4" + (", seq" if with_seq else "")
    buf = bytearray()
    with conn.cursor() as cur, cur.copy(f"COPY (SELECT {cols} FROM {table}) TO STDOUT (FORMAT binary)") as cp:
        for data in cp:
            buf += data
    fields = [("n", ">i2"), ("l1", ">i4"), ("k", "V16"), ("l2", ">i4"), ("p", ">i4")]
    if with_seq:
        fields += [("l3", ">i4"), ("s", ">i8")]
    dt = np.dtype(fields)
    n = (len(buf) - 19 - 2) // dt.itemsize
    arr = np.frombuffer(bytes(buf), dt, count=n, offset=19)
    if n and (arr["l1"][0] != 16 or arr["l2"][0] != 4):
        raise RuntimeError("unexpected COPY binary layout")
    keys = np.frombuffer(arr["k"].tobytes(), np.uint8).reshape(n, 16)
    return keys, arr["p"].astype(np.uint32), (arr["s"].astype(np.int64) if with_seq else None)


def load_bucketed(conn, src: str, dst: str, seed: int, b: int) -> dict:
    """Variant B table: a NEW empty table filled in (bucket, seq) order (protocol §4.3)."""
    t0 = time.perf_counter()
    keys, _, seq = read_kp(conn, src, with_seq=True)
    bucket = C.buckets_of(keys, seed, b)
    order = np.lexsort((seq, bucket))
    conn.execute("DROP TABLE IF EXISTS a2_ord")
    conn.execute("CREATE UNLOGGED TABLE a2_ord (seq bigint NOT NULL, pos bigint NOT NULL)")
    with conn.cursor() as cur, cur.copy("COPY a2_ord (seq, pos) FROM STDIN") as cp:
        lines = (b"%d\t%d\n" % (s, p) for p, s in enumerate(seq[order].tolist()))
        batch = []
        for line in lines:
            batch.append(line)
            if len(batch) == 100_000:
                cp.write(b"".join(batch))
                batch = []
        if batch:
            cp.write(b"".join(batch))
    create_table(conn, dst)
    conn.execute(f"INSERT INTO {dst} SELECT t.seq, t.k, t.v, t.payload FROM {src} t JOIN a2_ord o USING (seq)"
                 " ORDER BY o.pos")
    conn.execute("DROP TABLE a2_ord")
    conn.execute(f"VACUUM (ANALYZE) {dst}")
    return {"load_s": time.perf_counter() - t0}


# ------------------------------------------------------------------ clients
def tid(page: int) -> str:
    return f"({page},0)"


class Client:
    """One lookup method on one table. get(key) -> (ctid, v, seq, payload) or None."""

    def __init__(self, conn, table: str, kind: str, *, sa: C.SidecarA | None = None, sb: C.SidecarB | None = None,
                 watermark: int | None = None, mutant: str | None = None) -> None:
        self.conn, self.table, self.kind, self.sa, self.sb, self.w, self.mutant = conn, table, kind, sa, sb, watermark, mutant
        keycheck = "" if mutant == "c" else " AND k = %s"
        self.sql_tid = f"SELECT {COLS} FROM {table} WHERE ctid >= %s::tid AND ctid < %s::tid{keycheck}"
        self.sql_key = f"SELECT {COLS} FROM {table} WHERE k = %s"
        if watermark is not None:
            self.sql_delta = f"SELECT {COLS} FROM {table} WHERE k = %s AND seq > {int(watermark)}"
            self.sql_union = (f"(SELECT {COLS} FROM {table} WHERE k = %s AND seq > {int(watermark)}) UNION ALL "
                              f"(SELECT {COLS} FROM {table} WHERE ctid >= %s::tid AND ctid < %s::tid{keycheck}) LIMIT 1")
        self.addr_ns = 0

    def address(self, kb: bytes):
        t = time.perf_counter_ns()
        if self.kind == "A":
            p = self.sa.page_of(kb)
            rng = None if p is None else (p, p)
        elif self.kind == "B":
            rng = self.sb.range_of(kb)
        else:
            rng = None
        self.addr_ns = time.perf_counter_ns() - t
        return rng

    def query(self, kb: bytes):
        """(sql, params) of the lookup, or None when no SQL is needed (static A, fingerprint miss)."""
        k = uuid.UUID(bytes=kb)
        if self.kind in ("btree", "hash", "brin"):
            return self.sql_key, (k,)
        rng = self.address(kb)
        if self.w is None:
            if rng is None:
                return None
            return self.sql_tid, (tid(rng[0]), tid(rng[1] + 1)) + (() if self.mutant == "c" else (k,))
        if rng is None:
            return self.sql_delta, (k,)
        return self.sql_union, (k, tid(rng[0]), tid(rng[1] + 1)) + (() if self.mutant == "c" else (k,))

    def get(self, kb: bytes):
        q = self.query(kb)
        if q is None:
            return None
        rows = self.conn.execute(q[0], q[1], prepare=True).fetchall()
        if self.mutant == "c":
            return rows[0] if rows else None  # mutant: no key check anywhere -> first row of the page
        if len(rows) > 1:
            raise RuntimeError(f"{len(rows)} visible rows for one key")
        return rows[0] if rows else None


def literal(sql: str, params: tuple) -> str:
    out = sql
    for p in params:
        lit = f"'{p}'::uuid" if isinstance(p, uuid.UUID) else f"'{p}'"
        out = out.replace("%s", lit, 1)
    return out


def explain(conn, sql: str, params: tuple) -> dict:
    """Top-node executor buffers (no summing of the tree), planning buffers, execution time, node types (protocol M3)."""
    res = conn.execute("EXPLAIN (ANALYZE, BUFFERS, SERIALIZE TEXT, FORMAT JSON) " + literal(sql, params)).fetchone()[0]
    p = res[0]
    top = p["Plan"]
    nodes = []

    def walk(n, depth=0):
        nodes.append((depth, n["Node Type"], n.get("Actual Loops", 0), n.get("Actual Rows", 0)))
        for ch in n.get("Plans", []):
            walk(ch, depth + 1)
    walk(top)
    plan_buf = p.get("Planning", {})
    ser = p.get("Serialization", {})
    return {"buffers": top.get("Shared Hit Blocks", 0) + top.get("Shared Read Blocks", 0),
            "read": top.get("Shared Read Blocks", 0), "exec_ms": p["Execution Time"],
            "planning_buffers": plan_buf.get("Shared Hit Blocks", 0) + plan_buf.get("Shared Read Blocks", 0),
            "serialize_buffers": ser.get("Shared Hit Blocks", 0) + ser.get("Shared Read Blocks", 0), "nodes": nodes}


STATIO = ("SELECT heap_blks_read + heap_blks_hit, coalesce(idx_blks_read + idx_blks_hit, 0), "
          "coalesce(toast_blks_read + toast_blks_hit, 0) FROM pg_statio_user_tables WHERE relname = %s")


def statio_split(client_conn, reader_conn, table: str, run_batch) -> dict:
    """Per-relation buffer accesses of one isolated batch: pg_statio deltas after a forced stats flush (protocol M3)."""
    def read():
        reader_conn.execute("SELECT pg_stat_clear_snapshot()")
        return np.array(reader_conn.execute(STATIO, (table,)).fetchone(), dtype=np.int64)
    client_conn.execute("SELECT pg_stat_force_next_flush()")
    time.sleep(0.2)
    before = read()
    n = run_batch()
    client_conn.execute("SELECT pg_stat_force_next_flush()")
    after, waited = read(), 0.0
    while waited < 5.0:  # flush is asynchronous to the reader: wait until the counters stop moving
        time.sleep(0.2)
        waited += 0.2
        again = read()
        if np.array_equal(again, after):
            break
        after = again
    d = after - before
    return {"queries": n, "heap_per_q": float(d[0]) / n, "index_per_q": float(d[1]) / n, "toast_per_q": float(d[2]) / n}


def pct(values, ps=(50, 95, 99)) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(v, p)) for p in ps} | {"mean": float(v.mean()), "n": int(len(v))}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def r_bits_for(n_pages: int) -> int:
    return max(1, math.ceil(math.log2(max(n_pages, 2))))
