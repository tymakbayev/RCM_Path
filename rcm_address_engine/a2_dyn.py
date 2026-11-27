"""A2 dynamics (protocol §5, §6 C4, §7 M6, §8 A4a/A4b): one operation stream applied in lock-step to
A (sidecar + delta), B (bucket table + delta) and the B-tree control; reseals, VACUUM points, checkpoints, scenarios."""

from __future__ import annotations

import hashlib
import random
import time
import uuid
from pathlib import Path

import numpy as np

import a2_core as C
import a2_pg as G
from a2_runs import DATA, RES, absent_keys, build_a, build_b, log, sha16, write_manifest

N = 1_000_000
OPS = 500_000
BATCH = 1_000
RESEALS = (120_000, 240_000, 360_000, 480_000, 500_000)
VACUUMS = (240_000, 480_000)
SAMPLE = 20_000
N_EXPLAIN = 2_000


# ------------------------------------------------------------------ operation stream (seed 13)
def load_extra(path: Path) -> list[tuple[bytes, str]]:
    if not Path(path).exists():  # portability: the reference runs made dyn.copy by hand (see prepare_data.py)
        raise SystemExit(f"нет {path}: сначала запустите python prepare_data.py (README, «Как повторить эксперименты»)")
    out = []
    with open(path, "rb") as f:
        for line in f:
            _seq, k, _v, payload = line.rstrip(b"\n").split(b"\t", 3)
            out.append((uuid.UUID(k.decode()).bytes, payload.replace(b"\\\\", b"\\").decode("utf-8")))
    return out


def make_ops(base: list[bytes], extra: list[tuple[bytes, str]], n_ins: int, n_upd: int, n_del: int, seed: int):
    rng = random.Random(seed)
    kinds = ["I"] * n_ins + ["U"] * n_upd + ["D"] * n_del
    rng.shuffle(kinds)
    live, where = list(base), {k: i for i, k in enumerate(base)}
    src = iter(extra)
    ops = []
    for kind in kinds:
        if kind == "I":
            k, payload = next(src)
            where[k] = len(live)
            live.append(k)
            ops.append(("I", k, payload))
        elif kind == "U":
            _, payload = next(src)
            ops.append(("U", live[rng.randrange(len(live))], payload))
        else:
            i = rng.randrange(len(live))
            k = live[i]
            last = live.pop()
            if i < len(live):
                live[i] = last
                where[last] = i
            del where[k]
            ops.append(("D", k, None))
    return ops


class Model:
    """Expected state: key -> [v, sha16, category, last write op]; deleted set; live list for sampling."""

    def __init__(self, keys: list[bytes], digs: list[bytes]) -> None:
        self.state = {k: [1, d, "base", -1] for k, d in zip(keys, digs)}
        self.deleted: set[bytes] = set()
        self.live = list(keys)
        self.where = {k: i for i, k in enumerate(keys)}

    def apply(self, op_no: int, kind: str, k: bytes, payload: str | None) -> None:
        if kind == "I":
            self.state[k] = [1, sha16(payload), "ins", op_no]
            self.deleted.discard(k)
            self.where[k] = len(self.live)
            self.live.append(k)
        elif kind == "U":
            s = self.state[k]
            s[0], s[1], s[3] = s[0] + 1, sha16(payload), op_no
            s[2] = "upd"
        else:
            del self.state[k]
            self.deleted.add(k)
            i = self.where.pop(k)
            last = self.live.pop()
            if i < len(self.live):
                self.live[i] = last
                self.where[last] = i

    def delta_live(self, since_op: int) -> int:
        return sum(1 for s in self.state.values() if s[3] >= since_op)


# ------------------------------------------------------------------ systems
class System:
    def __init__(self, name: str, kind: str, table: str, seq: str) -> None:
        self.name, self.kind, self.table, self.seq = name, kind, table, seq
        self.conn = G.connect()
        self.sa = self.sb = None
        self.client = None
        self.watermark = None
        self.delta = None
        self.gen = 0
        self.errors = 0
        self.reseals: list[dict] = []
        self.rows_per_page: float | None = None

    def reopen(self) -> None:
        self.conn.close()
        self.conn = G.connect()
        if self.kind == "C":
            self.client = G.Client(self.conn, self.table, "btree")
        else:
            self.client = G.Client(self.conn, self.table, self.kind, sa=self.sa, sb=self.sb, watermark=self.watermark)

    def lookup(self, kb: bytes):
        return self.client.get(kb)

    def apply(self, ops, start_no: int) -> list[str]:
        """Returns per-op outcome: 'ins' | 'upd' | 'del' | 'none' (no row to change) | 'dup' (insert over a live row)."""
        out = []
        t = self.table
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
                if self.kind == "C":
                    where, params = "k = %s", (k,)
                else:
                    where, params = "ctid = %s::tid AND k = %s", (row[0], k)
                if kind == "U":
                    cur = self.conn.execute(f"UPDATE {t} SET payload = %s, v = v + 1, seq = nextval('{self.seq}') "
                                            f"WHERE {where}", (payload,) + params, prepare=True)
                else:
                    cur = self.conn.execute(f"DELETE FROM {t} WHERE {where}", params, prepare=True)
                if cur.rowcount != 1:
                    self.errors += 1
                out.append("upd" if kind == "U" else "del")
        return out

    # ---- sizes
    def sizes(self) -> dict:
        c = self.conn
        tb = G.relinfo(c, self.table)
        if self.kind == "C":
            ix = G.index_size(c, f"{self.table}_bt")
            return {"address_main": ix["main_bytes"], "address_total": ix["total_bytes"], "table_bytes": tb["table_bytes"]}
        d = G.index_size(c, self.delta)
        f = Path(self.sa.mm.filename if self.sa else self.sb.mm.filename).stat().st_size
        man = (DATA / f"{self.name}.manifest.json").stat().st_size
        return {"sidecar_bytes": f, "manifest_bytes": man, "delta_main": d["main_bytes"], "delta_total": d["total_bytes"],
                "address_main": f + man + d["main_bytes"], "address_total": f + man + d["total_bytes"],
                "table_bytes": tb["table_bytes"]}


def setup(conn, extra_ref_keys, include_b: bool = True):
    for t in ("dyn_a", "dyn_c") + (("dyn_b_src",) if include_b else ()):
        log("load", t)
        G.load(conn, t, 0, N, DATA / "ref_dyn_base")
    keys = [bytes(k) for k in np.load(DATA / "ref_dyn_base.keys")]
    digs = [bytes(d) for d in np.load(DATA / "ref_dyn_base.dig")]
    for s in ("dyn_a_seq", "dyn_b_seq", "dyn_c_seq"):
        conn.execute(f"DROP SEQUENCE IF EXISTS {s}")
        conn.execute(f"CREATE SEQUENCE {s} START {N + 1}")
    a = System("dyn_a", "A", "dyn_a", "dyn_a_seq")
    a.gen = 1
    a.sa, info, _, _ = build_a(conn, "dyn_a", 1, N, "dyn_a")
    a.watermark, a.delta = N, "dyn_a_delta_g1"
    conn.execute(f"CREATE INDEX {a.delta} ON dyn_a (k) WHERE seq > {N}")
    a.reseals.append({"gen": 1, "build": {k: v for k, v in info.items() if k != "path"}})
    b = None
    if include_b:
        b = System("dyn_b", "B", "dyn_b_g1", "dyn_b_seq")
        b.gen = 1
        b.rows_per_page = N / G.relinfo(conn, "dyn_b_src")["pages"]
        b.sb, binfo, _, _ = build_b(conn, "dyn_b_src", "dyn_b_g1", 1, N, "dyn_b")
        conn.execute("DROP TABLE dyn_b_src")
        b.watermark, b.delta = N, "dyn_b_delta_g1"
        conn.execute(f"CREATE INDEX {b.delta} ON dyn_b_g1 (k) WHERE seq > {N}")
        b.reseals.append({"gen": 1, "build": {k: v for k, v in binfo.items() if k != "path"}})
    c = System("dyn_c", "C", "dyn_c", "dyn_c_seq")
    conn.execute("CREATE UNIQUE INDEX dyn_c_bt ON dyn_c USING btree (k)")
    conn.execute("ANALYZE dyn_c")
    for s in (a, b, c):
        if s is not None:
            s.reopen()
    return a, b, c, keys, digs


def delta_height(conn, index: str) -> dict:
    """Delta B-tree levels from bt_metap (pageinspect): level + 1 and fastlevel + 1; descriptive (A3 §3)."""
    level, fast = conn.execute("SELECT level, fastlevel FROM bt_metap(%s)", (index,)).fetchone()
    return {"levels": int(level) + 1, "fast_levels": int(fast) + 1}


def ops_sha256(ops) -> str:
    h = hashlib.sha256()
    for kind, kb, payload in ops:
        h.update(kind.encode() + kb + (hashlib.sha256(payload.encode("utf-8")).digest() if payload else b"-"))
    return h.hexdigest()


def quiesce(conn) -> int:
    """Protocol §4.1: no other active transaction in the database before a reseal."""
    return int(conn.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                            "AND pid <> pg_backend_pid() AND state <> 'idle'").fetchone()[0])


def reseal(conn, s: System, op_no: int) -> dict:
    t0 = time.perf_counter()
    busy = quiesce(conn)
    old_delta, old_table = s.delta, s.table
    read_bytes = G.relinfo(conn, s.table)["main_bytes"]
    s.gen += 1
    if s.kind == "A":
        s.sa.close()
        s.sa, info, _, _ = build_a(conn, s.table, s.gen, 0, s.name, repeatable=True)
        w = info["watermark"]
    else:
        s.sb.close()
        new = f"{s.name}_g{s.gen}"
        s.sb, info, _, _ = build_b(conn, s.table, new, s.gen, None, s.name, rows_per_page=s.rows_per_page)
        w = info["watermark"]
        s.table = new
    t_build = time.perf_counter() - t0
    s.delta = f"{s.name}_delta_g{s.gen}"
    t1 = time.perf_counter()
    conn.execute(f"CREATE INDEX {s.delta} ON {s.table} (k) WHERE seq > {w}")
    conn.execute(f"ANALYZE {s.table}")
    t_index = time.perf_counter() - t1
    path = Path(info["path"])
    write_manifest(DATA / f"{s.name}.manifest.json", kind=s.kind, generation=s.gen, file=path.name,
                   sha256=info["sha256"], oid=G.relinfo(conn, s.table)["oid"],
                   relfilenode=G.relinfo(conn, s.table)["relfilenode"], watermark=w, table=s.table, delta_index=s.delta)
    s.watermark = w
    s.reopen()
    if s.kind == "A":
        conn.execute(f"DROP INDEX {old_delta}")
    else:
        conn.execute(f"DROP TABLE {old_table}")
    written = path.stat().st_size + G.index_size(conn, s.delta)["main_bytes"]
    if s.kind == "B":
        written += G.relinfo(conn, s.table)["table_bytes"]
    rec = {"gen": s.gen, "after_op": op_no, "busy_backends": busy, "build_s": t_build, "delta_index_s": t_index,
           "total_s": time.perf_counter() - t0, "read_bytes": read_bytes, "written_bytes": written,
           "build": {k: v for k, v in info.items() if k != "path"}}
    s.reseals.append(rec)
    return rec


# ------------------------------------------------------------------ checkpoints
def checkpoint(cp_no: int, name: str, op_no: int, systems, model: Model, ever_keys: np.ndarray, since: int) -> dict:
    rng = random.Random(21 + cp_no)
    live = rng.sample(model.live, min(SAMPLE, len(model.live)))
    dele = sorted(model.deleted)
    random.Random(41 + cp_no).shuffle(dele)
    dele = dele[:SAMPLE]
    absent = absent_keys(ever_keys, SAMPLE, 61 + cp_no)
    rec = {"cp": cp_no, "name": name, "after_op": op_no, "live_keys": len(model.state), "deleted_keys": len(model.deleted),
           "delta_live_rows": model.delta_live(since), "systems": {}}
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
        for kb in live[:N_EXPLAIN]:
            q = s.client.query(kb)
            e = G.explain(s.conn, *q)
            by_cat.setdefault(model.state[kb][2], []).append(e["buffers"])
        cat_means = {c: float(np.mean(v)) for c, v in by_cat.items()}

        def mean_buf(keys):
            vals = []
            for kb in keys[:500]:
                q = s.client.query(kb)
                vals.append(0 if q is None else G.explain(s.conn, *q)["buffers"])
            return float(np.mean(vals)) if vals else 0.0
        reader = G.connect()
        pk = live[:N_EXPLAIN]
        st = G.statio_split(s.conn, reader, s.table, lambda: [s.lookup(kb) for kb in pk] and len(pk))
        reader.close()
        rec["systems"][s.name] = {
            "errors": err, "client_us": G.pct(times),
            "buffers_by_category": cat_means, "buffers_equal_mix": float(np.mean(list(cat_means.values()))),
            "buffers_raw_mix": float(np.mean([b for v in by_cat.values() for b in v])),
            "category_counts": {c: len(v) for c, v in by_cat.items()},
            "buffers_deleted": mean_buf(dele), "buffers_absent": mean_buf(absent), "statio_live": st,
            "sizes": s.sizes(), "delta_height": delta_height(s.conn, s.delta) if s.delta else None}
        sz = rec["systems"][s.name]["sizes"]
        rec["systems"][s.name]["address_main_per_live_key"] = sz["address_main"] / len(model.state)
        rec["systems"][s.name]["address_total_per_live_key"] = sz["address_total"] / len(model.state)
        s.errors += sum(err.values())
    stats = systems[0].conn.execute("SELECT relname, n_live_tup, n_dead_tup FROM pg_stat_user_tables "
                                    "WHERE relname LIKE 'dyn%'").fetchall()
    rec["pg_stat_tuples"] = {r[0]: {"live": r[1], "dead": r[2]} for r in stats}
    log("checkpoint", cp_no, name, op_no, {k: (v["errors"], round(v["buffers_equal_mix"], 3),
                                              round(v["address_main_per_live_key"], 3)) for k, v in rec["systems"].items()})
    return rec


# ------------------------------------------------------------------ scenarios (10 000 rows, A and B)
def scenarios(conn) -> dict:
    out = {}
    G.load(conn, "scn_src", 0, 10_000, DATA / "scn")
    keys = [bytes(k) for k in np.load(DATA / "scn.keys")]
    digs = [bytes(d) for d in np.load(DATA / "scn.dig")]
    extra = load_extra(DATA / "dyn.copy")[-2000:]  # payloads and fresh keys from the end of the dynamics stream
    for kind in ("A", "B"):
        name = f"scn_{kind.lower()}"
        seqn = f"{name}_seq"
        conn.execute(f"DROP SEQUENCE IF EXISTS {seqn}")
        conn.execute(f"CREATE SEQUENCE {seqn} START 10001")
        if kind == "A":
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            conn.execute(f"CREATE TABLE {name} (LIKE scn_src) WITH (autovacuum_enabled = false)")
            conn.execute(f"INSERT INTO {name} SELECT * FROM scn_src ORDER BY seq")
            conn.execute(f"VACUUM (ANALYZE) {name}")
            s = System(name, "A", name, seqn)
            s.gen = 1
            s.sa, _, _, _ = build_a(conn, name, 1, 10_000, name)
        else:
            s = System(name, "B", f"{name}_g1", seqn)
            s.gen = 1
            s.rows_per_page = 10_000 / G.relinfo(conn, "scn_src")["pages"]
            s.sb, _, _, _ = build_b(conn, "scn_src", f"{name}_g1", 1, 10_000, name)
        s.watermark, s.delta = 10_000, f"{name}_delta_g1"
        conn.execute(f"CREATE INDEX {s.delta} ON {s.table} (k) WHERE seq > 10000")
        s.reopen()
        model = Model(keys, digs)
        pay = iter(extra)
        results = []

        def run(label, ops, expect_outcomes, check_keys):
            got = s.apply(ops, 0)
            for i, (kind_, kb, payload) in enumerate(ops):
                if got[i] in ("ins", "upd", "del"):
                    model.apply(0, kind_, kb, payload)
            ok = got == expect_outcomes
            for kb in check_keys:
                row = s.lookup(kb)
                st = model.state.get(kb)
                if st is None:
                    ok &= row is None
                else:
                    ok &= row is not None and row[1] == st[0] and sha16(row[3]) == st[1]
            results.append({"scenario": label, "ok": bool(ok), "outcomes": got})

        for phase in ("before_reseal", "after_reseal"):
            base = [k for k in model.live if model.state[k][2] == "base"]
            k1, k2, k3, k4, k5 = base[:5]
            p = lambda: next(pay)[1]  # noqa: E731
            run(f"{phase}: UPDATE a key already in the delta", [("U", k1, p()), ("U", k1, p())], ["upd", "upd"], [k1])
            run(f"{phase}: three UPDATEs -> v = 4", [("U", k2, p()), ("U", k2, p()), ("U", k2, p())],
                ["upd"] * 3, [k2])
            ok_v = s.lookup(k2)
            results[-1]["ok"] &= ok_v is not None and ok_v[1] == model.state[k2][0]
            run(f"{phase}: UPDATE -> DELETE", [("U", k3, p()), ("D", k3, None)], ["upd", "del"], [k3])
            run(f"{phase}: DELETE -> INSERT same key", [("D", k4, None), ("I", k4, p())], ["del", "ins"], [k4])
            run(f"{phase}: repeated DELETE", [("D", k5, None), ("D", k5, None)], ["del", "none"], [k5])
            ghost = next(pay)[0]
            run(f"{phase}: UPDATE of an absent key", [("U", ghost, p())], ["none"], [ghost])
            # VACUUM and space reuse: delete/update, VACUUM, insert/update, check actual ctids and neighbours
            victims = base[5:205]
            movers = base[205:405]
            run(f"{phase}: DELETE 200 + UPDATE 200", [("D", k, None) for k in victims] + [("U", k, p()) for k in movers],
                ["del"] * 200 + ["upd"] * 200, victims[:50] + movers[:50])
            conn.execute(f"VACUUM {s.table}")
            fresh = [next(pay) for _ in range(300)]
            more = base[405:505]
            max_page_before = G.relinfo(conn, s.table)["pages"]
            run(f"{phase}: after VACUUM: INSERT 300 + UPDATE 100",
                [("I", k, pl) for k, pl in fresh] + [("U", k, p()) for k in more], ["ins"] * 300 + ["upd"] * 100,
                [k for k, _ in fresh] + more + base[505:1505])
            pages = [int(r[0]) for r in conn.execute(
                f"SELECT (ctid::text::point)[0]::int FROM {s.table} WHERE k = ANY(%s)",
                ([uuid.UUID(bytes=k) for k, _ in fresh],)).fetchall()]
            results[-1]["inserted_rows_on_reused_pages"] = int(sum(1 for pg in pages if pg < max_page_before - 1))
            if phase == "before_reseal":
                results.append({"scenario": "reseal between phases", "ok": True, **{
                    k: v for k, v in reseal(conn, s, 0).items() if k in ("gen", "total_s", "busy_backends")}})
        out[kind] = results
        conn.execute(f"DROP TABLE IF EXISTS {s.table}")
        s.conn.close()
    conn.execute("DROP TABLE IF EXISTS scn_src")
    return out


# ------------------------------------------------------------------ main
def cmd_dynamics(reseal_every: int | None = None, include_b: bool = True, out_name: str = "a2_dynamics",
                 run_scenarios: bool = True) -> None:
    """A2 defaults: reseals at RESEALS, A + B + control, scenarios. A3: reseal_every=30_000, include_b=False."""
    reseals = RESEALS if reseal_every is None else tuple(range(reseal_every, OPS, reseal_every)) + (OPS,)
    conn = G.connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS pageinspect")
    log("dynamics: setup", out_name, "reseals at", reseals)
    extra = load_extra(DATA / "dyn.copy")
    extra_keys = np.load(DATA / "dyn.keys")
    a, b, c, keys, digs = setup(conn, extra_keys, include_b)
    ever = np.concatenate([np.load(DATA / "ref_dyn_base.keys"), extra_keys])
    ops = make_ops(keys, extra, 250_000, 150_000, 100_000, 13)
    stream_sha = ops_sha256(ops)
    log("operation stream sha256", stream_sha)
    model = Model(keys, digs)
    systems = tuple(s for s in (a, b, c) if s is not None)
    sidecars = tuple(s for s in (a, b) if s is not None)
    out = {"reseal_at": list(reseals), "systems": [s.name for s in systems], "ops_sha256": stream_sha,
           "checkpoints": [], "delta_trace": [], "vacuum": []}
    t_vac = {s.name: 0.0 for s in systems}
    cp = 0
    out["checkpoints"].append(checkpoint(cp, "base", 0, systems, model, ever, 0))
    since = 0
    t_ops = {s.name: 0.0 for s in systems}
    for start in range(0, OPS, BATCH):
        batch = ops[start: start + BATCH]
        for s in systems:
            t = time.perf_counter()
            got = s.apply(batch, start)
            t_ops[s.name] += time.perf_counter() - t
            expected = ["ins" if k == "I" else "upd" if k == "U" else "del" for k, _, _ in batch]
            if got != expected:
                s.errors += sum(1 for x, y in zip(got, expected) if x != y)
        for i, (kind, kb, payload) in enumerate(batch):
            model.apply(start + i, kind, kb, payload)
        op_no = start + BATCH
        out["delta_trace"].append({"after_op": op_no, "delta_live_rows": model.delta_live(since),
                                   **{s.name: G.index_size(conn, s.delta)["main_bytes"] for s in sidecars},
                                   **{s.name + "_height": delta_height(conn, s.delta) for s in sidecars}})
        if op_no in VACUUMS:
            for s in systems:
                tv = time.perf_counter()
                conn.execute(f"VACUUM {s.table}")
                t_vac[s.name] += time.perf_counter() - tv
            out["vacuum"].append(op_no)
            log("vacuum after", op_no)
        if op_no in reseals:
            cp += 1
            out["checkpoints"].append(checkpoint(cp, "before_reseal", op_no, systems, model, ever, since))
            for s in sidecars:
                r = reseal(conn, s, op_no)
                log("reseal", s.name, {k: r[k] for k in ("gen", "total_s", "read_bytes", "written_bytes")})
            since = op_no
            cp += 1
            out["checkpoints"].append(checkpoint(cp, "after_reseal", op_no, systems, model, ever, since))
        if op_no % 50_000 == 0:
            log("ops", op_no, {k: round(v, 1) for k, v in t_ops.items()})
    log("restart")
    t0 = time.perf_counter()
    G.pg_ctl("restart")
    conn = G.connect()
    for s in systems:
        s.reopen()
    first = a.lookup(model.live[0])
    out["restart_to_first_answer_s"] = time.perf_counter() - t0
    out["restart_first_ok"] = first is not None
    cp += 1
    out["checkpoints"].append(checkpoint(cp, "after_restart", OPS, systems, model, ever, since))
    out["ops_time_s"] = t_ops
    out["maintenance_s_per_op"] = {s.name: (t_ops[s.name] + sum(r.get("total_s", 0) for r in s.reseals[1:])) / OPS
                                   for s in systems}  # A2 definition (without VACUUM)
    out["vacuum_s"] = t_vac
    out["maintenance_with_vacuum_s_per_op"] = {  # A3 §4 definition
        s.name: (t_ops[s.name] + sum(r.get("total_s", 0) for r in s.reseals[1:]) + t_vac[s.name]) / OPS for s in systems}
    for s in sidecars:
        pauses = [r["total_s"] for r in s.reseals[1:]]
        total = t_ops[s.name] + sum(pauses) + t_vac[s.name]
        out.setdefault("reseal_pauses", {})[s.name] = {"count": len(pauses), "sum_s": sum(pauses), "max_s": max(pauses),
                                                      "share_of_maintenance": sum(pauses) / total}
    out["reseals"] = {s.name: s.reseals for s in sidecars}
    out["errors"] = {s.name: s.errors for s in systems}
    if run_scenarios:
        log("scenarios")
        out["scenarios"] = scenarios(conn)
    for t in ("dyn_a", "dyn_c") + ((b.table,) if b else ()):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    G.write_json(RES / f"{out_name}.json", out)
    log("dynamics done", out["errors"])
