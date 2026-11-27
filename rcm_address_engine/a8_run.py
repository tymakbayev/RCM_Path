"""A8 run (protocol A8_protocol.md edition 2, addenda 2026-11-17): why existing-key lookups on the single table cost
more at N10 in the A7 co-run. Run with .venv-pg:
  a8_run.py selfcheck | a8_run.py run
A2-A7 code is used unchanged; a7_run.checkpoint7 is wrapped at run time (module attribute), the file is not edited."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import a2_pg as G
import a6_pghash as PH
import hoststats as H
import a6_run as A6
import a7_run as A7
from a2_dyn import Model, load_extra, make_ops, ops_sha256
from a2_runs import DATA, RES, log, sha16
from a5_run import rss_mb

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT.parent / "docs" / "experiments" / "A8_protocol.md"
THR = 1.30
BLOCK = 8192
PER_LEAF = 6
STAGE_A_OPS = 200_000
CPS_MAIN = (0, 100_000, 200_000)
SEED_SETS, SEED_W, SEED_Z = 808, 809, 811
PAIRS = (("P", "K"), ("K", "P"), ("K", "P"), ("P", "K"), ("P", "K"), ("K", "P"), ("K", "P"), ("P", "K"))
HIST_EDGES = (0, 25, 50, 100, 200, 400, 800, 1600, float("inf"))
MUTANTS = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11")
SETTINGS = A7.SETTINGS + ("io_method", "io_workers", "io_combine_limit", "effective_io_concurrency",
                          "maintenance_io_concurrency", "block_size", "track_io_timing")
MIN_FREE = 24e9
JOURNAL = ["flush", "select1", "obs_flush", "clear", "read"]
SUPPORTED, UNSUPPORTED = "поддержана", "не поддержана"
NOT_SET = "НЕ УСТАНОВЛЕНО"
CONTAM_MAX = 0.25
MAIN = {"M": 5000, "b1": 16, "b2": 8, "b3_pairs": 8, "z_passes": 5, "pub_checks": 2}
SMALL = {"M": 500, "b1": 2, "b2": 2, "b3_pairs": 2, "z_passes": 5, "pub_checks": 99}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_protocol() -> None:
    want = (RES / "a8_protocol.sha256").read_text(encoding="utf-8").split()[0]
    if sha_file(PROTOCOL) != want:
        raise SystemExit("A8_protocol.md changed after it was frozen")


def impl_hashes() -> dict:
    out = A7.impl_hashes()
    out["a8_run.py"] = sha_file(ROOT / "a8_run.py")
    return out


# ------------------------------------------------------------------ memory and device (§3.2), see hoststats.py
def disk_counters() -> dict | None:
    """Device counters of the volume that holds the PostgreSQL data directory (C: on the reference stand)."""
    return H.disk_counters(G.require_data_dir())


def mem(where: str, bufs: dict | None = None) -> dict:
    return {"where": where, "t": time.time(), **H.memory(),
            "rss": rss_mb(), "disk": disk_counters(), "pg_buffers": bufs}


# ------------------------------------------------------------------ relations and PostgreSQL buffers by relation
def oid_of(conn, name: str) -> int:
    return int(conn.execute("SELECT %s::regclass::oid", (name,)).fetchone()[0])


def leaf_oids(conn, parent: str) -> list[int]:
    rows = conn.execute("SELECT relid::oid FROM pg_partition_tree(%s::regclass) WHERE isleaf ORDER BY 1",
                        (parent,)).fetchall()
    return [int(r[0]) for r in rows]


def index_oids(conn, heaps: list[int]) -> list[int]:
    rows = conn.execute("SELECT indexrelid::oid FROM pg_index WHERE indrelid = ANY(%s::oid[]) ORDER BY 1",
                        (heaps,)).fetchall()
    return [int(r[0]) for r in rows]


class Rels:
    """Storage relations of the three tables (heaps and their indexes), resolved once."""

    def __init__(self, conn, ctx: A7.Ctx) -> None:
        self.s_heap = [oid_of(conn, ctx.ts)]
        self.s_idx = index_oids(conn, self.s_heap)
        self.p_heap = leaf_oids(conn, ctx.tp)
        self.p_idx = index_oids(conn, self.p_heap)
        self.a_heap = leaf_oids(conn, ctx.ta)
        self.a_idx = index_oids(conn, self.a_heap)
        self.s_all, self.p_all, self.a_all = self.s_heap + self.s_idx, self.p_heap + self.p_idx, self.a_heap + self.a_idx


def buffers_of(obs, oids: list[int]) -> int:
    if not oids:
        return 0
    r = obs.execute("SELECT count(*) FROM pg_buffercache b WHERE b.reldatabase = "
                    "(SELECT oid FROM pg_database WHERE datname = current_database()) AND b.relfilenode = "
                    "ANY(SELECT pg_relation_filenode(o::regclass) FROM unnest(%s::oid[]) o)", (oids,)).fetchone()
    return int(r[0])


def bufsnap(obs, rels: Rels) -> dict:
    """(§3.2): PostgreSQL buffers by relation group (R5 Codex 262)."""
    return {"s7_heap": buffers_of(obs, rels.s_heap), "s7_idx": buffers_of(obs, rels.s_idx),
            "p7_heap": buffers_of(obs, rels.p_heap), "p7_idx": buffers_of(obs, rels.p_idx),
            "a7_heap": buffers_of(obs, rels.a_heap), "a7_idx": buffers_of(obs, rels.a_idx)}


# ------------------------------------------------------------------ stage A wrapper (§2.1, R9 Codex 254)
_ORIG_CHECKPOINT = A7.checkpoint7
_WRAP: dict = {"rels": None}


def _checkpoint_wrapper(ctx, cp_no, op_no, model, ever, a, work, diag=True):
    """Snapshots of the cumulative B7 timers (independent copies), memory and buffers before the unchanged A7 C7."""
    snap = {r: copy.deepcopy(c.kind_ns) for r, c in work.items()}
    bufs = None
    try:
        c = G.connect()
        try:
            if _WRAP["rels"] is None:
                _WRAP["rels"] = Rels(c, ctx)
            bufs = bufsnap(c, _WRAP["rels"])
        finally:
            c.close()
    except Exception as e:  # noqa: BLE001  an observation, not a correctness rule
        bufs = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    m = mem(f"c7 {cp_no}", bufs)
    rec = _ORIG_CHECKPOINT(ctx, cp_no, op_no, model, ever, a, work, diag)
    rec["a8"] = {"kind_ns": snap, "mem": m}
    return rec


def install_wrapper() -> None:
    _WRAP["rels"] = None
    A7.checkpoint7 = _checkpoint_wrapper


def schedule_prefix_ok(nb: int) -> bool:
    """(п5): schedule(nb) is the first nb batches of the A7 schedule of 500."""
    return A7.schedule(nb) == A7.schedule(500)[:nb]


def d1_from_checkpoints(cps: list[dict]) -> dict:
    """Д1: (a) U+D lookup time s / dir between the second and third C7 (wrapper snapshots); (b) C7 p50 s / par."""
    if len(cps) < 3 or any("a8" not in c for c in cps[1:3]):
        return {"defined": False}
    k1, k2 = cps[1]["a8"]["kind_ns"], cps[2]["a8"]["kind_ns"]

    def lookup_us(route: str) -> float | None:
        ns = sum(k2[route][x][0] - k1[route][x][0] for x in "UD")
        n = sum(k2[route][x][2] - k1[route][x][2] for x in "UD")
        return ns / n / 1000 if n else None

    s_us, d_us = lookup_us("s"), lookup_us("dir")
    ra = A7.div(s_us, d_us)
    rb = [A7.div(c["routes"]["s"]["client_us"]["p50"], c["routes"]["par"]["client_us"]["p50"]) for c in cps[1:3]]
    cond_a = ra is not None and ra >= THR
    cond_b = any(x is not None and x >= THR for x in rb)
    return {"defined": True, "lookup_ud_us": {"s": s_us, "dir": d_us}, "a_ratio": ra, "a_cond": cond_a,
            "b_ratios": rb, "b_cond": cond_b, "repeated": cond_a and cond_b,
            "ops_between": [cps[1]["after_op"], cps[2]["after_op"]]}


def wrapper_ok(stage_a: dict, ncp: int) -> dict:
    """(п7, R6 Codex 262): a snapshot at every C7; at each point the operation counts of every route equal the counts
    of the batches applied so far by that route; the last snapshot equals the final cumulative timers."""
    cps = stage_a.get("checkpoints") or []
    if len(cps) != ncp or any("a8" not in c for c in cps):
        return {"ok": False, "why": "snapshots missing"}
    bad = []
    for c in cps:
        upto = [b for b in stage_a["batches"] if b["op_no"] <= c["after_op"]]
        for r in ("s", "par", "dir"):
            bs = upto if r == "s" else [b for b in upto if b["route"] == r]
            want = {k: sum(b[f"n{k}"] for b in bs) for k in "IUD"}
            got = {k: c["a8"]["kind_ns"][r][k][2] for k in "IUD"}
            if want != got:
                bad.append([c["cp"], r, want, got])
    last_ok = cps[-1]["a8"]["kind_ns"] == stage_a["kind_ns_final"]
    return {"ok": not bad and last_ok, "bad": bad[:6], "last_equals_final": last_ok}


# ------------------------------------------------------------------ server-side counters (stage B)
def rel_counters(obs, heaps: list[int]) -> dict:
    r = obs.execute("SELECT coalesce(sum(heap_blks_read), 0), coalesce(sum(heap_blks_hit), 0), "
                    "coalesce(sum(idx_blks_read), 0), coalesce(sum(idx_blks_hit), 0) "
                    "FROM pg_statio_all_tables WHERE relid = ANY(%s::oid[])", (heaps,)).fetchone()
    return {"heap_read": int(r[0]), "heap_hit": int(r[1]), "idx_read": int(r[2]), "idx_hit": int(r[3])}


def backend_io(obs, pid: int) -> dict:
    r = obs.execute("SELECT coalesce(sum(reads), 0), coalesce(sum(read_bytes), 0), coalesce(sum(read_time), 0), "
                    "coalesce(sum(hits), 0) FROM pg_stat_get_backend_io(%s) WHERE object = 'relation'",
                    (pid,)).fetchone()
    return {"reads": float(r[0]), "read_bytes": float(r[1]), "read_time_ms": float(r[2]), "hits": float(r[3])}


def server_io(obs) -> dict:
    rows = obs.execute("SELECT backend_type, coalesce(sum(reads), 0), coalesce(sum(read_bytes), 0), "
                       "coalesce(sum(read_time), 0) FROM pg_stat_io WHERE object = 'relation' GROUP BY 1").fetchall()
    return {r[0]: {"reads": float(r[1]), "read_bytes": float(r[2]), "read_time_ms": float(r[3])} for r in rows}


def active_others(obs, own: list[int]) -> list[list]:
    rows = obs.execute("SELECT pid, backend_type, left(query, 60) FROM pg_stat_activity WHERE state = 'active' "
                       "AND backend_type = 'client backend' AND pid <> ALL(%s::int[])", (own,)).fetchall()
    return [list(r) for r in rows]


def obs_flush(obs) -> None:
    """Publish the observer's own pending statistics before it reads (R8 Codex 262)."""
    obs.execute("SELECT pg_stat_force_next_flush()")
    obs.execute("SELECT 1")


def evict(obs, oids: list[int], evict_list: list[int]) -> dict:
    """(§2.2) pg_buffercache_evict_relation over an explicit list; done if at most 1% of the buffers remain."""
    before = buffers_of(obs, oids)
    for o in evict_list:
        obs.execute("SELECT * FROM pg_buffercache_evict_relation(%s::oid::regclass)", (o,)).fetchall()
    after = buffers_of(obs, oids)
    return {"before": before, "after": after, "ok": after <= 0.01 * before}


def prewarm(obs, rels: Rels, oids: list[int] | None, what: str, pause_s: float | None = None) -> dict:
    """Sequential read through PostgreSQL (oids), or a pause of pause_s (control, oids None), wrapped in the SAME
    observation steps, so both branches of block 3 differ only in the core (R1 Codex 264). The server-side I/O
    accounting of the read path is checked against the returned block count (R8 Codex 262), with a delayed re-read.
    "s" is the core duration, "total_s" the whole procedure."""
    t_all = time.perf_counter()
    obs_flush(obs)
    obs.execute("SELECT pg_stat_clear_snapshot()")
    s0 = server_io(obs)
    m0 = mem(f"before {what}", bufsnap(obs, rels))
    t = time.perf_counter()
    blocks = 0
    if oids is None:
        time.sleep(pause_s or 0.0)
    else:
        for o in oids:
            blocks += int(obs.execute("SELECT pg_prewarm(%s::oid::regclass, 'read')", (o,)).fetchone()[0])
    dt = time.perf_counter() - t
    obs_flush(obs)
    obs.execute("SELECT pg_stat_clear_snapshot()")
    s1 = server_io(obs)
    time.sleep(1.5)
    obs.execute("SELECT pg_stat_clear_snapshot()")
    s2 = server_io(obs)

    def rb(a: dict, b: dict) -> float:
        return sum(b.get(k, {}).get("read_bytes", 0) - v["read_bytes"] for k, v in a.items())

    acc = rb(s0, s1) / (blocks * BLOCK) if blocks else None
    m1 = mem(f"after {what}", bufsnap(obs, rels))
    return {"what": what, "kind": "pause" if oids is None else "read", "s": dt, "pause_s": pause_s,
            "total_s": time.perf_counter() - t_all, "blocks": blocks, "server_read_bytes": rb(s0, s1),
            "server_read_bytes_late": rb(s0, s2), "accounted_share": acc,
            "accounting": ("не применимо" if oids is None else
                           "согласован" if acc is not None and 0.95 <= acc <= 1.05 else "недоступно или неполно"),
            "mem_before": m0, "mem_after": m1}


# ------------------------------------------------------------------ one series (§2.2, §3.1)
class StageB:
    def __init__(self, ctx: A7.Ctx, model: Model, sizes: dict, mutant: str | None, state: dict) -> None:
        self.ctx, self.model, self.sz, self.mutant, self.state = ctx, model, sizes, mutant, state
        self.obs = G.connect()
        self.obs_pid = int(self.obs.execute("SELECT pg_backend_pid()").fetchone()[0])
        self.rels = Rels(self.obs, ctx)
        self.parent_re = re.compile(rf"\b{re.escape(ctx.tp)}\b")
        self.leaf_re = re.compile(rf"\b{re.escape(ctx.tp)}_p(\d+)\b")
        self.w: list[tuple[bytes, int]] = []
        state.update({"series": [], "events": [], "mem": [], "orders": {}, "at": None})
        self.pub_left = sizes["pub_checks"]

    def target(self, table: str) -> tuple[list[int], list[int], list[int]]:
        """(heaps for counters, relations for eviction checks, heaps of the other table)."""
        r = self.rels
        if table == "s":
            return r.s_heap, r.s_all, r.p_heap
        return r.p_heap, r.p_all, r.s_heap

    def rel(self, table: str, leaf: int) -> str:
        ctx = self.ctx
        if table == "s":
            return ctx.tp if self.mutant == "N2" else ctx.ts
        if self.mutant == "N1":
            return ctx.tp
        j = (leaf + 1) % ctx.nparts if self.mutant == "N4" else leaf
        return f"{ctx.tp}_p{j}"

    def snapshot(self, conn, pid: int, heaps: list[int], other: list[int], journal: list[str]) -> dict:
        if self.mutant != "N7":
            conn.execute("SELECT pg_stat_force_next_flush()", prepare=False)
            journal.append("flush")
        conn.execute("SELECT 1", prepare=False)
        journal.append("select1")
        obs_flush(self.obs)
        journal.append("obs_flush")
        self.obs.execute("SELECT pg_stat_clear_snapshot()")
        journal.append("clear")
        snap = {"rel": rel_counters(self.obs, heaps), "other": rel_counters(self.obs, other),
                "backend": backend_io(self.obs, pid), "observer": backend_io(self.obs, self.obs_pid),
                "server": server_io(self.obs)}
        journal.append("read")
        return snap

    def prep_check(self, table: str, rows: list) -> dict:
        """(R2 Codex 262): after W the series connection holds exactly the expected prepared lookups — one text on s7
        for s, one per leaf (nparts) for d, none on the parent — each executed at least as often as W demands."""
        execs = [g + c for _, g, c in rows]
        if table == "s":
            ok = (len(rows) == 1 and self.ctx.ts in rows[0][0] and not self.parent_re.search(rows[0][0])
                  and execs[0] >= len(self.w))
            return {"ok": ok, "statements": len(rows), "min_exec": min(execs) if execs else 0}
        leaves = {int(m.group(1)) for r in rows if (m := self.leaf_re.search(r[0]))}
        parent = [r[0][:80] for r in rows if self.parent_re.search(r[0])]
        ok = (len(rows) == self.ctx.nparts and len(leaves) == self.ctx.nparts and not parent
              and bool(execs) and min(execs) >= PER_LEAF)
        return {"ok": ok, "statements": len(rows), "leaves": len(leaves), "parent": parent[:3],
                "min_exec": min(execs) if execs else 0}

    def run(self, table: str, keys: list[tuple[bytes, int]], *, evicted: bool, tags: dict) -> dict:
        obs = self.obs
        heaps, all_rel, other = self.target(table)
        self.state["at"] = {**tags, "table": table, "evicted": evicted, "step": "connect"}
        conn = G.connect()
        try:
            conn.prepared_max = 1_000_000
            if self.mutant != "N3":
                conn.execute("SET track_io_timing = on", prepare=False)
            conn.execute("SET plan_cache_mode = auto", prepare=False)
            tio = conn.execute("SHOW track_io_timing", prepare=False).fetchone()[0]
            pid = int(conn.execute("SELECT pg_backend_pid()", prepare=False).fetchone()[0])

            def get(kb: bytes, leaf: int):
                return conn.execute(f"SELECT {A7.COLS} FROM {self.rel(table, leaf)} WHERE k = %s",
                                    (uuid.UUID(bytes=kb),), prepare=True).fetchall()

            self.state["at"]["step"] = "warm-up W"
            for kb, leaf in self.w:  # prepared-statement warm-up, the same keys in the same order for both tables
                get(kb, leaf)
            warm_rows = A7.prepared_stats(conn)
            prep = self.prep_check(table, warm_rows)
            ev = None
            if evicted:
                self.state["at"]["step"] = "evict"
                ev_list = [] if (self.mutant == "N11" and table == "d") else all_rel
                ev = evict(obs, all_rel, ev_list)
            journal: list[str] = []
            self.state["at"]["step"] = "snapshot before"
            before = self.snapshot(conn, pid, heaps, other, journal)
            act_before = active_others(obs, [pid, self.obs_pid])
            self.state["at"]["step"] = "lookups"
            times, errors = [], 0
            for kb, leaf in keys:
                t = time.perf_counter_ns()
                rows = get(kb, leaf)
                times.append((time.perf_counter_ns() - t) / 1000)
                st_ = self.model.state.get(kb)
                if len(rows) != 1 or st_ is None or rows[0][1] != st_[0] or sha16(rows[0][3]) != st_[1]:
                    errors += 1
            self.state["at"]["step"] = "snapshot after"
            after = self.snapshot(conn, pid, heaps, other, journal)
            act_after = active_others(obs, [pid, self.obs_pid])
            end_rows = A7.prepared_stats(conn)
        finally:
            conn.close()
        pub = None
        if evicted and self.pub_left > 0:  # (п3б) publication: unchanged 1.5 s later, nonzero reads
            self.pub_left -= 1
            time.sleep(1.5)
            obs.execute("SELECT pg_stat_clear_snapshot()")
            late = rel_counters(obs, heaps)
            pub = {"late": late, "same": late == after["rel"],
                   "reads": after["rel"]["heap_read"] + after["rel"]["idx_read"]
                   - before["rel"]["heap_read"] - before["rel"]["idx_read"]}
        lat = np.asarray(times)
        rec = {**tags, "table": table, "evicted": evicted, "M": len(keys), "pid": pid, "track_io_timing": tio,
               "errors": errors, "lat": G.pct(times),
               "hist": np.histogram(lat, bins=np.asarray(HIST_EDGES))[0].tolist() if len(lat) else [],
               "before": before, "after": after, "journal": journal, "active": act_before + act_after,
               "prep": prep, "plans_warm": A7.plan_counts(warm_rows), "plans_end": A7.plan_counts(end_rows),
               "evict": ev, "publication": pub}
        self.state["series"].append(rec)
        return rec


# ------------------------------------------------------------------ key selection (§2.2, R7 Codex 254, R4 Codex 262)
def select_keys(live: list[bytes], nparts: int, m: int, nsets: int, mutant: str | None) -> dict:
    arr = np.frombuffer(b"".join(live), np.uint8).reshape(-1, 16)
    leaves = PH.part_of_np(arr, nparts)
    rw, rz = random.Random(SEED_W), random.Random(SEED_Z)
    w_idx, z_idx = [], []
    for j in range(nparts):
        cand = np.flatnonzero(leaves == j).tolist()
        q = PER_LEAF - 1 if (mutant == "N9" and j == 0) else PER_LEAF
        wj = rw.sample(cand, q)
        w_idx += wj
        ws = set(wj)
        z_idx += rz.sample([i for i in cand if i not in ws], PER_LEAF)
    used = set(w_idx) | set(z_idx)
    rs = random.Random(SEED_SETS)
    draw = rs.sample(range(len(live)), nsets * m + len(used))
    flat = [i for i in draw if i not in used][: nsets * m]
    sets = [[(live[i], int(leaves[i])) for i in flat[k * m:(k + 1) * m]] for k in range(nsets)]
    w = [(live[i], int(leaves[i])) for i in sorted(w_idx, key=lambda i: (int(leaves[i]), i))]
    z = [(live[i], int(leaves[i])) for i in sorted(z_idx, key=lambda i: (int(leaves[i]), i))]
    wq = np.bincount([lf for _, lf in w], minlength=nparts)
    zq = np.bincount([lf for _, lf in z], minlength=nparts)
    allset = set(flat)
    quotas = {"w_per_leaf_ok": bool((wq == PER_LEAF).all()), "z_per_leaf_ok": bool((zq == PER_LEAF).all()),
              "disjoint": not (allset & used) and not (set(w_idx) & set(z_idx)), "sets": len(sets),
              "set_sizes": sorted({len(s) for s in sets})}
    quotas["ok"] = quotas["w_per_leaf_ok"] and quotas["z_per_leaf_ok"] and quotas["disjoint"]
    return {"w": w, "z": z, "sets": sets, "quotas": quotas}


def abba(i: int) -> tuple[str, str]:
    return ("s", "d") if i % 4 in (0, 3) else ("d", "s")


# ------------------------------------------------------------------ stage B (§2.2); state is filled as it goes (R7)
def run_stage_b(ctx: A7.Ctx, model: Model, sizes: dict, mutant: str | None, state: dict) -> dict:
    sb = StageB(ctx, model, sizes, mutant, state)
    rels, obs = sb.rels, sb.obs
    try:
        nsets = sizes["b1"] + 2 * sizes["b2"] + 2 * sizes["b3_pairs"]
        state["at"] = {"step": "select keys"}
        sel = select_keys(model.live, ctx.nparts, sizes["M"], nsets, mutant)
        sb.w = sel["w"]
        state["quotas"] = sel["quotas"]
        state["relations"] = {"s": len(rels.s_all), "p": len(rels.p_all), "a7": len(rels.a_all)}
        sets = iter(sel["sets"])
        orders = state["orders"]

        def order(part: str, i: int, glob: int | None = None) -> tuple[str, str]:
            o = ("s", "d") if mutant == "N5" else abba(glob if (mutant == "N10" and glob is not None) else i)
            orders.setdefault(part, []).append(o)
            return o

        def snapmem(where: str) -> None:
            state["mem"].append(mem(where, bufsnap(obs, rels)))

        snapmem("stage B start")
        log("stage B block 1")
        for i in range(sizes["b1"]):
            ks = next(sets)
            x, y = order("b1", i)
            plan = ((x, False, "first"), (y, False, "first"), (x, False, "repeat"), (y, False, "repeat"),
                    (x, True, "evicted"), (y, True, "evicted"))
            for no, (tb, ev, kind) in enumerate(plan, 1):
                sb.run(tb, ks, evicted=ev, tags={"block": "1", "set": i, "no": no, "kind": kind})
        snapmem("block 1 end")
        for half, seq in (("2a", ((rels.s_all, "s7"), (rels.p_all, "p7"))), ("2b", ((rels.p_all, "p7"), (rels.s_all, "s7")))):
            log("stage B block", half)
            snapmem(f"block {half} start")
            for oids, name in seq:
                state["at"] = {"block": half, "step": f"prewarm {name}"}
                state["events"].append({"block": half, "prewarm": prewarm(obs, rels, oids, f"{half} {name}")})
            for i in range(sizes["b2"]):
                ks = next(sets)
                x, y = order(half, i)
                for no, (tb, kind) in enumerate(((x, "first"), (y, "first"), (x, "repeat"), (y, "repeat")), 1):
                    sb.run(tb, ks, evicted=False, tags={"block": half, "set": i, "no": no, "kind": kind})
            snapmem(f"block {half} end")
        log("stage B block 3")
        snapmem("block 3 start")
        last_p = None
        cnt = {"P": 0, "K": 0}
        g = 0
        for pair in range(sizes["b3_pairs"]):
            for br in PAIRS[pair]:
                ks = next(sets)
                x, y = order(f"3{br}", cnt[br], g)
                tags = {"block": "3", "pair": pair, "branch": br, "set": g}
                for tb in (x, y):
                    sb.run(tb, ks, evicted=False, tags={**tags, "kind": "first"})
                for tb in (x, y):
                    sb.run(tb, ks, evicted=True, tags={**tags, "kind": "before"})
                state["at"] = {**tags, "step": "intervention"}
                if br == "P":
                    pw = prewarm(obs, rels, rels.a_all, f"interference pair {pair}")
                    last_p = pw["s"]
                    state["events"].append({"block": "3", "pair": pair, "set": g, "interference": pw})
                else:  # the same procedure, the core is a pause as long as the last read
                    pw = prewarm(obs, rels, None, f"control pair {pair}", pause_s=last_p)
                    state["events"].append({"block": "3", "pair": pair, "set": g, "control": pw})
                for tb in (x, y):
                    sb.run(tb, ks, evicted=True, tags={**tags, "kind": "after"})
                cnt[br] += 1
                g += 1
        snapmem("block 3 end")
        log("stage B block 4")
        for p in range(sizes["z_passes"]):
            x, y = order("4", p)
            for tb in (x, y):
                sb.run(tb, sel["z"], evicted=False, tags={"block": "4", "pass": p, "kind": "pass"})
        snapmem("stage B end")
        state["at"] = {"step": "done"}
        return state
    finally:
        obs.close()


# ------------------------------------------------------------------ derived values (§3.1, §3.3); pure, self-tested
def delta(a: dict, b: dict) -> dict:
    return {k: b[k] - a[k] for k in a}


def derive(rec: dict, mode: str, mutant: str | None = None) -> dict:
    m = rec["M"]
    r = delta(rec["before"]["rel"], rec["after"]["rel"])
    o = delta(rec["before"]["other"], rec["after"]["other"])
    be = delta(rec["before"]["backend"], rec["after"]["backend"])
    ob = delta(rec["before"]["observer"], rec["after"]["observer"])
    s0, s1 = rec["before"]["server"], rec["after"]["server"]
    srv_rb = sum(s1.get(k, {}).get("read_bytes", 0) - v["read_bytes"] for k, v in s0.items() if k != "io worker")
    foreign_rb = srv_rb - be["read_bytes"] - ob["read_bytes"]
    rel_blocks = r["heap_read"] + r["idx_read"]
    be_blocks = be["read_bytes"] / BLOCK
    rel_ok = not rec["active"]
    if mode == SUPPORTED and rel_blocks > be_blocks * 1.05:
        rel_ok = False
    inconsistent = foreign_rb < -BLOCK  # negative foreign bytes: inconsistent accounting, not a sign of purity (R8)
    if mode != SUPPORTED or inconsistent:
        io_ok = False
    elif be["read_bytes"] > 0:
        io_ok = foreign_rb <= 0.05 * be["read_bytes"]
    else:
        io_ok = foreign_rb <= 0
    act_t = r["heap_read"] + r["heap_hit"] + r["idx_read"] + r["idx_hit"]
    act_o = o["heap_read"] + o["heap_hit"] + o["idx_read"] + o["idx_hit"]
    ev = rec.get("evict")
    procedure = {"track_io_timing": rec.get("track_io_timing") == "on", "prepared": bool(rec.get("prep", {}).get("ok")),
                 "activity": act_t > 0 and act_o == 0, "evict": (ev is None) or bool(ev.get("ok"))}
    out = {"correct": rec["errors"] == 0, "procedure_ok": all(procedure.values()), "procedure": procedure,
           "rel_ok": rel_ok, "io_ok": io_ok, "io_inconsistent": inconsistent,
           "activity_target": act_t, "activity_other": act_o,
           "rel_blocks": rel_blocks, "be_blocks": be_blocks, "foreign_read_bytes": foreign_rb,
           "L_mean": rec["lat"]["mean"], "L_p50": rec["lat"]["p50"]}
    out["rho_heap"] = r["heap_read"] / m if rel_ok else None
    out["rho_idx"] = r["idx_read"] / m if rel_ok else None
    out["rho"] = rel_blocks / m if rel_ok else None
    if io_ok:
        denom = be["reads"] if mutant == "N6" else be["read_bytes"] / BLOCK
        out["tau_us"] = 1000 * be["read_time_ms"] / denom if denom > 0 else None
        out["I_us"] = 1000 * be["read_time_ms"] / m
        out["op_bytes"] = be["read_bytes"] / be["reads"] if be["reads"] > 0 else None
    else:
        out["tau_us"] = out["I_us"] = out["op_bytes"] = None
    out["valid"] = out["correct"] and not (mutant == "N8" and out["rho"] == 0)
    return out


def ratio(a, b):
    return A7.div(a, b)


def describe(values: list) -> str:
    v = [x for x in values if x is not None]
    if not v:
        return "не определено"
    if all(x >= THR for x in v):
        d = "выше во всех"
    elif all(x <= 1 / THR for x in v):
        d = "ниже во всех"
    elif all(1 / THR < x < THR for x in v):
        d = "примерно равно во всех"
    else:
        d = "смешано"
    return d if len(v) == len(values) else f"{d} (не определено: {len(values) - len(v)})"


def coverage(ds: dict, dd: dict) -> dict:
    """Д2 (R6 Codex 256): only for Δ = L_s − L_d > 0, defined I and I_s − I_d > 0."""
    dl = ds["L_mean"] - dd["L_mean"]
    di = None if ds["I_us"] is None or dd["I_us"] is None else ds["I_us"] - dd["I_us"]
    cov = di / dl if (dl > 0 and di is not None and di > 0) else None
    return {"delta_L": dl, "delta_I": di, "coverage": cov}


def pair_metrics(ds: dict, dd: dict) -> dict:
    out = {"G": ratio(ds["L_mean"], dd["L_mean"]), "G_p50": ratio(ds["L_p50"], dd["L_p50"]),
           "rho_ratio": ratio(ds["rho"], dd["rho"]), "tau_ratio": ratio(ds["tau_us"], dd["tau_us"]),
           "rho_heap": [ds["rho_heap"], dd["rho_heap"]], "rho_idx": [ds["rho_idx"], dd["rho_idx"]],
           "L": [ds["L_mean"], dd["L_mean"]], "tau": [ds["tau_us"], dd["tau_us"]]}
    out.update(coverage(ds, dd))
    return out


def summarize_list(rows: list[dict], keys=("G", "G_p50", "rho_ratio", "tau_ratio")) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        v = [x for x in vals if x is not None]
        out[k] = {"values": vals, "min": min(v) if v else None, "max": max(v) if v else None, "describe": describe(vals)}
    cov = [r.get("coverage") for r in rows]
    out["coverage"] = {"values": cov, "n_defined": sum(c is not None for c in cov),
                       "ge_half": sum(1 for c in cov if c is not None and c >= 0.5)}
    return out


def block_status(ders: list[dict], mode: str) -> dict:
    """(R3 Codex 262, protocol §4): any incorrect or procedure-violating series -> the block is not established;
    more than 25% contaminated attributed I/O in supported mode -> τ and I of the block are not used. Unsupported
    attribution is not a failure."""
    n = len(ders)
    incorrect = sum(not d["correct"] for d in ders)
    procedure = sum(not d["procedure_ok"] for d in ders)
    contaminated = sum(not d["io_ok"] for d in ders) if mode == SUPPORTED else 0
    st_ = {"n": n, "incorrect": incorrect, "procedure_violations": procedure, "io_contaminated": contaminated,
           "rel_not_ok": sum(not d["rel_ok"] for d in ders)}
    st_["established"] = n > 0 and incorrect == 0 and procedure == 0
    st_["io_usable"] = mode == SUPPORTED and n > 0 and contaminated <= CONTAM_MAX * n
    return st_


def strip_io(d: dict) -> dict:
    return {**d, "tau_us": None, "I_us": None, "op_bytes": None}


def analyze(series: list[dict], mode: str, mutant: str | None = None) -> dict:
    der = [derive(s, mode, mutant) for s in series]
    blocks = sorted({s["block"] for s in series})
    status = {b: block_status([d for s, d in zip(series, der) if s["block"] == b], mode) for b in blocks}
    idx: dict[tuple, dict] = {}
    for s, d in zip(series, der):
        st_ = status[s["block"]]
        dd = d if st_["io_usable"] else strip_io(d)
        idx[(s["block"], s.get("pair"), s.get("branch"), s.get("set"), s.get("pass"), s["kind"], s["table"])] = dd

    def get(block, kind, table, **kw):
        return idx.get((block, kw.get("pair"), kw.get("branch"), kw.get("set"), kw.get("pass"), kind, table))

    out: dict = {"mode": mode, "block_status": status}
    for block in ("1", "2a", "2b"):
        if block not in status:
            continue
        if not status[block]["established"]:
            out[f"b{block}"] = NOT_SET
            continue
        sets = sorted({s["set"] for s in series if s["block"] == block})
        for kind in ("first", "repeat", "evicted"):
            rows = [pair_metrics(ds, dd) for i in sets
                    if (ds := get(block, kind, "s", set=i)) and (dd := get(block, kind, "d", set=i))]
            if rows:
                out[f"b{block}_{kind}"] = {"pairs": rows, "summary": summarize_list(rows)}
    if "3" in status:
        if not status["3"]["established"]:
            out["b3"] = NOT_SET
        else:
            out["b3"] = block3(series, get)
    if "4" in status:
        if not status["4"]["established"]:
            out["b4"] = NOT_SET
        else:
            b4 = []
            for p in sorted({s["pass"] for s in series if s["block"] == "4"}):
                ds, dd = get("4", "pass", "s", **{"pass": p}), get("4", "pass", "d", **{"pass": p})
                if ds and dd:
                    b4.append({"pass": p, **pair_metrics(ds, dd), "rho_s": ds["rho"], "rho_d": dd["rho"]})
            out["b4"] = {"passes": b4, "last_rho_zero": bool(b4) and b4[-1]["rho_s"] == 0 and b4[-1]["rho_d"] == 0,
                         "last_G": b4[-1]["G"] if b4 else None, "describe_G": describe([b["G"] for b in b4])}
    out["derived"] = der
    return out


def block3(series: list[dict], get) -> dict:
    """Д4: R_t = after / before of one set, E_t = R_t(interference) / R_t(control) of one pair."""
    b3 = [s for s in series if s["block"] == "3"]
    r_by_set: dict[int, dict] = {}
    for g_ in sorted({s["set"] for s in b3}):
        one = next(s for s in b3 if s["set"] == g_)
        rec = {"pair": one["pair"], "branch": one["branch"]}
        for t in ("s", "d"):
            bf = get("3", "before", t, pair=one["pair"], branch=one["branch"], set=g_)
            af = get("3", "after", t, pair=one["pair"], branch=one["branch"], set=g_)
            rec[f"R_{t}"] = ratio(af["L_mean"], bf["L_mean"]) if bf and af else None
            rec[f"Rtau_{t}"] = ratio(af["tau_us"], bf["tau_us"]) if bf and af else None
            rec[f"dL_{t}"] = af["L_mean"] - bf["L_mean"] if bf and af else None
            rec[f"drho_{t}"] = (af["rho"] - bf["rho"]) if bf and af and af["rho"] is not None and bf["rho"] is not None else None
        r_by_set[g_] = rec
    pairs = []
    for p in sorted({r["pair"] for r in r_by_set.values()}):
        pr = {r["branch"]: r for r in r_by_set.values() if r["pair"] == p}
        if "P" in pr and "K" in pr:
            e = {"pair": p, "E_s": ratio(pr["P"]["R_s"], pr["K"]["R_s"]), "E_d": ratio(pr["P"]["R_d"], pr["K"]["R_d"]),
                 "Etau_s": ratio(pr["P"]["Rtau_s"], pr["K"]["Rtau_s"]),
                 "Etau_d": ratio(pr["P"]["Rtau_d"], pr["K"]["Rtau_d"])}
            e["E_s_over_E_d"] = ratio(e["E_s"], e["E_d"])
            pairs.append(e)
    return {"sets": r_by_set, "pairs": pairs,
            "summary": {k: {"values": [p[k] for p in pairs], "describe": describe([p[k] for p in pairs])}
                        for k in ("E_s", "E_d", "E_s_over_E_d", "Etau_s", "Etau_d")}}


# ------------------------------------------------------------------ calculation selftest (§4.1)
def _snap(hr, hh, ir, ih, reads, rbytes, rtime, srv_extra=0.0):
    return {"rel": {"heap_read": hr, "heap_hit": hh, "idx_read": ir, "idx_hit": ih},
            "other": {"heap_read": 0, "heap_hit": 0, "idx_read": 0, "idx_hit": 0},
            "backend": {"reads": reads, "read_bytes": rbytes, "read_time_ms": rtime, "hits": 0.0},
            "observer": {"reads": 0.0, "read_bytes": 0.0, "read_time_ms": 0.0, "hits": 0.0},
            "server": {"client backend": {"reads": reads, "read_bytes": rbytes + srv_extra, "read_time_ms": rtime},
                       "io worker": {"reads": 0.0, "read_bytes": 0.0, "read_time_ms": 0.0}}}


_ZERO = _snap(0, 0, 0, 0, 0.0, 0.0, 0.0)
_OK_PROC = {"track_io_timing": "on", "prep": {"ok": True}, "evict": None}


def _rec(after, mean, p50, active=(), errors=0, **tags):
    return {"M": 100, "errors": errors, "lat": {"mean": mean, "p50": p50}, "before": _ZERO, "after": after,
            "active": list(active), **_OK_PROC, **tags}


def calc_selftest(mutant: str | None = None) -> dict:
    """Fixed artificial series: G, ρ, τ (several blocks per I/O op), I, coverage, E over two pairs, describe, zero reads,
    contaminated and inconsistent accounting, block rules, Д1 over three checkpoints."""
    s_ = _rec(_snap(200, 100, 100, 300, 150.0, 300.0 * BLOCK, 30.0), 400.0, 380.0)
    d_ = _rec(_snap(100, 200, 50, 350, 150.0, 150.0 * BLOCK, 6.0), 200.0, 190.0)
    ds, dd = derive(s_, SUPPORTED, mutant), derive(d_, SUPPORTED, mutant)
    pm = pair_metrics(ds, dd)
    z = derive(_rec(_snap(0, 50, 0, 50, 0.0, 0.0, 0.0), 50.0, 45.0), SUPPORTED, mutant)
    dirty = derive(_rec(_snap(10, 0, 0, 0, 10.0, 10.0 * BLOCK, 1.0, srv_extra=5.0 * BLOCK), 60.0, 55.0), SUPPORTED, mutant)
    neg = derive(_rec(_snap(10, 0, 0, 0, 10.0, 10.0 * BLOCK, 1.0, srv_extra=-5.0 * BLOCK), 60.0, 55.0), SUPPORTED, mutant)
    act = derive(_rec(_snap(10, 0, 0, 0, 10.0, 10.0 * BLOCK, 1.0), 60.0, 55.0, active=[[1, "client backend", "x"]]),
                 SUPPORTED, mutant)
    uns = derive(s_, UNSUPPORTED, mutant)

    # block 3: two pairs, latency before/after chosen so that R and E are known exactly
    def b3rec(pair, br, setno, kind, table, mean):
        return _rec(_snap(10, 10, 0, 10, 10.0, 10.0 * BLOCK, 1.0), mean, mean, block="3", pair=pair, branch=br,
                    set=setno, kind=kind, table=table)

    b3 = []
    plan = [(0, "P", 0, 100.0, 150.0, 100.0, 110.0), (0, "K", 1, 100.0, 100.0, 100.0, 100.0),
            (1, "K", 2, 200.0, 220.0, 50.0, 50.0), (1, "P", 3, 200.0, 440.0, 50.0, 100.0)]
    for pair, br, setno, sb_, sa_, db_, da_ in plan:
        b3 += [b3rec(pair, br, setno, "before", "s", sb_), b3rec(pair, br, setno, "after", "s", sa_),
               b3rec(pair, br, setno, "before", "d", db_), b3rec(pair, br, setno, "after", "d", da_)]
    an = analyze(b3, SUPPORTED, mutant)
    e = an["b3"]["pairs"] if isinstance(an["b3"], dict) else []
    # E_s pair0 = 1.5 / 1.0 = 1.5; E_d pair0 = 1.1; E_s pair1 = 2.2 / 1.1 = 2.0; E_d pair1 = 2.0 / 1.0 = 2.0
    e_ok = (len(e) == 2 and abs(e[0]["E_s"] - 1.5) < 1e-12 and abs(e[0]["E_d"] - 1.1) < 1e-12
            and abs(e[1]["E_s"] - 2.0) < 1e-12 and abs(e[1]["E_d"] - 2.0) < 1e-12
            and an["b3"]["summary"]["E_s"]["describe"] == "выше во всех")
    bad_block = analyze(b3[:3] + [{**b3[3], "errors": 1}] + b3[4:], SUPPORTED, mutant)
    # Д1 over three artificial checkpoints: s 100 → 300 µs per U+D lookup between cp1 and cp2, dir 100 µs
    def kn(su, sd, du, dd_):
        return {"s": {"I": [0, 0, 0, 0, 0], "U": [su * 1000, 0, 10, 0, 0], "D": [sd * 1000, 0, 10, 0, 0]},
                "dir": {"I": [0, 0, 0, 0, 0], "U": [du * 1000, 0, 5, 0, 0], "D": [dd_ * 1000, 0, 5, 0, 0]},
                "par": {"I": [0, 0, 0, 0, 0], "U": [0, 0, 0, 0, 0], "D": [0, 0, 0, 0, 0]}}

    def cp(k, i, s50, p50):
        out = {"a8": {"kind_ns": k}, "after_op": i, "routes": {"s": {"client_us": {"p50": s50}},
                                                                "par": {"client_us": {"p50": p50}}}}
        if i == 0:
            out["a8"]["kind_ns"] = kn(0, 0, 0, 0)
        return out

    k1 = kn(1000, 1000, 500, 500)
    k2 = {"s": {"I": [0, 0, 0, 0, 0], "U": [7000 * 1000, 0, 30, 0, 0], "D": [7000 * 1000, 0, 30, 0, 0]},
          "dir": {"I": [0, 0, 0, 0, 0], "U": [2500 * 1000, 0, 25, 0, 0], "D": [2500 * 1000, 0, 25, 0, 0]},
          "par": k1["par"]}
    d1 = d1_from_checkpoints([cp(None, 0, 1, 1), cp(k1, 100, 300.0, 100.0), cp(k2, 200, 200.0, 180.0)])
    checks = {
        "rho": abs(ds["rho"] - 3.0) < 1e-12 and abs(dd["rho"] - 1.5) < 1e-12,
        "rho_parts": ds["rho_heap"] == 2.0 and ds["rho_idx"] == 1.0,
        "tau_per_block": ds["tau_us"] is not None and abs(ds["tau_us"] - 100.0) < 1e-9 and abs(dd["tau_us"] - 40.0) < 1e-9,
        "I": abs(ds["I_us"] - 300.0) < 1e-9 and abs(dd["I_us"] - 60.0) < 1e-9,
        "op_bytes": abs(ds["op_bytes"] - 2 * BLOCK) < 1e-9,
        "G": abs(pm["G"] - 2.0) < 1e-12 and abs(pm["G_p50"] - 2.0) < 1e-12,
        "coverage": pm["coverage"] is not None and abs(pm["coverage"] - 1.2) < 1e-12,
        "coverage_sign": coverage(dd, ds)["coverage"] is None,
        "zero_reads": z["valid"] and z["rho"] == 0 and z["tau_us"] is None and z["I_us"] == 0,
        "dirty_io": dirty["io_ok"] is False and dirty["tau_us"] is None and dirty["rho"] is not None,
        "negative_foreign": neg["io_ok"] is False and neg["io_inconsistent"] is True,
        "active_rel": act["rel_ok"] is False and act["rho"] is None,
        "unsupported": uns["tau_us"] is None and uns["I_us"] is None and uns["rho"] == 3.0,
        "E_pairs": e_ok,
        "incorrect_block": bad_block["b3"] == NOT_SET,
        "d1": (d1["defined"] and abs(d1["lookup_ud_us"]["s"] - 300.0) < 1e-9
               and abs(d1["lookup_ud_us"]["dir"] - 100.0) < 1e-9 and abs(d1["a_ratio"] - 3.0) < 1e-12
               and d1["b_ratios"] == [3.0, 200.0 / 180.0] and d1["repeated"] is True),
        "describe": (describe([1.4, 2.0]) == "выше во всех" and describe([0.7, 0.5]) == "ниже во всех"
                     and describe([1.0, 1.29]) == "примерно равно во всех" and describe([1.0, 1.5]) == "смешано"
                     and describe([None, None]) == "не определено"),
        "ratio_none": ratio(1.0, 0) is None and ratio(None, 1.0) is None,
    }
    return {"ok": all(checks.values()), "checks": checks}


# ------------------------------------------------------------------ checks of a stage B result (§4, §4.1)
def balanced(orders: list) -> bool:
    n_sd = sum(1 for o in orders if tuple(o) == ("s", "d"))
    return abs(n_sd - (len(orders) - n_sd)) <= (len(orders) % 2)


def attribution_mode(series: list[dict]) -> dict:
    """(п3а) property of the stand: on evicted series the process of the series has read_bytes and read_time, and its
    blocks match the relation counters within 5%."""
    rows = []
    for s in (x for x in series if x["evicted"]):
        r = delta(s["before"]["rel"], s["after"]["rel"])
        be = delta(s["before"]["backend"], s["after"]["backend"])
        rel_b = r["heap_read"] + r["idx_read"]
        be_b = be["read_bytes"] / BLOCK
        rows.append({"rel_blocks": rel_b, "be_blocks": be_b, "be_read_time_ms": be["read_time_ms"],
                     "ok": be["read_bytes"] > 0 and be["read_time_ms"] > 0 and rel_b > 0
                     and abs(be_b - rel_b) <= 0.05 * rel_b})
    sup = bool(rows) and all(r["ok"] for r in rows)
    return {"mode": SUPPORTED if sup else UNSUPPORTED, "evicted_series": len(rows),
            "ok_share": sum(r["ok"] for r in rows) / len(rows) if rows else None, "rows": rows[:12]}


def stage_b_checks(b: dict, mode: str, mutant: str | None = None) -> dict:
    ser = b["series"]
    der = [derive(s, mode, mutant) for s in ser]
    evs = [s for s in ser if s["evicted"]]
    pubs = [s["publication"] for s in ser if s.get("publication")]
    zero_valid = [d for d in der if d["rho"] == 0 and d["rel_ok"]]
    return {
        "p1_correct": bool(der) and all(d["correct"] for d in der),
        "p1_track_io_timing": bool(ser) and all(s["track_io_timing"] == "on" for s in ser),
        "p1_prepared_rule": bool(ser) and all(s["prep"]["ok"] for s in ser),
        "p1_activity": bool(der) and all(d["procedure"]["activity"] for d in der),
        "p1_quotas": b["quotas"]["ok"],
        "p2_zero_read_valid": bool(zero_valid) and all(d["valid"] for d in zero_valid),
        "p2_evicted_reads": bool(evs) and all(derive(s, mode)["rel_blocks"] > 0 for s in evs),
        "p3b_journal": bool(ser) and all(s["journal"] == JOURNAL + JOURNAL for s in ser),
        "p3b_publication": bool(pubs) and all(p["same"] and p["reads"] > 0 for p in pubs),
        "p4_abba": bool(b["orders"]) and all(balanced(v) for v in b["orders"].values()),
        "p6_memory": bool(b["mem"]) and all(m.get("avail_phys_gb") is not None and m.get("system_cache_gb") is not None
                                            and isinstance(m.get("pg_buffers"), dict) for m in b["mem"]),
        "p6_evict": bool(evs) and all(s["evict"]["ok"] for s in evs),
    }


DESIGNATED = {"N1": "p1_prepared_rule", "N2": "p1_activity", "N3": "p1_track_io_timing", "N4": "p1_correct",
              "N5": "p4_abba", "N7": "p3b_journal", "N8": "p2_zero_read_valid", "N9": "p1_quotas",
              "N10": "p4_abba", "N11": "p6_evict"}


# ------------------------------------------------------------------ extensions, preconditions, cleanup
def ensure_extensions(conn, ext: dict) -> dict:
    """Each extension created here is journaled in ext["created"] right after its CREATE, so a later failure still
    lets the cleanup drop it (R3 Codex 266)."""
    have = {r[0] for r in conn.execute("SELECT extname FROM pg_extension").fetchall()}
    for e in ("pg_buffercache", "pg_prewarm"):
        if e not in have:
            conn.execute(f"CREATE EXTENSION {e}")
            ext["created"].append(e)
    ext["pg_buffercache_version"] = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'pg_buffercache'").fetchone()[0]
    return ext


def tables_absent(conn, ctx: A7.Ctx) -> bool:
    return all(conn.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0] is None for t in (ctx.ts, ctx.tp, ctx.ta))


def preconditions(conn) -> dict:
    """block_size must be the BLOCK used in the conversions; the default track_io_timing must be off so that N3 can
    break it (R5, N3 Codex 262)."""
    bs = int(conn.execute("SHOW block_size").fetchone()[0])
    tio = conn.execute("SHOW track_io_timing").fetchone()[0]
    return {"block_size": bs, "block_size_ok": bs == BLOCK, "track_io_timing_default": tio,
            "track_io_timing_default_off": tio == "off"}


def cleanup(ctx: A7.Ctx | None, a, created_tables: bool, created_ext: list[str]) -> list[str]:
    """Close every resource independently; drop only the tables this run created (R1 Codex 262); never raise."""
    errs: list[str] = []

    def attempt(what: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errs.append(f"{what}: {type(e).__name__}: {str(e)[:150]}")

    if ctx is not None:
        for r, c in getattr(ctx, "work", {}).items():
            attempt(f"close {r}", c.conn.close)
        if getattr(ctx, "loop_conn", None) is not None:
            attempt("close loop connection", ctx.loop_conn.close)
    if a is not None:
        attempt("close address", a.lsm.close)
        attempt("close a7 connection", a.conn.close)
    try:
        c = G.connect()
    except Exception as e:  # noqa: BLE001
        return errs + [f"connect for cleanup: {type(e).__name__}: {str(e)[:150]}"]
    if created_tables and ctx is not None:
        for t in (ctx.ts, ctx.tp, ctx.ta):
            attempt(f"drop {t}", lambda t=t: c.execute(f"DROP TABLE IF EXISTS {t} CASCADE"))
    for e in created_ext:
        attempt(f"drop extension {e}", lambda e=e: c.execute(f"DROP EXTENSION IF EXISTS {e}"))
    attempt("close cleanup connection", c.close)
    return errs


# ------------------------------------------------------------------ stage A (§2.1)
def new_ctx(tag: str, bits: int) -> A7.Ctx:
    ctx = A7.Ctx(tag, 1 << bits, None)
    ctx.work = {}
    return ctx


def run_stage_a(ctx: A7.Ctx, *, n: int, bits: int, t_run: int, ref: Path, d: Path, size: str,
                stage_a_ops: int | None, cps: tuple, out: dict, conn) -> Model:
    out["tables_absent_before"] = tables_absent(conn, ctx)
    if not out["tables_absent_before"]:
        raise A7.Stop("tables with the A8 names exist before loading; nothing is dropped")
    out["created_tables"] = True  # from here on the tables are ours
    out["stage"] = "load"
    info, a = A7.load_three(ctx.tag, n, ref, bits, d, t_run)
    out["_a"] = a
    out["load"] = info
    log(size, "load", {k: v for k, v in info.items() if k != "pages"}, info["pages"])
    keys = [bytes(k) for k in np.load(f"{ref}.keys")]
    digs = [bytes(x) for x in np.load(f"{ref}.dig")]
    out["ref_lengths"] = [len(keys), len(digs)]
    if any(v != n for v in info["rows"].values()) or out["ref_lengths"] != [n, n]:
        raise A7.Stop("row counts or reference lengths differ from N")
    extra = load_extra(DATA / "dyn.copy")
    if stage_a_ops is None:  # small selfcheck stream (as the A7 small run)
        ops = make_ops(keys, extra, 10_000, 6_000, 4_000, 13)
    else:
        full = make_ops(keys, extra, 250_000, 150_000, 100_000, 13)
        out["ops_sha256_full"] = ops_sha256(full)
        out["stream_matches_a4"] = out["ops_sha256_full"].startswith(A6.STREAM_SHA[size])
        if not out["stream_matches_a4"]:
            raise A7.Stop("stream differs from A4")
        ops = full[:stage_a_ops]
        out["ops_sha256_prefix"] = ops_sha256(ops)
    out["stage"] = "function_check"
    ins_keys = np.frombuffer(b"".join(kb for kind, kb, _ in ops if kind == "I"), np.uint8).reshape(-1, 16)
    fc = A6.function_check(conn, ctx.ta, a.lsm, ins_keys, None)
    out["function_check"] = fc
    if not fc["ok"]:
        raise A7.Stop("partition function check failed")
    out["schedule_prefix_ok"] = schedule_prefix_ok(len(ops) // A7.BATCH)
    ever = np.concatenate([np.load(f"{ref}.keys"), np.load(DATA / "dyn.keys")])
    model = Model(keys, digs)
    del keys, digs
    out["stage"] = "stage A"
    install_wrapper()
    loop: dict = {}
    out["_loop"] = loop
    A7.run_loop(ctx, a, ops, model, ever, size=size, vacuums=(), cps=cps, stop_on_error=True, diag=False, out=loop)
    out["stage_a"] = {k: v for k, v in loop.items() if k != "schedule"}
    out["stage_a"]["kind_ns_final"] = {r: copy.deepcopy(c.kind_ns) for r, c in ctx.work.items()}
    out["d1"] = d1_from_checkpoints(loop["checkpoints"])
    log(size, "D1", out["d1"])
    return model


def stage_a_ok(stage_a: dict) -> bool:
    return (not any(stage_a["apply_errors"].values()) and stage_a["witness_errors"] == 0
            and all(c["errors_total"] == 0 and not c["prepared_rule_violations"] for c in stage_a["checkpoints"])
            and not stage_a["timer_violations"])


BUF_KEYS = ("s7_heap", "s7_idx", "p7_heap", "p7_idx", "a7_heap", "a7_idx")


def c7_observations_ok(stage_a: dict) -> dict:
    """(R3 Codex 264): every C7 of stage A has memory values and the six numeric buffer counters."""
    bad = []
    for c in stage_a.get("checkpoints") or []:
        m = (c.get("a8") or {}).get("mem") or {}
        bufs = m.get("pg_buffers")
        ok = (m.get("avail_phys_gb") is not None and m.get("system_cache_gb") is not None and isinstance(bufs, dict)
              and all(isinstance(bufs.get(k), int) for k in BUF_KEYS))
        if not ok:
            bad.append({"cp": c.get("cp"), "mem": {k: m.get(k) for k in ("avail_phys_gb", "system_cache_gb")},
                        "pg_buffers": bufs})
    n = len(stage_a.get("checkpoints") or [])
    return {"ok": n > 0 and not bad, "checkpoints": n, "bad": bad}


def public(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def safe_write(path: Path, obj: dict) -> str | None:
    """Writing the result never prevents the cleanup and never replaces the original cause (R2 Codex 264)."""
    try:
        G.write_json(path, obj)
        return None
    except Exception as e:  # noqa: BLE001
        return f"write {path.name}: {type(e).__name__}: {str(e)[:150]}"


def close_quietly(conn) -> str | None:
    try:
        conn.close()
        return None
    except Exception as e:  # noqa: BLE001
        return f"close main connection: {type(e).__name__}"


# ------------------------------------------------------------------ selfcheck (§4.1)
def selfcheck_body(out: dict, res: dict, b: dict, ctx: A7.Ctx, conn) -> None:
    if not out["preconditions"]["block_size_ok"]:
        raise A7.Stop("block_size differs from 8192")
    model = run_stage_a(ctx, n=100_000, bits=3, t_run=3_000, ref=DATA / "ref_a8_small", d=DATA / "a8_small",
                        size="small", stage_a_ops=None, cps=(0, 10_000, 20_000), out=res, conn=conn)
    out["stage_a_ok"] = stage_a_ok(res["stage_a"])
    out["p5_schedule_prefix"] = res["schedule_prefix_ok"] and schedule_prefix_ok(200)
    out["p7_wrapper"] = wrapper_ok(res["stage_a"], 3)
    out["p6_c7_observations"] = c7_observations_ok(res["stage_a"])
    out["d1_small"] = res["d1"]
    run_stage_b(ctx, model, SMALL, None, b)
    out["p3a_attribution"] = attribution_mode(b["series"])
    mode = out["p3a_attribution"]["mode"]
    out["checks"] = stage_b_checks(b, mode)
    out["analysis_small"] = {k: v for k, v in analyze(b["series"], mode).items() if k != "derived"}
    out["interventions_small"] = [e.get("prewarm") or e.get("interference") or e.get("control") for e in b["events"]]
    log("normal checks", out["checks"], "attribution", mode)
    out["mutants"] = {}
    for m in MUTANTS:
        if m == "N6":
            out["mutants"][m] = {"caught": out["calc_selftest_N6"]["ok"] is False, "check": "calc_selftest"}
        elif m == "N8":
            ch = stage_b_checks(b, mode, "N8")
            out["mutants"][m] = {"caught": ch["p2_zero_read_valid"] is False, "check": DESIGNATED[m]}
        else:
            bm: dict = {}
            try:
                run_stage_b(ctx, model, SMALL, m, bm)
                ch = stage_b_checks(bm, mode)
                out["mutants"][m] = {"caught": ch[DESIGNATED[m]] is False, "check": DESIGNATED[m], "checks": ch}
            except Exception as e:  # noqa: BLE001  a crash is not a catch (positive evidence only)
                out["mutants"][m] = {"caught": False, "exception": f"{type(e).__name__}: {str(e)[:200]}",
                                     "at": bm.get("at"), "series_done": len(bm.get("series") or [])}
        log("mutant", m, "caught" if out["mutants"][m]["caught"] else "NOT CAUGHT", out["mutants"][m].get("exception"))
    out["all_ok"] = (out["calc_selftest"]["ok"] and out["preconditions"]["track_io_timing_default_off"]
                     and out["stage_a_ok"] and out["p5_schedule_prefix"] and out["p7_wrapper"]["ok"]
                     and out["p6_c7_observations"]["ok"] and all(out["checks"].values())
                     and all(v["caught"] for v in out["mutants"].values()))


def cmd_selfcheck() -> None:
    check_protocol()
    A6.require_selfcheck()
    A7.require_selfcheck()
    out: dict = {"impl_sha256": impl_hashes(), "calc_selftest": calc_selftest(), "all_ok": False}
    out["calc_selftest_N6"] = calc_selftest("N6")
    log("calc selftest", out["calc_selftest"]["ok"], "N6 variant ok:", out["calc_selftest_N6"]["ok"])
    conn = G.connect()
    ctx = new_ctx("s8", 3)
    res: dict = {}
    b: dict = {}
    ext = {"created": []}
    try:
        out["preconditions"] = preconditions(conn)
        ensure_extensions(conn, ext)
        out["extensions"] = ext
        selfcheck_body(out, res, b, ctx, conn)
    except BaseException as e:  # keep what was gathered, including the partial stage B (R2 Codex 264)
        out["status"] = f"ОСТАНОВЛЕН: {type(e).__name__}: {str(e)[:300]}"
        out["stage"] = res.get("stage")
        out["all_ok"] = False
        if b:
            out["stage_b_partial"] = {"at": b.get("at"), "n_series": len(b.get("series") or []),
                                      "series": b.get("series"), "events": b.get("events"), "orders": b.get("orders")}
    finally:
        out["setup"] = {k: v for k, v in public(res).items() if k in ("tables_absent_before", "load", "function_check")}
        out["cleanup_errors"] = cleanup(ctx, res.get("_a"), bool(res.get("created_tables")), ext["created"])
        if (err := close_quietly(conn)) is not None:
            out["cleanup_errors"].append(err)
        werr = safe_write(RES / "a8_selfcheck.json", out)
        log("selfcheck all_ok", out["all_ok"], out.get("status", ""), werr or "")
    if not out["all_ok"] or werr is not None:
        raise SystemExit(1)


def require_selfcheck() -> dict:
    f = RES / "a8_selfcheck.json"
    if not f.exists():
        raise SystemExit("no a8_selfcheck.json")
    sc = json.loads(f.read_text(encoding="utf-8"))
    if not sc.get("all_ok") or sc.get("impl_sha256") != impl_hashes():
        raise SystemExit("A8 selfcheck failed or the implementation changed since the selfcheck")
    return sc


# ------------------------------------------------------------------ main run (§2)
def run_body(out: dict, res: dict, b: dict, ctx: A7.Ctx, n: int, bits: int, conn) -> None:
    model = run_stage_a(ctx, n=n, bits=bits, t_run=A6.T_RUN, ref=DATA / "ref_a7_n10", d=DATA / "a8_n10",
                        size="N10", stage_a_ops=STAGE_A_OPS, cps=CPS_MAIN, out=res, conn=conn)
    out.update(public(res))
    out["stage_a_ok"] = stage_a_ok(res["stage_a"])
    out["c7_observations"] = c7_observations_ok(res["stage_a"])
    if not out["stage_a_ok"]:
        raise A7.Stop("stage A correctness violated")
    out["stage"] = "stage B"
    run_stage_b(ctx, model, MAIN, None, b)
    out["attribution_main"] = attribution_mode(b["series"])
    mode = SUPPORTED if (out["attribution_selfcheck"] == SUPPORTED
                         and out["attribution_main"]["mode"] == SUPPORTED) else UNSUPPORTED
    out["attribution_used"] = mode
    out["stage_b"] = b
    out["stage_b_checks"] = stage_b_checks(b, mode)
    out["analysis"] = analyze(b["series"], mode)
    out["rss_end"] = rss_mb()
    bad = [k for k, v in out["stage_b_checks"].items() if not v]
    if not out["c7_observations"]["ok"]:
        bad.append("c7_observations")
    out["status"] = "ok" if not bad else f"выполнен, нарушены проверки: {bad}"
    out["stage"] = "end"


def cmd_run() -> None:
    check_protocol()
    A6.require_selfcheck()
    A7.require_selfcheck()
    sc = require_selfcheck()
    conn = G.connect()
    out: dict = {"impl_sha256": impl_hashes(), "attribution_selfcheck": sc["p3a_attribution"]["mode"], "stage": "start"}
    n = A7.SIZES["N10"]
    bits = A7.L.bits_for(n, A7.L.SIGMA)  # as A7 cmd_dynamics: 128 partitions at N10
    ctx = new_ctx("n10", bits)
    res: dict = {}
    b: dict = {}
    ext = {"created": []}
    failed = False
    try:
        out["settings"] = {k: conn.execute(f"SHOW {k}").fetchone()[0] for k in SETTINGS}
        out["preconditions"] = preconditions(conn)
        out["free_bytes_before"] = shutil.disk_usage(G.require_data_dir()).free
        if out["free_bytes_before"] < MIN_FREE or not out["preconditions"]["block_size_ok"]:
            raise A7.Stop("НЕ ВЫПОЛНЕНО: мало места на томе данных PostgreSQL или block_size не 8192")
        ensure_extensions(conn, ext)
        out["extensions"] = ext
        out["mem_start"] = mem("start")
        run_body(out, res, b, ctx, n, bits, conn)
    except BaseException as e:  # keep what was gathered, the stage and the cause (R7 Codex 262, R2 Codex 264)
        failed = True
        out.update({k: v for k, v in public(res).items() if k not in out})
        if b:
            out["stage_b_partial"] = b
        loop = res.get("_loop") or {}
        out["failure_stage"] = b.get("at") if b else (loop.get("stage") or res.get("stage"))
        out["status"] = f"ОСТАНОВЛЕН: {type(e).__name__}: {str(e)[:300]}"
    finally:
        name = "a8_n10_FAILED.json" if failed else "a8_n10.json"
        out["write_errors"] = [w for w in [safe_write(RES / name, out)] if w]  # on disk before cleanup
        out["cleanup_errors"] = cleanup(ctx, res.get("_a"), bool(res.get("created_tables")), ext["created"])
        if (err := close_quietly(conn)) is not None:
            out["cleanup_errors"].append(err)
        if out["cleanup_errors"] or out["write_errors"]:
            w2 = safe_write(RES / name, out)
            if w2:
                out["write_errors"].append(w2)
    if failed:
        log("FAILED", out["status"], out.get("failure_stage"), out["write_errors"])
        raise SystemExit(2)
    if out["write_errors"] and safe_write(RES / name, out) is not None:  # R3 Codex 266: the result must be on disk
        log("RESULT NOT SAVED", out["write_errors"])
        raise SystemExit(3)
    an = out["analysis"]
    log("done", out["status"], "D1", out["d1"].get("repeated"), "mode", out["attribution_used"],
        {k: v["summary"]["G"]["describe"] for k, v in an.items()
         if isinstance(v, dict) and isinstance(v.get("summary"), dict) and "G" in v["summary"]}, out["write_errors"])


if __name__ == "__main__":
    G.check_cluster()  # SQL and RCM_PG_DATA address the same cluster (free space, device counters)
    if len(sys.argv) >= 2 and sys.argv[1] == "selfcheck":
        cmd_selfcheck()
    elif len(sys.argv) >= 2 and sys.argv[1] == "run":
        cmd_run()
    else:
        raise SystemExit(__doc__)
