"""A2 runner (protocol A2_protocol.md edition 2, sha256 in results/a2_protocol.sha256). Run in .venv-pg.

Commands: pilot | selfcheck | static N1|N10 | dynamics | open-probe <manifest>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import a2_core as C
import a2_pg as G

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "results"  # layout: results/ and docs/experiments/ at the top of the repository
DOCS = ROOT.parent / "docs" / "experiments"
DATA = ROOT / "data"
PROTOCOL_SHA = "e68edb6cfb5101a9dbd23baae02eb22a50a19a086dbe8462865b79e457150c36"
SEED_A, SEED_B = 0xA2A2_0001, 0xA2B2_0001


def check_protocol() -> None:
    sha = hashlib.sha256((DOCS / "A2_protocol.md").read_bytes()).hexdigest()
    if sha != PROTOCOL_SHA:
        raise SystemExit(f"protocol changed: {sha}")


class Server:
    """Start PostgreSQL for the run if it is not running; stop it afterwards only if we started it."""

    def __enter__(self):
        self.started = not G.pg_running()
        if self.started:
            G.pg_ctl("start")
        G.check_cluster()  # SQL and pg_ctl address the same cluster, checked before any measurement
        G.ensure_db()
        return self

    def __exit__(self, *exc):
        if self.started:
            G.pg_ctl("stop")


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def build_sidecar_a(conn, table: str, path: Path, generation: int, watermark: int) -> tuple[C.SidecarA, dict]:
    ri = G.relinfo(conn, table)
    (keys, pages, _), t_read = timed(G.read_kp, conn, table)
    r = G.r_bits_for(ri["pages"])
    ca, t_build = timed(C.build_a, keys, pages, SEED_A ^ generation, r)
    info = C.write_a(path, ca, table_oid=ri["oid"], relfilenode=ri["relfilenode"], generation=generation,
                     n_rows=len(keys), n_pages=ri["pages"], watermark=watermark)
    info |= {"read_s": t_read, "build_s": t_build, "r_bits": r, "rows": len(keys), "pages": ri["pages"]}
    return C.SidecarA(path), info | {"keys": keys, "pages_arr": pages}


# ------------------------------------------------------------------ pilot P0
def cmd_pilot() -> None:
    check_protocol()
    out: dict = {"protocol_sha256": PROTOCOL_SHA}
    with Server():
        conn = G.connect()
        out["settings"] = {k: v for k, v in G.show_all(conn).items()
                           if k in ("shared_buffers", "work_mem", "wal_level", "max_wal_size", "server_version", "jit")}
        n = 100_000
        out["load"] = G.load(conn, "p0", 0, n, DATA / "p0")
        ri = G.relinfo(conn, "p0")
        lens = np.load(DATA / "p0.len")
        out["table"] = ri | {"rows_per_page": n / ri["pages"], "payload_len": {
            "min": int(lens.min()), "p50": float(np.median(lens)), "p99": float(np.percentile(lens, 99)),
            "max": int(lens.max())}}
        sa, info = build_sidecar_a(conn, "p0", DATA / "p0.adr", 1, 0)
        keys, pages = info.pop("keys"), info.pop("pages_arr")
        ref = np.load(DATA / "p0.keys")
        same = np.array_equal(np.sort(ref.view("V16").ravel()), np.sort(keys.view("V16").ravel()))
        c1 = int((sa.pages_of(keys) != pages.astype(np.int64)).sum())
        out["sidecar_a"] = info | {"bytes_per_row": info["bytes"] / n, "keys_match_ref": bool(same), "c1_errors": c1}
        print("A:", out["sidecar_a"])
        # ---- plans
        plans = {}
        a = G.Client(conn, "p0", "A", sa=sa)
        q = a.query(bytes(keys[123]))
        plans["static_A"] = G.explain(conn, *q)
        conn.execute("DROP SEQUENCE IF EXISTS p0_seq")
        conn.execute(f"CREATE SEQUENCE p0_seq START {n + 1}")
        w = n - 1000
        conn.execute(f"CREATE INDEX p0_delta ON p0 (k) WHERE seq > {w}")
        ad = G.Client(conn, "p0", "A", sa=sa, watermark=w)
        upd_key, unchanged_key = bytes(keys[500]), bytes(keys[600])
        row = ad.get(upd_key)
        upd_sql = "UPDATE p0 SET payload = payload, v = v + 1, seq = nextval('p0_seq') WHERE ctid = %s::tid AND k = %s"
        plans["update_by_ctid"] = G.explain(conn, upd_sql, (row[0], uuid.UUID(bytes=upd_key)))
        plans["lookup_updated"] = G.explain(conn, *ad.query(upd_key))
        plans["lookup_unchanged"] = G.explain(conn, *ad.query(unchanged_key))
        absent = None
        for cand in np.random.default_rng(12).integers(0, 256, size=(1000, 16), dtype=np.uint8):
            if sa.page_of(bytes(cand)) is None:
                absent = bytes(cand)
                break
        plans["lookup_absent_fp_miss"] = G.explain(conn, *ad.query(absent))
        del_key = bytes(keys[700])
        row = ad.get(del_key)
        plans["delete_by_ctid"] = G.explain(conn, "DELETE FROM p0 WHERE ctid = %s::tid AND k = %s",
                                            (row[0], uuid.UUID(bytes=del_key)))
        got_upd, got_del = ad.get(upd_key), ad.get(del_key)
        plans["answers"] = {"updated_v": got_upd[1] if got_upd else None, "deleted_found": got_del is not None}
        out["plans"] = plans
        for name, p in plans.items():
            if isinstance(p, dict) and "nodes" in p:
                print(name, p["buffers"], p["nodes"])
        print("answers", plans["answers"])
        conn.execute("DROP INDEX p0_delta")
        # ---- statio methodology: sidecar A (heap only) and B-tree (index + heap)
        sample = [bytes(k) for k in keys[np.random.default_rng(11).choice(n, 2000, replace=False)]]
        reader = G.connect()
        mc = G.connect()
        am = G.Client(mc, "p0", "A", sa=sa)
        out["statio_A"] = G.statio_split(mc, reader, "p0", lambda: sum(1 for kb in sample if am.get(kb) is not None))
        (_, t_bt) = timed(conn.execute, "CREATE UNIQUE INDEX p0_bt ON p0 USING btree (k)")
        bm = G.Client(mc, "p0", "btree")
        out["statio_btree"] = G.statio_split(mc, reader, "p0", lambda: sum(1 for kb in sample if bm.get(kb) is not None))
        out["btree_size"] = G.index_size(conn, "p0_bt") | {"create_s": t_bt}
        ex = [G.explain(conn, *bm.query(kb)) for kb in sample[:200]]
        out["explain_btree_mean_buffers"] = float(np.mean([e["buffers"] for e in ex]))
        conn.execute("DROP INDEX p0_bt")
        ex = [G.explain(conn, *am.query(kb)) for kb in sample[:200]]
        out["explain_A_mean_buffers"] = float(np.mean([e["buffers"] for e in ex]))
        print("statio A", out["statio_A"], "btree", out["statio_btree"])
        print("explain mean buffers A", out["explain_A_mean_buffers"], "btree", out["explain_btree_mean_buffers"])
        # ---- variant B on the real payload length distribution
        conn.execute("VACUUM (ANALYZE) p0")
        b = C.bucket_bits(n, n / ri["pages"])
        out["load_b"] = G.load_bucketed(conn, "p0", "p0_b", SEED_B, b)
        kb_, pb_, _ = G.read_kp(conn, "p0_b")
        bucket = C.buckets_of(kb_, SEED_B, b)
        starts = C.bucket_starts(bucket, pb_, b)
        span = starts[1:].astype(np.int64) - starts[:-1].astype(np.int64) + 1
        out["sidecar_b"] = {"b": b, "invariant_violations": C.bucket_invariant(bucket, pb_, starts),
                            "bytes_per_row": (4 * len(starts) + 64 + 32) / len(kb_),
                            "pages_per_bucket": {"mean": float(span.mean()), "p99": float(np.percentile(span, 99)),
                                                 "max": int(span.max())}}
        print("B:", out["sidecar_b"])
        # ---- timing extrapolation to N10 (x100)
        t_lookup = []
        for kb in sample[:500]:
            t = time.perf_counter_ns()
            am.get(kb)
            t_lookup.append(time.perf_counter_ns() - t)
        out["lookup_us_p50"] = float(np.median(t_lookup)) / 1000
        f = 100
        est = {"load_s": out["load"]["copy_s"] * f, "vacuum_s": out["load"]["vacuum_s"] * f,
               "read_build_s": (info["read_s"] + info["build_s"]) * f, "load_b_s": out["load_b"]["load_s"] * f,
               "btree_s": t_bt * f}
        est["lookups_s_per_100k_per_method"] = out["lookup_us_p50"] * 1e-6 * 100_000 * 2
        out["n10_estimate"] = est
        print("N10 estimate:", est)
        for t in ("p0", "p0_b"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.execute("DROP SEQUENCE IF EXISTS p0_seq")
        sa.close()
    G.write_json(RES / "a2_p0.json", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pilot":
        cmd_pilot()
    elif cmd == "open-probe":
        import a2_runs
        a2_runs.cmd_open_probe(sys.argv[2], sys.argv[3])
    elif cmd == "dynamics-a3":
        import hashlib as _h
        sha = _h.sha256((DOCS / "A3_protocol.md").read_bytes()).hexdigest()
        frozen = (RES / "a3_protocol.sha256").read_text().split()[0]
        if sha != frozen:
            raise SystemExit(f"A3 protocol changed: {sha}")
        with Server():
            import a2_dyn
            a2_dyn.cmd_dynamics(reseal_every=30_000, include_b=False, out_name="a3_dynamics", run_scenarios=False)
    elif cmd in ("selfcheck", "static", "dynamics"):
        import a2_runs
        check_protocol()
        with Server():
            if cmd == "selfcheck":
                a2_runs.cmd_selfcheck()
            elif cmd == "static":
                a2_runs.cmd_static(sys.argv[2])
            else:
                import a2_dyn
                a2_dyn.cmd_dynamics()
    else:
        raise SystemExit(__doc__)
