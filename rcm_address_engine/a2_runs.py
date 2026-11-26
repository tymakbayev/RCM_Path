"""A2 main runs: selfcheck (C5), static (N1 / N10), dynamics (N1), open-probe (M2a).
Protocol A2_protocol.md edition 2 + A2_addendum_2026-11-16.md. Called from a2.py."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import a2_core as C
import a2_pg as G

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DATA = ROOT / "data"
SEED_A, SEED_B = 0xA2A2_0001, 0xA2B2_0001
PY = Path(sys.executable)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha16(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()[:16]


# ------------------------------------------------------------------ sidecar builds + manifests
def write_manifest(path: Path, **fields) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields, indent=1), encoding="utf-8")
    tmp.replace(path)


def build_a(conn, table: str, gen: int, watermark: int, tag: str, *, repeatable: bool = False):
    """Variant A from SELECT k, ctid. With repeatable=True the watermark and the export share one snapshot (§5)."""
    t0 = time.perf_counter()
    if repeatable:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")  # autocommit connection: explicit transaction block
        watermark = int(conn.execute(f"SELECT coalesce(max(seq), 0) FROM {table}").fetchone()[0])
        keys, pages, _ = G.read_kp(conn, table)
        conn.execute("COMMIT")
    else:
        keys, pages, _ = G.read_kp(conn, table)
    t_read = time.perf_counter() - t0
    ri = G.relinfo(conn, table)
    r = G.r_bits_for(ri["pages"])
    ca, t_build = _timed(C.build_a, keys, pages, SEED_A ^ gen, r)
    path = DATA / f"{tag}.g{gen}.adr"
    info = C.write_a(path, ca, table_oid=ri["oid"], relfilenode=ri["relfilenode"], generation=gen, n_rows=len(keys),
                     n_pages=ri["pages"], watermark=watermark)
    write_manifest(DATA / f"{tag}.manifest.json", kind="A", generation=gen, file=path.name, sha256=info["sha256"],
                   oid=ri["oid"], relfilenode=ri["relfilenode"], watermark=watermark, table=table,
                   delta_index=f"{tag}_delta_g{gen}")
    info |= {"read_s": t_read, "build_s": t_build, "r_bits": r, "rows": len(keys), "pages": ri["pages"],
             "watermark": watermark, "path": str(path)}
    return C.SidecarA(path), info, keys, pages


def build_b(conn, src: str, dst: str, gen: int, watermark: int | None, tag: str, rows_per_page: float | None = None):
    """Variant B: new table in (bucket, seq) order, layout with back-fill exceptions (addendum §3).
    rows_per_page: density of the freshly loaded table A (a reseal passes the base density, not the bloated one)."""
    n_src = int(conn.execute(f"SELECT count(*) FROM {src}").fetchone()[0])
    if rows_per_page is None:
        rows_per_page = n_src / max(G.relinfo(conn, src)["pages"], 1)
    b = C.bucket_bits(n_src, rows_per_page)
    load = G.load_bucketed(conn, src, dst, SEED_B ^ gen, b)
    t0 = time.perf_counter()
    keys, pages, seq = G.read_kp(conn, dst, with_seq=True)
    if watermark is None:
        watermark = int(seq.max())
    bucket = C.buckets_of(keys, SEED_B ^ gen, b)
    starts, exc_rows = C.bucket_layout(bucket, pages, seq, b)
    violations = C.bucket_invariant(bucket, pages, starts, exc_rows)
    ri = G.relinfo(conn, dst)
    path = DATA / f"{tag}.g{gen}.hpl"
    info = C.write_b(path, starts, seed=SEED_B ^ gen, b=b, table_oid=ri["oid"], relfilenode=ri["relfilenode"],
                     generation=gen, n_rows=len(keys), n_pages=ri["pages"], watermark=watermark,
                     exc_keys=[bytes(keys[i]) for i in exc_rows], exc_pages=[int(pages[i]) for i in exc_rows])
    write_manifest(DATA / f"{tag}.manifest.json", kind="B", generation=gen, file=path.name, sha256=info["sha256"],
                   oid=ri["oid"], relfilenode=ri["relfilenode"], watermark=watermark, table=dst,
                   delta_index=f"{tag}_delta_g{gen}")
    span = starts[1:].astype(np.int64) - starts[:-1].astype(np.int64) + 1
    info |= {"rows_per_page_rule": rows_per_page, "b": b, "load_s": load["load_s"], "layout_s": time.perf_counter() - t0, "rows": len(keys),
             "pages": ri["pages"], "misplaced_rows": int(len(exc_rows)), "c1_violations": violations,
             "pages_per_bucket": {"mean": float(span.mean()), "p50": float(np.median(span)),
                                  "p99": float(np.percentile(span, 99)), "max": int(span.max())},
             "watermark": watermark, "path": str(path), "table": dst}
    return C.SidecarB(path), info, keys, pages


def _timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


# ------------------------------------------------------------------ measurement of one method
def absent_keys(ref_keys: np.ndarray, n: int, seed: int) -> list[bytes]:
    ref_h = np.sort(C.key_hash(ref_keys, 1))
    cand = np.random.default_rng(seed).integers(0, 256, size=(int(n * 1.01) + 10, 16), dtype=np.uint8)
    h = C.key_hash(cand, 1)
    pos = np.searchsorted(ref_h, h)
    pos[pos == len(ref_h)] = 0
    free = ref_h[pos] != h  # equal 64-bit hash -> treated as present and dropped (conservative)
    return [bytes(k) for k in cand[free][:n]]


def measure(table: str, kind: str, present: list[tuple[bytes, bytes, int]], absent: list[bytes], *, sa=None, sb=None,
            watermark=None, n_explain: int = 2000) -> dict:
    """C2/C3 correctness + M3 for one method. present = [(key, sha16 of payload, v)]."""
    conn, reader = G.connect(), G.connect()
    cl = G.Client(conn, table, kind, sa=sa, sb=sb, watermark=watermark)
    err_p, err_a, times, addr = 0, 0, [], []
    for pass_ in (0, 1):
        for kb, dig, v in present:
            t = time.perf_counter_ns()
            row = cl.get(kb)
            dt = time.perf_counter_ns() - t
            if pass_ == 0:
                if row is None or row[1] != v or sha16(row[3]) != dig:
                    err_p += 1
            else:
                times.append(dt / 1000)
                addr.append(cl.addr_ns / 1000)
    t_abs = []
    for kb in absent:
        t = time.perf_counter_ns()
        if cl.get(kb) is not None:
            err_a += 1
        t_abs.append((time.perf_counter_ns() - t) / 1000)
    ex_p = [G.explain(conn, *cl.query(kb)) for kb, _, _ in present[:n_explain]]
    ex_a, no_sql = [], 0
    for kb in absent[:n_explain]:
        q = cl.query(kb)
        if q is None:
            no_sql += 1
        else:
            ex_a.append(G.explain(conn, *q))
    pk = [kb for kb, _, _ in present[:n_explain]]
    ak = absent[:n_explain]
    st_p = G.statio_split(conn, reader, table, lambda: [cl.get(kb) for kb in pk] and len(pk))
    st_a = G.statio_split(conn, reader, table, lambda: [cl.get(kb) for kb in ak] and len(ak))
    conn.close()
    reader.close()

    def mean(xs, f):
        return float(np.mean([f(x) for x in xs])) if xs else 0.0
    n_a = len(ak)
    return {"kind": kind, "c2_errors": err_p, "c3_errors": err_a, "present": len(present), "absent": len(absent),
            "client_us": G.pct(times), "client_absent_us": G.pct(t_abs),
            "address_us": G.pct(addr) if kind in ("A", "B") else None,
            "explain_present": {"buffers_mean": mean(ex_p, lambda e: e["buffers"]),
                                "buffers_max": max(e["buffers"] for e in ex_p),
                                "read_mean": mean(ex_p, lambda e: e["read"]),
                                "exec_us": G.pct([e["exec_ms"] * 1000 for e in ex_p]),
                                "planning_buffers_mean": mean(ex_p, lambda e: e["planning_buffers"]),
                                "serialize_buffers_mean": mean(ex_p, lambda e: e["serialize_buffers"]),
                                "plan": ex_p[0]["nodes"]},
            "explain_absent": {"buffers_mean_over_all": (sum(e["buffers"] for e in ex_a) / n_a) if n_a else 0.0,
                               "no_sql_fraction": no_sql / n_a if n_a else 0.0,
                               "plan": ex_a[0]["nodes"] if ex_a else None},
            "statio_present": st_p, "statio_absent": st_a}


def restart_probe(table: str, kind: str, keys: list[bytes], *, sa=None, sb=None, watermark=None, runs: int = 3) -> list:
    out = []
    for r in range(runs):
        G.pg_ctl("restart")
        conn = G.connect()
        cl = G.Client(conn, table, kind, sa=sa, sb=sb, watermark=watermark)
        first = G.explain(conn, *cl.query(keys[r * 201]))
        rest = [G.explain(conn, *cl.query(kb)) for kb in keys[r * 201 + 1: r * 201 + 201]]
        out.append({"first_buffers": first["buffers"], "first_read": first["read"], "first_exec_us": first["exec_ms"] * 1000,
                    "next200_buffers_mean": float(np.mean([e["buffers"] for e in rest])),
                    "next200_read_mean": float(np.mean([e["read"] for e in rest])),
                    "next200_exec_us_mean": float(np.mean([e["exec_ms"] * 1000 for e in rest]))})
        conn.close()
    return out


def open_probe_runs(manifest: Path, keys: list[bytes], runs: int = 21) -> dict:
    res = []
    for i in range(runs):
        r = subprocess.run([str(PY), str(ROOT / "a2.py"), "open-probe", str(manifest), keys[i].hex()],
                           capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        res.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return {k: G.pct([x[k] for x in res]) | {"max": max(x[k] for x in res)} for k in res[0]}


def cmd_open_probe(manifest: str, keyhex: str) -> None:
    """M2(a): new process; connect (separately); manifest -> sidecar ready (mmap + header + pg_class check) -> answer."""
    t0 = time.perf_counter()
    conn = G.connect()
    t1 = time.perf_counter()
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    path = Path(manifest).parent / m["file"]
    if m["kind"] == "A":
        sa, sb = C.SidecarA(path, count_pages=True), None
        oid, rfn = sa.table_oid, sa.relfilenode
    else:
        sa, sb = None, C.SidecarB(path)
        oid, rfn = sb.table_oid, sb.relfilenode
    row = conn.execute("SELECT oid, relfilenode FROM pg_class WHERE relname = %s", (m["table"],)).fetchone()
    if row is None or int(row[0]) != oid or int(row[1]) != rfn or oid != m["oid"]:
        raise SystemExit("sidecar does not match the table (relfilenode / oid)")
    t2 = time.perf_counter()
    cl = G.Client(conn, m["table"], m["kind"], sa=sa, sb=sb)
    kb = bytes.fromhex(keyhex)
    row = cl.get(kb)
    t3 = time.perf_counter()
    pages = len(sa.touched) if sa else sb.touched_pages(kb)
    print(json.dumps({"connect_ms": (t1 - t0) * 1e3, "open_ms": (t2 - t1) * 1e3, "first_answer_ms": (t3 - t2) * 1e3,
                      "found": row is not None, "sidecar_pages_touched": pages}))


# ------------------------------------------------------------------ static N1 / N10
def cmd_static(size: str) -> None:
    n = {"N1": 1_000_000, "N10": 10_000_000}[size]
    tag = size.lower()
    out: dict = {"size": size, "rows": n}
    conn = G.connect()
    out["settings"] = {k: v for k, v in G.show_all(conn).items()
                       if k in ("shared_buffers", "work_mem", "wal_level", "max_wal_size", "server_version", "jit",
                                "effective_cache_size", "random_page_cost")}
    table = f"a_{tag}"
    log(size, "load")
    out["load"] = G.load(conn, table, 0, n, DATA / f"ref_{tag}")
    ref_keys, ref_dig, ref_len = (np.load(DATA / f"ref_{tag}.{x}") for x in ("keys", "dig", "len"))
    out["table"] = G.relinfo(conn, table) | {"payload_len": {"min": int(ref_len.min()), "p50": float(np.median(ref_len)),
                                                             "p99": float(np.percentile(ref_len, 99)),
                                                             "max": int(ref_len.max())}}
    out["table"]["rows_per_page"] = n / out["table"]["pages"]
    log(size, "table", out["table"])
    rng = np.random.default_rng(11)
    pidx = rng.choice(n, 100_000, replace=False)
    present = [(bytes(ref_keys[i]), bytes(ref_dig[i]), 1) for i in pidx]
    absent = absent_keys(ref_keys, 100_000, 12)
    pkeys = [p[0] for p in present]
    # ---- A
    log(size, "sidecar A")
    sa, info, keys, pages = build_a(conn, table, 1, 0, f"st_{tag}")
    same = np.array_equal(np.sort(ref_keys.view("V16").ravel()), np.sort(keys.view("V16").ravel()))
    info["keys_match_ref"] = bool(same)
    info["c1_errors"] = int((sa.pages_of(keys) != pages.astype(np.int64)).sum())
    info["bytes_per_row"] = info["bytes"] / n
    manifest = DATA / f"st_{tag}.manifest.json"
    info["manifest_bytes"] = manifest.stat().st_size
    out["A"] = {"build": info}
    log(size, "A build", {k: info[k] for k in ("bytes_per_row", "levels", "exceptions", "c1_errors", "r_bits")})
    pre, sp = C.build_sort(keys, pages, SEED_A)
    out["SORT"] = {"bytes_per_row": (pre.nbytes + sp.nbytes + 64) / n, "c1_errors": C.sort_c1(keys, pages, SEED_A, pre, sp)}
    out["EF"] = {"bytes_per_row": C.ef_bytes(keys, SEED_A, info["r_bits"])}
    del pre, sp
    log(size, "A measure")
    out["A"]["measure"] = measure(table, "A", present, absent, sa=sa)
    out["A"]["open_probe"] = open_probe_runs(manifest, pkeys)
    out["A"]["restart"] = restart_probe(table, "A", pkeys, sa=sa)
    conn = G.connect()
    # ---- PostgreSQL indexes, one at a time on the same table
    for kind, ddl in (("btree", f"CREATE UNIQUE INDEX {table}_bt ON {table} USING btree (k)"),
                      ("hash", f"CREATE INDEX {table}_hash ON {table} USING hash (k)")):
        log(size, kind)
        _, t = _timed(conn.execute, ddl)
        name = f"{table}_bt" if kind == "btree" else f"{table}_hash"
        conn.execute(f"ANALYZE {table}")
        size_info = G.index_size(conn, name)
        out[kind] = {"create_s": t, "size": size_info, "bytes_per_row": size_info["main_bytes"] / n,
                     "total_bytes_per_row": size_info["total_bytes"] / n}
        out[kind]["measure"] = measure(table, kind, present, absent)
        out[kind]["restart"] = restart_probe(table, kind, pkeys)
        conn = G.connect()
        conn.execute(f"DROP INDEX {name}")
    # ---- B
    log(size, "sidecar B")
    sb, binfo, bkeys, bpages = build_b(conn, table, f"b_{tag}", 1, None, f"stb_{tag}")
    binfo["bytes_per_row"] = binfo["bytes"] / n
    out["B"] = {"build": binfo, "table": G.relinfo(conn, f"b_{tag}")}
    log(size, "B build", {k: binfo[k] for k in ("bytes_per_row", "b", "misplaced_rows", "c1_violations")})
    del bkeys, bpages
    out["B"]["measure"] = measure(f"b_{tag}", "B", present, absent, sb=sb)
    out["B"]["open_probe"] = open_probe_runs(DATA / f"stb_{tag}.manifest.json", pkeys)
    out["B"]["restart"] = restart_probe(f"b_{tag}", "B", pkeys, sb=sb)
    conn = G.connect()
    # ---- BRIN control on a key-ordered copy (N1 only, descriptive)
    if size == "N1":
        log(size, "BRIN control")
        conn.execute(f"DROP TABLE IF EXISTS c_{tag}")
        G.create_table(conn, f"c_{tag}")
        conn.execute(f"INSERT INTO c_{tag} SELECT * FROM {table} ORDER BY k")
        conn.execute(f"VACUUM (ANALYZE) c_{tag}")
        out["BRIN"] = {"table": G.relinfo(conn, f"c_{tag}")}
        for ppr in (1, 4):
            name = f"c_{tag}_brin{ppr}"
            _, t = _timed(conn.execute, f"CREATE INDEX {name} ON c_{tag} USING brin (k) WITH (pages_per_range = {ppr})")
            conn.execute(f"ANALYZE c_{tag}")
            s = G.index_size(conn, name)
            m = measure(f"c_{tag}", "brin", present[:2000], absent[:2000])
            out["BRIN"][f"ppr{ppr}"] = {"create_s": t, "size": s, "bytes_per_row": s["main_bytes"] / n, "measure": m}
            conn.execute(f"DROP INDEX {name}")
        conn.execute(f"DROP TABLE c_{tag}")
    sa.close()
    sb.close()
    G.write_json(RES / f"a2_static_{tag}.json", out)
    log(size, "done")


# ------------------------------------------------------------------ C5 selfcheck (100 000 rows)
def cmd_selfcheck() -> None:
    out: dict = {}
    r = subprocess.run([str(PY), str(ROOT / "a2_selftest_core.py")], capture_output=True, text=True, cwd=str(ROOT))
    out["core_selftest"] = {"exit": r.returncode, "tail": r.stdout.strip().splitlines()[-1]}
    conn = G.connect()
    n = 100_000
    G.load(conn, "sc", 0, n, DATA / "sc")
    ref_keys, ref_dig = np.load(DATA / "sc.keys"), np.load(DATA / "sc.dig")
    dig_of = {bytes(k): bytes(d) for k, d in zip(ref_keys, ref_dig)}
    sa, info, keys, pages = build_a(conn, "sc", 1, 0, "sc")

    def ok_on(client, kb) -> bool:
        row = client.get(kb)
        return row is not None and row[1] == 1 and sha16(row[3]) == dig_of[kb]

    witness = bytes(keys[4321])
    results = {}
    # (a) D of the witness points to a page without it
    h = C.key_hash(keys, SEED_A ^ 1)
    ca = C.build_a(keys, pages, SEED_A ^ 1, info["r_bits"])
    wpage = int(pages[4321])
    other = (wpage + 1) % info["pages"]
    sl = None
    for level, (m, off) in enumerate(zip(ca.level_m, ca.level_off)):
        g = off + C.level_pos_int(int(h[4321]), level, m)
        b, i = divmod(g, C.BLOCK_BITS)
        w = int(ca.blocks[b, 1 + (i >> 5)])
        if (w >> (i & 31)) & 1:
            sl = int(ca.blocks[b, 0]) + sum(int(ca.blocks[b, 1 + j]).bit_count() for j in range(i >> 5)) + \
                (w & ((1 << (i & 31)) - 1)).bit_count()
            wblock = b
            break
    width = C.F_BITS + info["r_bits"]
    vals = C.unpack_fixed(ca.d_words, np.arange(ca.n_slots), width)

    def mutant_file(name, ca_m) -> C.SidecarA:
        p = DATA / f"sc_mut_{name}.adr"
        C.write_a(p, ca_m, table_oid=0, relfilenode=0, generation=1, n_rows=n, n_pages=info["pages"], watermark=0)
        return C.SidecarA(p)

    va = vals.copy()
    va[sl] = (va[sl] & np.uint64(0xFF)) | (np.uint64(other) << np.uint64(C.F_BITS))
    ma = C.CascadeA(ca.seed, ca.r_bits, ca.level_m, ca.level_off, ca.blocks, C.pack_fixed(va, width), ca.n_slots,
                    ca.exc_keys, ca.exc_pages)
    results["a_D_wrong_page"] = not ok_on(G.Client(conn, "sc", "A", sa=mutant_file("a", ma)), witness)
    # (b) rank of the witness block one lower
    bl = ca.blocks.copy()
    bl[wblock, 0] -= 1
    mb = C.CascadeA(ca.seed, ca.r_bits, ca.level_m, ca.level_off, bl, ca.d_words, ca.n_slots, ca.exc_keys, ca.exc_pages)
    results["b_rank_minus_one"] = not ok_on(G.Client(conn, "sc", "A", sa=mutant_file("b", mb)), witness)
    # (c) no key check in SQL: the first row of the page is returned
    mc = G.Client(conn, "sc", "A", sa=sa, mutant="c")
    caught_c2 = sum(1 for kb in [bytes(k) for k in keys[:2000]] if not ok_on(mc, kb))
    results["c_no_key_check"] = caught_c2 > 0
    results["c_no_key_check_errors_on_2000"] = caught_c2
    # (d) exceptions not consulted, with one artificial full-hash collision
    real = C.key_hash
    wi, wj = 4321, 777

    def colliding(k, seed):
        hh = real(k, seed)
        if len(k) == n:
            hh = hh.copy()
            hh[wi] = hh[wj]
        return hh
    C.key_hash = colliding
    cd = C.build_a(keys, pages, SEED_A ^ 1, info["r_bits"])
    C.key_hash = real
    sd = mutant_file("d_ok", cd)
    results["d_exceptions_used_when_present"] = len(cd.exc_keys) == 2 and ok_on(G.Client(conn, "sc", "A", sa=sd), witness)
    md = C.CascadeA(cd.seed, cd.r_bits, cd.level_m, cd.level_off, cd.blocks, cd.d_words, cd.n_slots, [], [])
    results["d_exceptions_skipped"] = not ok_on(G.Client(conn, "sc", "A", sa=mutant_file("d", md)), witness)
    # B: (g) start of the witness bucket shifted one page forward; (e) exceptions skipped
    sb, binfo, bkeys, bpages = build_b(conn, "sc", "sc_b", 1, None, "scb")
    bucket = C.buckets_of(bkeys, SEED_B ^ 1, binfo["b"])
    starts = np.asarray(sb.starts).copy()
    exc_set = set(sb.exc_keys)
    cand = [i for i in range(len(bkeys)) if bytes(bkeys[i]) not in exc_set and bpages[i] == starts[bucket[i]]
            and starts[bucket[i]] < starts[bucket[i] + 1]]
    wb = cand[0]
    st_m = starts.copy()
    st_m[bucket[wb]] += 1
    pm = DATA / "scb_mut_g.hpl"
    C.write_b(pm, st_m, seed=SEED_B ^ 1, b=binfo["b"], table_oid=0, relfilenode=0, generation=1, n_rows=n,
              n_pages=binfo["pages"], watermark=0, exc_keys=list(sb.exc_keys), exc_pages=list(sb.exc_pages))
    results["g_bucket_start_shift"] = not ok_on(G.Client(conn, "sc_b", "B", sb=C.SidecarB(pm)), bytes(bkeys[wb]))
    if sb.exc_keys:
        we = sb.exc_keys[0]
        results["e_B_exceptions_used"] = ok_on(G.Client(conn, "sc_b", "B", sb=sb), we)
        results["e_B_exceptions_skipped"] = not ok_on(
            G.Client(conn, "sc_b", "B", sb=C.SidecarB(Path(binfo["path"]), skip_exceptions=True)), we)
    else:
        results["e_B_exceptions_skipped"] = None  # no back-filled rows in this table: mutant not applicable
    # edge: absent key that meets no set bit at any level -> no SQL at all
    cl = G.Client(conn, "sc", "A", sa=sa)
    no_bit = 0
    for kb in absent_keys(ref_keys, 20_000, 99):
        hh = C.key_hash_int(kb, sa.seed)
        if all(not ((int(sa.blocks[(off + C.level_pos_int(hh, l, m)) // C.BLOCK_BITS,
                                    1 + (((off + C.level_pos_int(hh, l, m)) % C.BLOCK_BITS) >> 5)])
                     >> (((off + C.level_pos_int(hh, l, m)) % C.BLOCK_BITS) & 31)) & 1)
               for l, (m, off) in enumerate(sa.lev)):
            no_bit += 1
            if cl.query(kb) is not None or cl.get(kb) is not None:
                results["edge_no_bit_absent"] = False
                break
    results.setdefault("edge_no_bit_absent", no_bit > 0)
    results["edge_no_bit_absent_count"] = no_bit
    # baseline correctness of the unmutated paths on the whole table
    full_ok = sum(1 for k in keys[::50] if not ok_on(cl, bytes(k)))
    results["unmutated_A_errors_on_2000"] = full_ok
    out["mutants"] = results
    flags = [v for k, v in results.items() if isinstance(v, bool)]
    out["c1_errors_A"] = int((sa.pages_of(keys) != pages.astype(np.int64)).sum())
    out["c1_violations_B"] = binfo["c1_violations"]
    out["all_ok"] = (r.returncode == 0 and all(flags) and full_ok == 0 and out["c1_errors_A"] == 0
                     and out["c1_violations_B"] == 0)
    for t in ("sc", "sc_b"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    G.write_json(RES / "a2_selfcheck.json", out)
    log("selfcheck", json.dumps(out, ensure_ascii=False))
