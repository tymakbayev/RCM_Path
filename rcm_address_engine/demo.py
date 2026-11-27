"""Демонстрация одной командой: вычисляемый адрес RCM против B-tree на одной и той же таблице PostgreSQL.

    python demo.py [--rows 1000000] [--lookups 20000] [--keep]

Что делает:
1. загружает в новую таблицу rcm_demo_<номер> N записей тем же генератором, что и эксперименты A2–A8 (ключ uuid,
   JSON-полезная нагрузка; около 0,65 ГБ на 1 млн строк);
2. строит обычный B-tree по ключу и адрес RCM (rcm.build) по той же таблице;
3. ищет одни и те же ключи обоими способами, чередуя порядок, и сверяет каждый ответ с эталоном генератора;
4. печатает и записывает в results/demo_result.json: байт на ключ, обращения к буферам на поиск (EXPLAIN BUFFERS),
   задержку p50/p95 в клиенте, долю отсутствующих ключей, отсеянных без SQL, и число неверных ответов;
5. удаляет свою таблицу и свои файлы (если не задан --keep).

Оба способа выполняются с plan_cache_mode = force_custom_plan (план строится по фактическим параметрам, как в
EXPLAIN), и демо проверяет, что запрос адреса действительно идёт через Tid Range Scan, а B-tree — через индекс.

Нужно: PostgreSQL 17+ (EXPLAIN SERIALIZE), существующая база из RCM_DB (или право её создать) и право создавать в
ней таблицы; суперпользователь, pg_ctl и перезапуск сервера не нужны. Подключение — PGHOST, PGPORT, PGUSER,
PGPASSWORD (README). Каждый запуск работает со своей новой таблицей и своей временной папкой.

Собрано из функций, на которых получены результаты A2–A8 (a2_pg.load, a2_pg.explain, a2_pg.index_size,
a2_runs.absent_keys, a2_runs.sha16), и rcm.py. Проверено на PostgreSQL 18.7 (README).

Copyright (c) 2026 Abu (github.com/tymakbayev). Лицензия MIT, см. LICENSE.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import psycopg

import a2_pg as G
import rcm
from a2_runs import absent_keys, sha16  # те же отсутствующие ключи и сверка полезной нагрузки, что в A2

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / "demo_result.json"
KEY = "k"
EXPLAIN_SAMPLE = 300
WARMUP = 2000


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pct(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(v.mean()),
            "n": int(len(v))}


def connect():
    """Подключение к базе RCM_DB; создать её, только если её нет (тогда нужен доступ к базе postgres и CREATEDB)."""
    try:
        return G.connect()
    except psycopg.OperationalError as e:
        if "does not exist" not in str(e) and "не существует" not in str(e):
            raise
    log(f"базы {G.DB} нет, создаю её")
    G.ensure_db()
    return G.connect()


def lookups(conn, addr, btree_sql: str, keys: list[bytes], digests: list[bytes] | None) -> dict:
    """Каждый ключ ищется обоими способами; порядок чередуется (чётные: адрес первым), чтобы первый не прогревал
    страницу второму в одну сторону. digests is None — ключи отсутствуют, правильный ответ None у обоих."""
    t_addr, t_bt, wrong = [], [], 0
    for i, kb in enumerate(keys):
        u = uuid.UUID(bytes=kb)
        for method in (("rcm", "btree") if i % 2 == 0 else ("btree", "rcm")):
            t0 = time.perf_counter_ns()
            if method == "rcm":
                ra = addr.get(conn, kb)
                t_addr.append((time.perf_counter_ns() - t0) / 1000)
            else:
                rb = conn.execute(btree_sql, (u,), prepare=True).fetchone()
                t_bt.append((time.perf_counter_ns() - t0) / 1000)
        if digests is None:
            wrong += (ra is not None) + (rb is not None)
        else:  # колонки таблицы: seq, k, v, payload; v = 1 у загруженных строк (a2_pg.load)
            ok_a = ra is not None and ra[2] == 1 and sha16(ra[3]) == digests[i]
            ok_b = rb is not None and rb[2] == 1 and sha16(rb[3]) == digests[i]
            wrong += (not ok_a) + (not ok_b) + (ok_a and ok_b and ra != rb)
    no_sql = sum(addr.page_of(kb) is None for kb in keys)  # отдельным проходом: не прогревает адрес перед замером
    return {"address_us": pct(t_addr), "btree_us": pct(t_bt), "wrong": int(wrong), "no_sql": int(no_sql),
            "keys": len(keys)}


def buffers(conn, addr, btree_sql: str, keys: list[bytes]) -> dict:
    """Обращения к буферам на поиск по EXPLAIN (ANALYZE, BUFFERS, SERIALIZE): верхний узел исполнителя, как в A2
    (a2_pg.explain), и отдельно — сериализация результата. Если по адресу ключа нет, запроса нет — 0 обращений.
    Заодно проверяются планы: адрес — Tid Range Scan, B-tree — Index Scan."""
    a, a_ser, b, b_ser, plans = [], [], [], [], {"address": {}, "btree": {}}
    for kb in keys:
        q = addr.query(conn, kb)
        if q is None:
            a.append(0)
            a_ser.append(0)
        else:
            e = G.explain(conn, *q)
            a.append(e["buffers"])
            a_ser.append(e["serialize_buffers"])
            node = e["nodes"][0][1]
            plans["address"][node] = plans["address"].get(node, 0) + 1
        e = G.explain(conn, btree_sql, (uuid.UUID(bytes=kb),))
        b.append(e["buffers"])
        b_ser.append(e["serialize_buffers"])
        node = e["nodes"][0][1]
        plans["btree"][node] = plans["btree"].get(node, 0) + 1
    return {"address": float(np.mean(a)), "btree": float(np.mean(b)), "address_serialize": float(np.mean(a_ser)),
            "btree_serialize": float(np.mean(b_ser)), "plans": plans, "n": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser(description="RCM: вычисляемый адрес против B-tree на одной таблице")
    ap.add_argument("--rows", type=int, default=1_000_000, help="строк в демо-таблице (по умолчанию 1 000 000)")
    ap.add_argument("--lookups", type=int, default=20_000, help="поисков существующих ключей (по умолчанию 20 000)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1), help="процессов генератора")
    ap.add_argument("--keep", action="store_true", help="не удалять таблицу и файлы после демо")
    a = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    if a.rows <= 0 or a.lookups <= 0 or a.workers <= 0:
        raise SystemExit("--rows, --lookups и --workers должны быть больше нуля")
    if a.lookups > a.rows:
        raise SystemExit("--lookups не может быть больше --rows")

    conn = connect()
    version = int(conn.execute("SHOW server_version_num").fetchone()[0])
    if version < 170000:
        raise SystemExit(f"нужен PostgreSQL 17+ (EXPLAIN SERIALIZE), а сервер {version}")
    conn.execute("SET plan_cache_mode = force_custom_plan")
    table = f"rcm_demo_{os.getpid()}_{secrets.token_hex(3)}"  # своя таблица на каждый запуск
    btree = f"{table}_k_btree"
    work = Path(tempfile.mkdtemp(prefix="rcm_demo_"))  # свои файлы на каждый запуск
    ref, adr = work / "ref", work / "rcm_demo.adr"
    out: dict = {"rows": a.rows, "table": {"name": table}, "server_version_num": version,
                 "settings": {k: conn.execute(f"SHOW {k}").fetchone()[0]
                              for k in ("shared_buffers", "block_size", "plan_cache_mode")},
                 "host": {"platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
                          "python": platform.python_version()}}
    created = False
    try:
        log(f"загрузка {a.rows} строк в {table} (генератор экспериментов)")
        created = True  # a2_pg.load создаёт таблицу первым делом; имя уникально, чужую таблицу оно не задевает
        out["load"] = G.load(conn, table, 0, a.rows, ref, workers=a.workers)
        size, pages = conn.execute("SELECT pg_relation_size(%s::regclass), pg_relation_size(%s::regclass) / "
                                   "current_setting('block_size')::int", (table, table)).fetchone()
        out["table"] |= {"pages": int(pages), "main_bytes": int(size)}

        log("B-tree по ключу (CREATE UNIQUE INDEX + ANALYZE)")
        t0 = time.perf_counter()
        conn.execute(f"CREATE UNIQUE INDEX {btree} ON {table} USING btree ({KEY})")
        t1 = time.perf_counter()
        conn.execute(f"ANALYZE {table}")
        bt = G.index_size(conn, btree)
        out["btree"] = {"index_s": t1 - t0, "analyze_s": time.perf_counter() - t1,
                        "total_s": time.perf_counter() - t0, "bytes": bt["main_bytes"],
                        "bytes_per_row": bt["main_bytes"] / a.rows}

        log("адрес RCM по той же таблице (чтение, проверка уникальности, сборка, проверка всех ключей)")
        t0 = time.perf_counter()
        info = rcm.build(conn, table, KEY, adr)
        out["address"] = {k: info[k] for k in ("bytes", "bytes_per_row", "read_s", "unique_s", "build_s", "verify_s",
                                               "r_bits", "verified_rows")} | {"total_s": time.perf_counter() - t0}
        with rcm.Address.open(adr) as addr:
            addr.check(conn)
            keys = np.load(f"{ref}.keys")
            dig = np.load(f"{ref}.dig")
            idx = np.random.default_rng(2026).choice(a.rows, size=a.lookups, replace=False)
            present = [keys[i].tobytes() for i in idx]
            digests = [dig[i].tobytes() for i in idx]
            absent = absent_keys(keys, max(1, a.lookups // 4), 12)
            btree_sql = f"SELECT * FROM {table} WHERE {KEY} = %s"

            log(f"прогрев: {min(WARMUP, len(present))} поисков каждым способом")
            lookups(conn, addr, btree_sql, present[:WARMUP], digests[:WARMUP])
            log(f"замер: {len(present)} существующих и {len(absent)} отсутствующих ключей")
            out["present"] = lookups(conn, addr, btree_sql, present, digests)
            out["absent"] = lookups(conn, addr, btree_sql, absent, None)
            log(f"EXPLAIN BUFFERS на {min(EXPLAIN_SAMPLE, len(present))} ключах каждого вида")
            out["buffers_present"] = buffers(conn, addr, btree_sql, present[:EXPLAIN_SAMPLE])
            out["buffers_absent"] = buffers(conn, addr, btree_sql, absent[:EXPLAIN_SAMPLE])
    finally:
        if a.keep:
            log(f"оставлены: таблица {table}, файлы {work}")
        else:
            if created:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            shutil.rmtree(work, ignore_errors=True)
    bp_, ba_ = out["buffers_present"]["plans"], out["buffers_absent"]["plans"]
    plans_ok = (set(bp_["address"]) == {"Tid Range Scan"}  # у существующих ключей план адреса обязателен
                and set(ba_["address"]) <= {"Tid Range Scan"}  # у отсутствующих — пусто, если все отсеяны без SQL
                and set(bp_["btree"]) | set(ba_["btree"]) <= {"Index Scan", "Index Only Scan"})
    out["plans_ok"] = plans_ok
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    report(out)
    return 0 if out["present"]["wrong"] == 0 and out["absent"]["wrong"] == 0 and plans_ok else 1


def report(o: dict) -> None:
    p, ab, bp, ba = o["present"], o["absent"], o["buffers_present"], o["buffers_absent"]
    rows = [
        ("Байт на ключ", f"{o['address']['bytes_per_row']:.2f}", f"{o['btree']['bytes_per_row']:.2f}"),
        ("Размер, МБ", f"{o['address']['bytes'] / 2**20:.1f}", f"{o['btree']['bytes'] / 2**20:.1f}"),
        ("Сборка целиком, с", f"{o['address']['total_s']:.1f}", f"{o['btree']['total_s']:.1f}"),
        ("Буферов на поиск: есть ключ", f"{bp['address']:.2f}", f"{bp['btree']:.2f}"),
        ("  + сериализация результата", f"{bp['address_serialize']:.2f}", f"{bp['btree_serialize']:.2f}"),
        ("Буферов на поиск: нет ключа", f"{ba['address']:.3f}", f"{ba['btree']:.2f}"),
        ("Задержка p50, мкс: есть ключ", f"{p['address_us']['p50']:.0f}", f"{p['btree_us']['p50']:.0f}"),
        ("Задержка p95, мкс: есть ключ", f"{p['address_us']['p95']:.0f}", f"{p['btree_us']['p95']:.0f}"),
        ("Задержка p50, мкс: нет ключа", f"{ab['address_us']['p50']:.0f}", f"{ab['btree_us']['p50']:.0f}"),
    ]
    print()
    print(f"Таблица {o['rows']} строк, {o['table']['pages']} страниц; PostgreSQL {o['server_version_num']}, "
          f"shared_buffers {o['settings']['shared_buffers']}, plan_cache_mode {o['settings']['plan_cache_mode']}")
    print(f"{'':32}{'адрес RCM':>12}{'B-tree':>12}")
    for name, x, y in rows:
        print(f"{name:32}{x:>12}{y:>12}")
    print(f"Отсутствующих ключей отсеяно без SQL: {ab['no_sql']} из {ab['keys']} ({ab['no_sql'] / ab['keys']:.1%})")
    print(f"Неверных ответов: {p['wrong'] + ab['wrong']} (сверка с эталоном генератора, оба способа)")
    print(f"Планы, ключ есть: адрес {bp['plans']['address']}, B-tree {bp['plans']['btree']}; "
          f"ключа нет: адрес {ba['plans']['address'] or 'без SQL'}, B-tree {ba['plans']['btree']}"
          + ("" if o["plans_ok"] else "  <- НЕ ТОТ ПЛАН: сравнение задержек недействительно"))
    print("Задержка — в клиенте на Python, тёплый кэш; буферы — только PostgreSQL, без чтения адреса.")
    print(f"Результат: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
