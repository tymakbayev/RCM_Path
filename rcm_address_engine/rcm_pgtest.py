"""Проверка rcm.py на живом PostgreSQL: своя таблица с первичным ключом uuid и все обещания из README.

    python rcm_pgtest.py [--rows 20000]

Создаёт в базе RCM_DB таблицы с уникальными именами rcm_t_<номер>_*, проверяет и удаляет их. Нужны PostgreSQL 17+
(EXPLAIN SERIALIZE) и право создавать таблицы; суперпользователь не нужен. Подключение — PGHOST, PGPORT, PGUSER, PGPASSWORD, RCM_DB.
Код выхода 0 — все проверки прошли.

Copyright (c) 2026 Abu (github.com/tymakbayev). Лицензия MIT, см. LICENSE.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

import a2_pg as G
import rcm

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("OK   " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def refused(fn, exc=Exception) -> str | None:
    """Текст ошибки, если fn() отказала, иначе None."""
    try:
        fn()
    except exc as e:
        return str(e).splitlines()[0][:100]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    a = ap.parse_args()
    if a.rows < 2000:
        raise SystemExit("--rows должно быть не меньше 2000 (выборки по 2000 ключей)")
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    conn = G.connect()
    conn.execute("SET plan_cache_mode = force_custom_plan")
    who = conn.execute("SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user").fetchone()
    print(f"PostgreSQL {conn.execute('SHOW server_version').fetchone()[0]}, база {G.DB}, роль {who[0]}"
          f"{' (суперпользователь)' if who[1] else ' (не суперпользователь)'}")
    p = f"rcm_t_{os.getpid()}_{secrets.token_hex(2)}"
    t, part, view, parent, rls, ints, dup = (f"{p}_{s}" for s in ("items", "part", "view", "parent", "rls", "int", "dup"))
    work = Path(tempfile.mkdtemp(prefix="rcm_pgtest_"))
    created: list[str] = []
    try:
        # своя таблица: первичный ключ uuid (значит, рядом уже есть B-tree), текст, место для HOT-обновлений
        conn.execute(f"CREATE TABLE {t} (id uuid PRIMARY KEY, name text NOT NULL, note text) WITH (fillfactor = 70)")
        created.append(t)
        # без сжатия и TOAST: длинное значение остаётся в строке, и строка уходит на другую страницу
        conn.execute(f"ALTER TABLE {t} ALTER COLUMN note SET STORAGE PLAIN")
        rng = np.random.default_rng(5)
        keys = [uuid.UUID(bytes=rng.bytes(16)) for _ in range(a.rows)]
        with conn.cursor() as cur, cur.copy(f"COPY {t} (id, name) FROM STDIN") as cp:
            for i, k in enumerate(keys):
                cp.write_row((k, f"item-{i}"))
        conn.execute(f"VACUUM (ANALYZE) {t}")
        adr = work / "items.adr"
        info = rcm.build(conn, t, "id", adr)
        check("build: адрес построен и проверен", info["verified_rows"] == a.rows,
              f"{info['bytes_per_row']:.2f} Б/строку, {info['pages']} страниц")
        with rcm.Address.open(adr) as addr:
            addr.check(conn)
            check("check: та же база и таблица", True)
            sample = [keys[i] for i in rng.choice(a.rows, 2000, replace=False)]
            got = [addr.get(conn, k) for k in sample]
            check("get: 2000 ключей найдены, строка своя", all(r is not None and r[0] == k for r, k in zip(got, sample)))
            check("get: по строке uuid и по байтам тот же ответ",
                  addr.get(conn, str(sample[0])) == addr.get(conn, sample[0].bytes) == got[0])
            absent = [uuid.UUID(bytes=rng.bytes(16)) for _ in range(2000)]
            check("get: 2000 отсутствующих -> None", all(addr.get(conn, k) is None for k in absent),
                  f"без SQL {sum(addr.page_of(k) is None for k in absent)} из 2000")
            plans = {G.explain(conn, *addr.query(conn, k))["nodes"][0][1] for k in sample[:200]}
            check("план: Tid Range Scan, хотя рядом есть B-tree первичного ключа", plans == {"Tid Range Scan"}, str(plans))
            bufs = np.mean([G.explain(conn, *addr.query(conn, k))["buffers"] for k in sample[:200]])
            check("буферов на поиск: 1", bufs == 1.0, f"{bufs:.2f}")

            k_hot, k_move = sample[0], sample[1]
            k_new = next(k for k in (uuid.UUID(bytes=rng.bytes(16)) for _ in range(100)) if addr.page_of(k) is None)
            page_before = addr.page_of(k_hot)
            with conn.transaction():  # счётчик сервера за текущую транзакцию: обновление действительно HOT
                conn.execute(f"UPDATE {t} SET note = 'hot' WHERE id = %s", (k_hot,))
                hot = conn.execute("SELECT pg_stat_get_xact_tuples_hot_updated(%s::regclass)", (t,)).fetchone()[0]
            page_now = conn.execute(f"SELECT (ctid::text::point)[0]::int FROM {t} WHERE id = %s", (k_hot,)).fetchone()[0]
            r = addr.get(conn, k_hot)
            check("HOT-обновление (по счётчику сервера): та же страница, строка находится, видна новая версия",
                  hot == 1 and page_now == page_before and r is not None and r[2] == "hot",
                  f"HOT {hot}, страница {page_before} -> {page_now}")
            conn.execute(f"UPDATE {t} SET note = repeat('x', 6000) WHERE id = %s", (k_move,))
            moved = conn.execute(f"SELECT (ctid::text::point)[0]::int FROM {t} WHERE id = %s", (k_move,)).fetchone()[0]
            check("перемещённая строка: без fallback -> None", moved != addr.page_of(k_move) and addr.get(conn, k_move) is None,
                  f"страница {addr.page_of(k_move)} -> {moved}")
            r = addr.get(conn, k_move, fallback=True)
            check("перемещённая строка: fallback находит", r is not None and r[0] == k_move)
            conn.execute(f"INSERT INTO {t} (id, name) VALUES (%s, 'new')", (k_new,))
            check("новая строка (ключ вне адреса): без fallback -> None, с fallback найдена",
                  addr.get(conn, k_new) is None and addr.get(conn, k_new, fallback=True) is not None)
            conn.execute(f"VACUUM FULL {t}")
            e = refused(lambda: addr.check(conn), rcm.StaleAddress)
            check("VACUUM FULL: check() отказывает", e is not None, e or "")

        # то, что статический адрес не поддерживает, должно отвергаться с понятной ошибкой
        conn.execute(f"CREATE TABLE {part} (id uuid NOT NULL) PARTITION BY HASH (id)")
        created.append(part)
        conn.execute(f"CREATE TABLE {part}_0 PARTITION OF {part} FOR VALUES WITH (MODULUS 1, REMAINDER 0)")
        e = refused(lambda: rcm.build(conn, part, "id", work / "p.adr"), rcm.Unsupported)
        check("секционированная таблица отвергнута", e is not None, e or "")
        conn.execute(f"INSERT INTO {part} SELECT id FROM {t} LIMIT 100")
        info_p = rcm.build(conn, f"{part}_0", "id", work / "p0.adr")
        check("отдельная секция поддерживается", info_p["verified_rows"] == 100)
        conn.execute(f"CREATE VIEW {view} AS SELECT * FROM {t}")
        created.append(view)
        e = refused(lambda: rcm.build(conn, view, "id", work / "v.adr"), rcm.Unsupported)
        check("представление отвергнуто", e is not None, e or "")
        conn.execute(f"CREATE TABLE {parent} (id uuid)")
        created.append(parent)
        conn.execute(f"CREATE TABLE {parent}_c () INHERITS ({parent})")
        created.append(f"{parent}_c")
        e = refused(lambda: rcm.build(conn, parent, "id", work / "h.adr"), rcm.Unsupported)
        check("таблица с наследником отвергнута", e is not None, e or "")
        conn.execute(f"CREATE TABLE {rls} (id uuid)")
        created.append(rls)
        conn.execute(f"ALTER TABLE {rls} ENABLE ROW LEVEL SECURITY")
        e = refused(lambda: rcm.build(conn, rls, "id", work / "r.adr"), rcm.Unsupported)
        check("таблица с RLS отвергнута", e is not None, e or "")
        conn.execute(f"CREATE TABLE {ints} (id bigint)")
        created.append(ints)
        conn.execute(f"INSERT INTO {ints} VALUES (1)")
        e = refused(lambda: rcm.build(conn, ints, "id", work / "i.adr"), rcm.Unsupported)
        check("ключ не uuid отвергнут", e is not None, e or "")
        conn.execute(f"CREATE TABLE {dup} (id uuid)")
        created.append(dup)
        conn.execute(f"INSERT INTO {dup} SELECT id FROM {t} LIMIT 10")
        conn.execute(f"INSERT INTO {dup} SELECT id FROM {t} LIMIT 1")
        e = refused(lambda: rcm.build(conn, dup, "id", work / "d.adr"), rcm.Unsupported)
        check("повторяющиеся ключи отвергнуты", e is not None, e or "")
    finally:
        for name in reversed(created):
            kind = conn.execute("SELECT relkind FROM pg_class WHERE oid = to_regclass(%s)", (name,)).fetchone()
            if kind:
                conn.execute(f"DROP {'VIEW' if kind[0] == 'v' else 'TABLE'} IF EXISTS {name} CASCADE")
        shutil.rmtree(work, ignore_errors=True)
    print("ALL OK" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
