"""RCM Address Engine: API статического адреса (вариант А эксперимента A2) для своей таблицы PostgreSQL.

Строит вычисляемый адрес для существующей таблицы с ключом uuid и ищет строки по ключу: адрес даёт номер страницы
кучи, PostgreSQL читает эту страницу (Tid Range Scan) и проверяет ключ.

    import psycopg
    import rcm
    conn = psycopg.connect("host=localhost dbname=mydb user=me", autocommit=True)
    info = rcm.build(conn, "public.items", "id", "items.adr")   # один раз; рядом пишется items.adr.json
    with rcm.Address.open("items.adr") as addr:
        addr.check(conn)                                         # та же база и таблица, её не переписали
        row = addr.get(conn, some_uuid)                          # кортеж или None

Поддерживается (иначе build() откажет с объяснением):
- одна обычная heap-таблица (relkind 'r'; отдельная секция подходит), без наследников и без включённой RLS;
- ключ: колонка uuid с уникальными значениями; NULL не адресуются.

Что означает адрес:
- он соответствует снимку таблицы на момент чтения в build(). Строки, вставленные или перемещённые на другую
  страницу после начала чтения, по адресу находить не гарантируется: get(..., fallback=True) тогда ищет обычным
  запросом по ключу.
  Обновления, при которых строка остаётся на своей странице (HOT), адрес не ломают;
- VACUUM FULL, CLUSTER и TRUNCATE переписывают таблицу: check() это обнаружит в момент вызова (он не блокирует
  последующий DDL), адрес нужно пересобрать;
- None от page_of() означает «ключа нет в адресуемом снимке», номер означает страницу-кандидата; это не проверка текущего
  наличия строки в таблице;
- если по ключу есть и другой индекс, планировщик может выбрать его вместо страницы адреса: проверяйте план
  (query() + EXPLAIN) или задайте plan_cache_mode = force_custom_plan, как в demo.py.

Собрано из функций, на которых получены результаты A2-A8: a2_core.build_a / write_a / SidecarA (файл JLMADR01),
разбор COPY binary и запрос страницы из a2_pg (read_kp, Client). Добавлено после эталонных прогонов; проверено на
PostgreSQL 18.7 (rcm_pgtest.py, demo.py) и без базы (rcm_selftest.py), см. README.

Copyright (c) 2026 Abu (github.com/tymakbayev). Лицензия MIT, см. LICENSE.
"""

from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from psycopg import sql

import a2_core as C
import a2_pg as G

SEED = 0xA2A2_0001  # зерно варианта А в A2 (a2_runs.SEED_A)
COPY_SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
MANIFEST_FORMAT = "rcm-address-manifest/1"


class StaleAddress(RuntimeError):
    """Адрес не соответствует таблице: другая база или таблица, таблица переписана (другой relfilenode)."""


class Unsupported(ValueError):
    """Таблица или ключ не подходят для статического адреса (см. «Поддерживается» в описании модуля)."""


# ------------------------------------------------------------------ имена и ключи
def ident(name: str) -> sql.Identifier:
    """'items' или 'schema.items' -> идентификатор. Имена берутся как есть (с учётом регистра), без кавычек."""
    parts = name.split(".")
    if not 1 <= len(parts) <= 2 or not all(parts):
        raise ValueError(f"ожидается 'таблица' или 'схема.таблица', получено {name!r}")
    return sql.Identifier(*parts)


def key_bytes(key) -> bytes:
    """uuid.UUID, строка uuid или 16 байт -> 16 байт ключа."""
    if isinstance(key, uuid.UUID):
        return key.bytes
    if isinstance(key, (bytes, bytearray, memoryview)):
        b = bytes(key)
        if len(b) != 16:
            raise ValueError("ключ в байтах должен быть длиной 16")
        return b
    if isinstance(key, str):
        return uuid.UUID(key).bytes
    raise TypeError(f"ключ должен быть uuid.UUID, строкой uuid или 16 байтами, а не {type(key).__name__}")


def _oid(conn, table: str) -> int:
    row = conn.execute("SELECT to_regclass(%s)::oid", (ident(table).as_string(conn),)).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"таблица {table!r} не найдена")
    return int(row[0])


# ------------------------------------------------------------------ чтение (ключ, страница) через COPY binary
def parse_copy_binary(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Поток COPY (FORMAT binary) из двух полей (uuid, int4) -> (ключи uint8[n,16], страницы uint32[n]).

    Тот же разбор, что a2_pg.read_kp, плюс проверка подписи, флагов, заголовка, длин всех полей, концевика и знака."""
    if len(buf) < 21 or buf[:11] != COPY_SIGNATURE:
        raise RuntimeError("не поток COPY binary")
    if int.from_bytes(buf[11:15], "big") & 0xFFFF0000:
        raise RuntimeError("неизвестные критические флаги заголовка COPY binary")
    if int.from_bytes(buf[15:19], "big") != 0:
        raise RuntimeError("неожиданное расширение заголовка COPY binary")
    if buf[-2:] != b"\xff\xff":
        raise RuntimeError("нет концевика COPY binary")
    dt = np.dtype([("n", ">i2"), ("l1", ">i4"), ("k", "V16"), ("l2", ">i4"), ("p", ">i4")])
    body = len(buf) - 19 - 2
    if body % dt.itemsize:
        raise RuntimeError("неожиданная раскладка COPY binary (NULL или другой тип поля?)")
    n = body // dt.itemsize
    arr = np.frombuffer(buf, dt, count=n, offset=19)
    if n and not ((arr["n"] == 2).all() and (arr["l1"] == 16).all() and (arr["l2"] == 4).all()):
        raise RuntimeError("неожиданная раскладка COPY binary")
    if n and (arr["p"] < 0).any():
        raise RuntimeError("отрицательный номер страницы в COPY binary")
    keys = np.frombuffer(arr["k"].tobytes(), np.uint8).reshape(n, 16)
    return keys, arr["p"].astype(np.uint32)


def read_key_pages(conn, table: str, key_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Все непустые ключи таблицы (без наследников) и номера их страниц кучи (как a2_pg.read_kp)."""
    q = sql.SQL("COPY (SELECT {k}, (ctid::text::point)[0]::int4 FROM ONLY {t} WHERE {k} IS NOT NULL) TO STDOUT "
                "(FORMAT binary)").format(k=sql.Identifier(key_col), t=ident(table))
    buf = bytearray()
    with conn.cursor() as cur, cur.copy(q) as cp:
        for data in cp:
            buf += data
    return parse_copy_binary(bytes(buf))


# ------------------------------------------------------------------ сборка
def build_from_arrays(path, keys: np.ndarray, pages: np.ndarray, *, table_oid: int, relfilenode: int, n_pages: int,
                      seed: int = SEED, verify: bool = True) -> dict:
    """Собрать и записать адрес JLMADR01 по массивам (ключи uint8[n,16], страницы). Без PostgreSQL."""
    keys, pages = np.asarray(keys), np.asarray(pages)
    if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[1] != 16:
        raise ValueError("ключи: ожидается массив uint8 формы [n, 16]")
    if pages.ndim != 1 or len(pages) != len(keys) or not np.issubdtype(pages.dtype, np.integer):
        raise ValueError("страницы: ожидается одномерный целочисленный массив той же длины, что и ключи")
    n = len(keys)
    if n == 0:
        raise ValueError("нет ключей: адрес пустой таблицы не строится")
    if pages.min() < 0 or pages.max() >= 2**32:
        raise ValueError("номер страницы вне диапазона 0 .. 2^32-1")
    t0 = time.perf_counter()
    if len(np.unique(np.ascontiguousarray(keys).view("V16"))) != n:
        raise Unsupported("ключи не уникальны: статический адрес требует уникального ключа")
    t1 = time.perf_counter()
    pages = pages.astype(np.uint32)
    r_bits = G.r_bits_for(max(int(n_pages), int(pages.max()) + 1))
    ca = C.build_a(keys, pages, seed, r_bits)
    info = C.write_a(Path(path), ca, table_oid=table_oid, relfilenode=relfilenode, generation=1, n_rows=n,
                     n_pages=int(n_pages), watermark=0)
    t2 = time.perf_counter()
    info |= {"rows": n, "pages": int(n_pages), "r_bits": r_bits, "bytes_per_row": info["bytes"] / n,
             "unique_s": t1 - t0, "build_s": t2 - t1, "path": str(path)}
    if verify:  # каждый ключ должен вернуть свою страницу
        sa = C.SidecarA(Path(path))
        try:
            bad = int((sa.pages_of(keys) != pages.astype(np.int64)).sum())
        finally:
            sa.close()
        if bad:
            raise RuntimeError(f"проверка сборки: {bad} ключей получили не свою страницу")
        info["verified_rows"] = n
    info["verify_s"] = time.perf_counter() - t2
    return info


def build(conn, table: str, key_col: str, path, *, seed: int = SEED, verify: bool = True) -> dict:
    """Построить адрес таблицы `table` по колонке-ключу `key_col` (uuid) и записать его в файл `path`
    (рядом записывается описание path + '.json': база, таблица, колонка, снимок).

    Читает (ключ, страница) одним снимком REPEATABLE READ под блокировкой ACCESS SHARE: таблицу нельзя переписать,
    пока идёт чтение. Нужен autocommit-режим подключения."""
    if not conn.autocommit:
        raise ValueError("нужно подключение с autocommit=True: сборка открывает свою транзакцию REPEATABLE READ")
    t0 = time.perf_counter()
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        conn.execute(sql.SQL("LOCK TABLE ONLY {t} IN ACCESS SHARE MODE").format(t=ident(table)))
        oid = _oid(conn, table)
        kind, am, rls, has_children = conn.execute(
            "SELECT c.relkind, am.amname, c.relrowsecurity, EXISTS (SELECT 1 FROM pg_inherits WHERE inhparent = c.oid)"
            " FROM pg_class c LEFT JOIN pg_am am ON am.oid = c.relam WHERE c.oid = %s", (oid,)).fetchone()
        if kind != "r" or am != "heap":
            raise Unsupported(f"{table!r}: нужна обычная heap-таблица (relkind 'r'), а это relkind {kind!r}, метод "
                              f"{am!r}; для секционированной таблицы стройте адрес каждой секции отдельно")
        if has_children:
            raise Unsupported(f"{table!r}: у таблицы есть наследники или секции; адрес строится для одной таблицы")
        if rls:
            raise Unsupported(f"{table!r}: включена защита строк (RLS); адрес увидел бы только разрешённые строки")
        typ = conn.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = %s"
                           " AND attname = %s AND NOT attisdropped AND attnum > 0", (oid, key_col)).fetchone()
        if typ is None:
            raise ValueError(f"в таблице {table!r} нет колонки {key_col!r}")
        if typ[0] != "uuid":
            raise Unsupported(f"колонка {key_col!r} имеет тип {typ[0]}, а адрес пока поддерживает только uuid")
        keys, pages = read_key_pages(conn, table, key_col)
        relfilenode, n_pages, database, snapshot = conn.execute(
            "SELECT relfilenode, pg_relation_size(oid) / current_setting('block_size')::int, current_database(),"
            " pg_current_snapshot()::text FROM pg_class WHERE oid = %s", (oid,)).fetchone()
    read_s = time.perf_counter() - t0
    info = build_from_arrays(path, keys, pages, table_oid=oid, relfilenode=int(relfilenode), n_pages=int(n_pages),
                             seed=seed, verify=verify)
    with open(path, "rb") as f:  # BLAKE2b-256 всего файла адреса = его последние 32 байта (a2_core.write_a)
        f.seek(-32, os.SEEK_END)
        digest = f.read(32).hex()
    manifest = {"format": MANIFEST_FORMAT, "database": database, "table": table, "key_col": key_col,
                "table_oid": oid, "relfilenode": int(relfilenode), "rows": info["rows"], "pages": int(n_pages),
                "address_bytes": info["bytes"], "address_digest": digest, "snapshot": snapshot,
                "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    mpath = Path(f"{path}.json")
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, mpath)
    return info | {"read_s": read_s, "total_s": time.perf_counter() - t0, "table": table, "key_col": key_col,
                   "manifest": str(mpath)}


# ------------------------------------------------------------------ поиск
class Address:
    """Открытый адрес JLMADR01 (только чтение, через mmap; время открытия не зависит от числа строк).

    Закрывайте его (with ... или close()) до пересборки того же файла: Windows не заменяет файл, открытый через mmap."""

    def __init__(self, sidecar: C.SidecarA, path: Path, manifest: dict) -> None:
        self.sa, self.path, self.manifest = sidecar, path, manifest
        self._sql: dict = {}

    @classmethod
    def open(cls, path) -> "Address":
        path = Path(path)
        mpath = Path(f"{path}.json")
        if not mpath.exists():
            raise FileNotFoundError(f"нет описания {mpath.name}: адрес строится через rcm.build()")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{mpath.name}: неизвестный формат описания")
        sa = C.SidecarA(path)
        same = (sa.table_oid, sa.relfilenode, sa.n_rows) == (manifest["table_oid"], manifest["relfilenode"],
                                                             manifest["rows"]) and len(sa.mm) == manifest["address_bytes"]
        if not same or bytes(sa.mm[-32:]).hex() != manifest.get("address_digest"):  # O(1): последние 32 байта
            sa.close()
            raise ValueError(f"{path.name} и {mpath.name} не от одной сборки")
        return cls(sa, path, manifest)

    def __enter__(self) -> "Address":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def table(self) -> str:
        return self.manifest["table"]

    @property
    def key_col(self) -> str:
        return self.manifest["key_col"]

    @property
    def rows(self) -> int:
        return int(self.sa.n_rows)

    @property
    def pages(self) -> int:
        return int(self.sa.n_pages)

    def page_of(self, key) -> int | None:
        """Страница-кандидат кучи или None, если ключа нет в адресуемом снимке (тогда SQL не нужен)."""
        return self.sa.page_of(key_bytes(key))

    def check(self, conn) -> None:
        """В момент вызова: та же база, имя ведёт к той же таблице, её не переписали (VACUUM FULL, CLUSTER, TRUNCATE),
        колонка-ключ на месте. Не блокирует последующий DDL; другой кластер с совпавшими числами не распознаёт."""
        database = conn.execute("SELECT current_database()").fetchone()[0]
        if database != self.manifest["database"]:
            raise StaleAddress(f"адрес собран в базе {self.manifest['database']!r}, а подключение к {database!r}")
        oid = _oid(conn, self.table)
        relfilenode = int(conn.execute("SELECT relfilenode FROM pg_class WHERE oid = %s", (oid,)).fetchone()[0])
        if oid != self.sa.table_oid or relfilenode != self.sa.relfilenode:
            raise StaleAddress(f"адрес {self.path.name} собран для oid {self.sa.table_oid} / relfilenode "
                               f"{self.sa.relfilenode}, а сейчас {oid} / {relfilenode}: пересоберите адрес")
        col = conn.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = %s"
                           " AND attname = %s AND NOT attisdropped AND attnum > 0", (oid, self.key_col)).fetchone()
        if col is None or col[0] != "uuid":
            raise StaleAddress(f"колонки-ключа {self.key_col!r} типа uuid больше нет: пересоберите адрес")

    @staticmethod
    def _columns(columns) -> tuple | str:
        """Колонки: "*" для всех, строка для одной, иначе непустая последовательность имён."""
        if columns == "*":
            return "*"
        cols = (columns,) if isinstance(columns, str) else tuple(columns)
        if not cols or not all(isinstance(c, str) and c for c in cols):
            raise ValueError("columns: '*', имя колонки или непустой список имён")
        return cols

    def _queries(self, conn, columns):
        k = self._columns(columns)
        if k not in self._sql:
            cols = sql.SQL("*") if k == "*" else sql.SQL(", ").join(sql.Identifier(c) for c in k)
            t, kc = ident(self.table), sql.Identifier(self.key_col)
            page = sql.SQL("SELECT {c} FROM ONLY {t} WHERE ctid >= %s::tid AND ctid < %s::tid AND {k} = %s").format(
                c=cols, t=t, k=kc)
            by_key = sql.SQL("SELECT {c} FROM ONLY {t} WHERE {k} = %s LIMIT 2").format(c=cols, t=t, k=kc)
            self._sql[k] = (page.as_string(conn), by_key.as_string(conn))
        return self._sql[k]

    def query(self, conn, key, columns="*"):
        """(sql, параметры) запроса страницы для ключа, или None, если по адресу ключа нет. Для EXPLAIN и отладки."""
        kb = key_bytes(key)
        page = self.sa.page_of(kb)
        if page is None:
            return None
        q_page, _ = self._queries(conn, columns)
        return q_page, (G.tid(page), G.tid(page + 1), uuid.UUID(bytes=kb))

    def get(self, conn, key, *, columns="*", fallback: bool = False):
        """Строка с этим ключом (кортеж) или None.

        Без SQL, если ключа нет в адресуемом снимке; иначе одна страница кучи (Tid Range Scan) с проверкой ключа.
        fallback=True: если по адресу строка не нашлась (вставлена или перемещена после сборки), искать обычным
        запросом по ключу; это правильно на изменяющейся таблице, но отсутствующий ключ тогда стоит обычного поиска."""
        kb = key_bytes(key)
        q_page, q_key = self._queries(conn, columns)
        k = uuid.UUID(bytes=kb)
        page = self.sa.page_of(kb)
        if page is not None:
            rows = conn.execute(q_page, (G.tid(page), G.tid(page + 1), k), prepare=True).fetchall()
            if len(rows) > 1:
                raise RuntimeError(f"{len(rows)} видимых строк для одного ключа: ключ не уникален")
            if rows:
                return rows[0]
        if fallback:
            rows = conn.execute(q_key, (k,), prepare=True).fetchall()
            if len(rows) > 1:
                raise RuntimeError(f"{len(rows)} видимых строк для одного ключа: ключ не уникален")
            return rows[0] if rows else None
        return None

    def close(self) -> None:
        if self.sa is not None:
            self.sa.close()
            self.sa = None
