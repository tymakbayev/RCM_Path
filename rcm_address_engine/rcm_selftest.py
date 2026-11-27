"""Самопроверка rcm.py без PostgreSQL: разбор COPY binary, сборка и чтение адреса, логика get() на поддельном
подключении, формы SQL. Запуск: python rcm_selftest.py  (код выхода 0: всё верно).
Проверка на работающем PostgreSQL: rcm_pgtest.py."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
from psycopg import sql

import rcm

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("OK   " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def copy_binary(keys: np.ndarray, pages: np.ndarray, flags: int = 0) -> bytes:
    """Поток COPY (FORMAT binary) по описанию формата PostgreSQL: подпись, флаги, длина расширения, кортежи, -1."""
    out = bytearray(rcm.COPY_SIGNATURE + struct.pack(">ii", flags, 0))
    for k, p in zip(keys, pages):
        out += struct.pack(">hi", 2, 16) + bytes(k) + struct.pack(">ii", 4, int(p))
    return bytes(out + struct.pack(">h", -1))


def fake_build(path: Path, keys, pages, n_pages, table="public.items", key_col="id", seed=rcm.SEED) -> dict:
    """Как rcm.build(), но без базы: адрес по массивам и описание рядом."""
    info = rcm.build_from_arrays(path, keys, pages, table_oid=16384, relfilenode=16390, n_pages=n_pages, seed=seed)
    Path(f"{path}.json").write_text(json.dumps({
        "format": rcm.MANIFEST_FORMAT, "database": "test", "table": table, "key_col": key_col, "table_oid": 16384,
        "relfilenode": 16390, "rows": info["rows"], "pages": n_pages, "address_bytes": info["bytes"],
        "address_digest": Path(path).read_bytes()[-32:].hex()}), encoding="utf-8")
    return info


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    """Отвечает на запрос страницы строкой только для ключа на его странице; запоминает запросы."""

    connection = None  # для psycopg.sql ... as_string(): как без подключения (кодировка по умолчанию)

    def __init__(self, table: dict[bytes, int], moved: dict[bytes, int] | None = None, dup: bool = False):
        self.table, self.moved, self.dup, self.calls = table, moved or {}, dup, []

    def execute(self, q, params=(), prepare=None):
        self.calls.append((q, params, prepare))
        if "ctid >=" in q:
            lo, hi, k = params
            page = int(lo.strip("()").split(",")[0])
            assert hi == f"({page + 1},0)"
            real = self.moved.get(k.bytes, self.table.get(k.bytes))
            return FakeResult([("row", str(k))] if real == page else [])
        (k,) = params
        real = self.moved.get(k.bytes, self.table.get(k.bytes))
        rows = [("row", str(k))] * (2 if self.dup else 1) if real is not None else []
        return FakeResult(rows)


def main() -> int:
    rng = np.random.default_rng(7)
    n, n_pages = 200_000, 15_000
    keys = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    pages = np.sort(rng.integers(0, n_pages, size=n)).astype(np.uint32)

    # 1. разбор COPY binary
    k2, p2 = rcm.parse_copy_binary(copy_binary(keys[:5000], pages[:5000]))
    check("COPY binary: ключи и страницы разобраны", np.array_equal(k2, keys[:5000]) and np.array_equal(p2, pages[:5000]))
    k0, p0 = rcm.parse_copy_binary(copy_binary(keys[:0], pages[:0]))
    check("COPY binary: пустой поток", len(k0) == 0 and len(p0) == 0)
    good = copy_binary(keys[:3], pages[:3])
    check("COPY binary: чужая подпись отвергнута", raises(lambda: rcm.parse_copy_binary(b"X" + good[1:]), RuntimeError))
    check("COPY binary: нет концевика", raises(lambda: rcm.parse_copy_binary(good[:-2] + b"\x00\x00"), RuntimeError))
    null_row = bytearray(good[:19]) + struct.pack(">hi", 2, -1) + struct.pack(">ii", 4, 1) + good[19:]
    check("COPY binary: NULL-поле отвергнуто", raises(lambda: rcm.parse_copy_binary(bytes(null_row)), RuntimeError))
    check("COPY binary: критический флаг отвергнут",
          raises(lambda: rcm.parse_copy_binary(copy_binary(keys[:3], pages[:3], flags=1 << 16)), RuntimeError))
    check("COPY binary: отрицательная страница отвергнута",
          raises(lambda: rcm.parse_copy_binary(copy_binary(keys[:3], np.array([1, -5, 2]))), RuntimeError))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 2. сборка с проверкой: каждый ключ -> своя страница
        info = fake_build(tmp / "t.adr", keys, pages, n_pages)
        check("сборка: проверка всех ключей пройдена", info.get("verified_rows") == n, f"{info['bytes_per_row']:.2f} Б/ключ")
        check("сборка: размер адреса < 8 Б/ключ", info["bytes_per_row"] < 8.0)
        b = lambda k, p: rcm.build_from_arrays(tmp / "x.adr", k, p, table_oid=1, relfilenode=1, n_pages=10)
        check("сборка: неуникальные ключи отвергнуты",
              raises(lambda: b(np.vstack([keys[:10], keys[:1]]), np.zeros(11, np.uint32)), rcm.Unsupported))
        check("сборка: пустая таблица отвергнута", raises(lambda: b(keys[:0], pages[:0]), ValueError))
        check("сборка: ключи не uint8[n,16] отвергнуты", raises(lambda: b(keys[:5, :8], pages[:5]), ValueError))
        check("сборка: длина страниц не равна числу ключей", raises(lambda: b(keys[:5], pages[:4]), ValueError))
        check("сборка: отрицательная страница отвергнута", raises(lambda: b(keys[:3], np.array([0, -1, 2])), ValueError))
        check("сборка: страницы не целые отвергнуты", raises(lambda: b(keys[:3], np.array([0.5, 1.0, 2.0])), ValueError))

        with rcm.Address.open(tmp / "t.adr") as addr:
            address_checks(addr, rng, keys, pages, n, n_pages)
        check("with: адрес закрыт при выходе", addr.sa is None)
        fake_build(tmp / "u.adr", keys[:1000], pages[:1000], n_pages)
        m = json.loads(Path(f"{tmp / 'u.adr'}.json").read_text(encoding="utf-8"))
        m["rows"] += 1
        Path(f"{tmp / 'u.adr'}.json").write_text(json.dumps(m), encoding="utf-8")
        check("open: адрес и описание не от одной сборки -> отказ", raises(lambda: rcm.Address.open(tmp / "u.adr"), ValueError))
        rcm.build_from_arrays(tmp / "v.adr", keys[:100], pages[:100], table_oid=1, relfilenode=1, n_pages=10)
        check("open: без описания -> отказ", raises(lambda: rcm.Address.open(tmp / "v.adr"), FileNotFoundError))
        # другая сборка с теми же oid, relfilenode, числом строк и размером файла: подмена файла адреса
        fake_build(tmp / "w.adr", keys[:1000], pages[:1000], n_pages)
        for s in range(1, 50):  # зерно, при котором файл адреса того же размера
            fake_build(tmp / "w2.adr", keys[:1000], pages[:1000], n_pages, seed=rcm.SEED + s)
            if (tmp / "w.adr").stat().st_size == (tmp / "w2.adr").stat().st_size:
                break
        same_size = (tmp / "w.adr").stat().st_size == (tmp / "w2.adr").stat().st_size
        (tmp / "w.adr").write_bytes((tmp / "w2.adr").read_bytes())
        check("open: файл адреса от другой сборки (те же oid, строки, размер) -> отказ",
              same_size and raises(lambda: rcm.Address.open(tmp / "w.adr"), ValueError))

    # 4. имена
    check("ident: схема.таблица", rcm.ident("public.items").as_string(None) == '"public"."items"')
    check("ident: лишние точки отвергнуты", raises(lambda: rcm.ident("a.b.c"), ValueError))
    check("ident: кавычки внутри экранируются", sql.Identifier('x"y').as_string(None) == '"x""y"')
    print("ALL OK" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


def address_checks(addr, rng, keys, pages, n, n_pages) -> None:
    check("открытие: строки, страницы, таблица и ключ из описания",
          addr.rows == n and addr.pages == n_pages and addr.table == "public.items" and addr.key_col == "id")
    sample = rng.choice(n, 3000, replace=False)
    check("page_of: 3000 ключей -> свои страницы", all(addr.page_of(bytes(keys[i])) == int(pages[i]) for i in sample))
    absent = rng.integers(0, 256, size=(20_000, 16), dtype=np.uint8)
    miss = sum(addr.page_of(bytes(a)) is None for a in absent)
    check("page_of: отсутствующие ключи почти всегда отсеиваются без SQL", miss / len(absent) > 0.98,
          f"{miss / len(absent):.4f}")
    u = uuid.UUID(bytes=bytes(keys[sample[0]]))
    check("ключ: UUID, строка и байты дают одну страницу",
          addr.page_of(u) == addr.page_of(str(u)) == addr.page_of(u.bytes) == int(pages[sample[0]]))
    check("ключ: неверная длина отвергнута", raises(lambda: addr.page_of(b"short"), ValueError))

    # 3. get() на поддельном подключении
    table = {bytes(keys[i]): int(pages[i]) for i in sample}
    conn = FakeConn(table)
    found = sum(addr.get(conn, bytes(keys[i])) is not None for i in sample)
    check("get: все 3000 ключей найдены одним запросом страницы", found == len(sample) and len(conn.calls) == len(sample))
    check("get: запросы подготовлены (prepare=True)", all(c[2] is True for c in conn.calls))
    check("get: запрос только к самой таблице (ONLY)", all("FROM ONLY" in c[0] for c in conn.calls))
    conn.calls.clear()
    none_sql = [a for a in absent[:2000] if addr.page_of(bytes(a)) is None][:500]
    res = [addr.get(conn, bytes(a)) for a in none_sql]
    check("get: отсутствующий ключ -> None без SQL", all(r is None for r in res) and not conn.calls)
    mk = bytes(keys[sample[1]])
    moved = FakeConn(table, moved={mk: int(pages[sample[1]]) + 1})
    check("get: перемещённая строка без fallback -> None", addr.get(moved, mk) is None)
    check("get: перемещённая строка с fallback найдена", addr.get(moved, mk, fallback=True) is not None)
    dup = FakeConn(table, moved={mk: int(pages[sample[1]]) + 1}, dup=True)
    check("get: две строки в fallback -> ошибка", raises(lambda: addr.get(dup, mk, fallback=True), RuntimeError))
    one = addr.query(FakeConn({}), u, columns="name")
    gen = addr.query(FakeConn({}), u, columns=(c for c in ("id", "name")))
    check("columns: одна колонка-строка и генератор имён",
          one is not None and one[0].startswith('SELECT "name" FROM') and gen[0].startswith('SELECT "id", "name" FROM'),
          f"{one[0][:24]} | {gen[0][:30]}")
    check("columns: пустой список отвергнут", raises(lambda: addr.query(FakeConn({}), u, columns=[]), ValueError))
    q = addr.query(FakeConn({}), u)
    check("query: SQL и параметры страницы",
          q is not None and 'FROM ONLY "public"."items"' in q[0] and q[1][0] == f"({int(pages[sample[0]])},0)"
          and q[1][2] == u, q[0] if q else "")


if __name__ == "__main__":
    sys.exit(main())
