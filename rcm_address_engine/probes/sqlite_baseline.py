"""Baseline probe: bytes per record of a SQLite B-tree that maps a key to an 8-byte address (1M records)."""
import hashlib, os, sqlite3, struct, sys, time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000


def key16(i):
    return hashlib.blake2b(struct.pack("<Q", i), digest_size=16).digest()


def run(name, schema, rows, vacuum):
    path = f"{name}.db"
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.execute("PRAGMA page_size=4096")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute(schema)
    t = time.perf_counter()
    db.executemany("INSERT INTO t VALUES (?, ?)", rows)
    db.commit()
    build = time.perf_counter() - t
    if vacuum:
        db.execute("VACUUM")
    db.close()
    size = os.path.getsize(path)
    print(f"{name:34s} {size / N:6.2f} bytes/record  file {size / 2**20:7.1f} MiB  build {build:5.1f} s")
    os.remove(path)


blob = "CREATE TABLE t(k BLOB PRIMARY KEY, a INTEGER) WITHOUT ROWID"
integer = "CREATE TABLE t(k INTEGER PRIMARY KEY, a INTEGER)"
addr = lambda i: (i * 2654435761) % (1 << 40)  # a 40-bit physical offset
for vacuum in (False, True):
    tag = "vacuum" if vacuum else "as-built"
    run(f"key16 random order, {tag}", blob, ((key16(i), addr(i)) for i in range(N)), vacuum)
    run(f"int64 random keys, {tag}", integer,
        ((int.from_bytes(key16(i)[:8], "little") >> 1, addr(i)) for i in range(N)), vacuum)
