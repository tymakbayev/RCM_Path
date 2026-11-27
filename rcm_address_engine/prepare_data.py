"""Подготовка входных файлов, которые скрипты экспериментов не создают сами, и их сверка с прогонами (sha256 в
results/data_manifest.sha256).

    python prepare_data.py            # data/dyn.copy + data/dyn.{keys,dig,len}: нужно для A2 dynamics, A3, A4-A8
    python prepare_data.py --replay   # плюс data/ref_a4_n1.* и ref_a4_n10.*: нужно для replay в A5 и A6

dyn: позиции потока [10 000 000, 10 400 000): свежие ключи и полезная нагрузка для вставок в динамике (в эталонных
прогонах файл создавался вручную этой же командой генератора: a2_gen.py 10000000 400000 --ref data/dyn > dyn.copy).
ref_a4_n1 / ref_a4_n10: позиции [0, 1 000 000) и [0, 10 000 000); их оставлял прогон A4 dynamics.
Генерация идёт с начала потока, поэтому занимает минуты (dyn и n10 около 3 минут каждый на эталонном стенде).
PostgreSQL не нужен. Код выхода 0: все файлы совпали с эталоном.

Copyright (c) 2026 Abu (github.com/tymakbayev). Лицензия MIT, см. LICENSE.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
JOBS = {  # имя -> (старт, число, префикс, имя в манифесте, писать ли COPY-текст в .copy)
    "dyn": (10_000_000, 400_000, "dyn", "dyn", True),
    "ref_a4_n1": (0, 1_000_000, "ref_a4_n1", "n1", False),
    "ref_a4_n10": (0, 10_000_000, "ref_a4_n10", "n10", False),
}


def manifest() -> dict[str, str]:
    out = {}
    for line in (ROOT.parent / "results" / "data_manifest.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            out[name] = digest
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def make(name: str, want: dict[str, str], workers: int) -> bool:
    start, count, prefix, mname, with_copy = JOBS[name]
    files = [(DATA / f"{prefix}.{e}", f"{mname}.{e}") for e in ("keys", "dig", "len")]
    if with_copy:
        files.append((DATA / f"{prefix}.copy", f"{mname}.copy"))
    if all(p.exists() and sha(p) == want[m] for p, m in files):
        print(f"{name}: уже есть и совпадает с эталоном")
        return True
    print(f"{name}: генерация позиций [{start}, {start + count}) ...", flush=True)
    cmd = [sys.executable, str(ROOT / "a2_gen.py"), str(start), str(count), "--ref", str(DATA / prefix),
           "--workers", str(workers)]
    if with_copy:
        tmp = DATA / f"{prefix}.copy.tmp"
        with open(tmp, "wb") as out:
            r = subprocess.run(cmd, stdout=out, cwd=str(ROOT))
        if r.returncode == 0:
            tmp.replace(DATA / f"{prefix}.copy")
    else:
        r = subprocess.run(cmd + ["--no-stdout"], cwd=str(ROOT))
    if r.returncode != 0:
        print(f"{name}: a2_gen.py завершился с ошибкой")
        return False
    ok = True
    for p, m in files:
        match = sha(p) == want[m]
        ok &= match
        print(f"  {p.name}: {'совпадает с эталоном' if match else 'НЕ СОВПАДАЕТ с эталоном'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Подготовка входных файлов экспериментов")
    ap.add_argument("--replay", action="store_true", help="ещё ref_a4_n1 и ref_a4_n10 для replay в A5 и A6")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    DATA.mkdir(exist_ok=True)
    want = manifest()
    names = ["dyn"] + (["ref_a4_n1", "ref_a4_n10"] if a.replay else [])
    ok = all([make(n, want, a.workers) for n in names])
    print("готово, всё совпадает с эталоном" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
