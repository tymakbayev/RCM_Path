"""Check that a2_gen.py reproduces the reference arrays byte for byte (results/data_manifest.sha256): keys, payload
digests (sha256[:16]) and payload lengths of every record. The COPY text (seq, v, escaping) is not part of this
check; it was compared with the original generator separately (commit message of the review fixes).

Usage: python verify_corpus.py [small|n1|n10] [--workers 6]
  small  stream positions [0, 100 000)      seconds
  n1     stream positions [0, 1 000 000)    about a minute
  n10    stream positions [0, 10 000 000)   minutes; ~2 GB RAM, ~0.4 GB of temporary files
No PostgreSQL is needed. Exit code 0 = all three arrays match, 1 = mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COUNTS = {"small": 100_000, "n1": 1_000_000, "n10": 10_000_000}


def manifest() -> dict[str, str]:
    out = {}
    for line in (ROOT.parent / "results" / "data_manifest.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            out[name] = digest
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", nargs="?", default="small", choices=sorted(COUNTS))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    want = manifest()
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / a.stream
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(ROOT / "a2_gen.py"), "0", str(COUNTS[a.stream]), "--ref", str(prefix),
                            "--workers", str(a.workers), "--no-stdout"], cwd=str(ROOT))
        if r.returncode != 0:
            print("a2_gen.py failed")
            return 1
        ok = True
        for ext in ("keys", "dig", "len"):
            got = hashlib.sha256(Path(f"{prefix}.{ext}").read_bytes()).hexdigest()
            match = got == want[f"{a.stream}.{ext}"]
            ok &= match
            print(f"{a.stream}.{ext}: {'OK' if match else 'MISMATCH'}  {got}")
    print(f"{'all match' if ok else 'MISMATCH'} ({COUNTS[a.stream]} records, {time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
