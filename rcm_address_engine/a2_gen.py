"""A2 record stream (protocol A2_protocol.md §2): E19/E20 corpus copies -> COPY text rows on stdout + reference arrays.

Usage: a2_gen.py START COUNT --ref PREFIX [--workers 6] [--no-stdout]
  stdout       COPY text rows "seq<TAB>k<TAB>v<TAB>payload" for stream positions [START, START+COUNT); seq = position + 1
  PREFIX.keys  uint8 [COUNT, 16]  uuid bytes of k
  PREFIX.dig   uint8 [COUNT, 16]  sha256(payload utf-8)[:16]
  PREFIX.len   uint32 [COUNT]     payload length in bytes
Copy c of the corpus: corpus.build_corpus(2026 + c, 0.0), epoch "e{c+1}", event ids re-derived, time shifted by c * span.
The corpus code lives in corpus.py (stand-alone copy of the JARVIS E19/E20 generator used by the reference runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import corpus

DAY_US = 86_400_000_000
SEED0 = 2026


def _corpus(copy: int):
    assert corpus.TIME_UNIT_MS == 1  # the reference runs set e19.TIME_UNIT_MS = 1
    atoms, dicts, _ = corpus.build_corpus(SEED0 + copy, 0.0)
    return corpus.to_observations(atoms, dicts["entities"])


_SPAN: int | None = None


def span() -> int:
    global _SPAN
    if _SPAN is None:
        base = _corpus(0)
        _SPAN = max(o.observed_at_us for o in base) - min(o.observed_at_us for o in base) + DAY_US
    return _SPAN


def _shift(t: int | None, d: int) -> int | None:
    return None if t is None else t + d


def copy_records(copy: int, span_us: int) -> list[tuple[bytes, bytes]]:
    """All records of one corpus copy in arrival order: (uuid bytes of k, canonical JSON payload utf-8)."""
    from corpus import derived_event_uuid, memory_id
    epoch, d = f"e{copy + 1}", copy * span_us
    out = []
    for o in _corpus(copy):
        event = derived_event_uuid(o.device_id, epoch, o.device_seq)
        head = {"event_uuid": event, "device_id": o.device_id, "runtime_epoch": epoch, "device_seq": o.device_seq,
                "observed_at_us": o.observed_at_us + d, "occurred_at_us": _shift(o.occurred_at_us, d),
                "source": {"source_uuid": o.source.source_uuid, "kind": o.source.kind.value,
                           "independence_group": o.source.independence_group, "trust": o.source.trust},
                "privacy": o.privacy.value, "retention": o.retention.value}
        for i, c in enumerate(o.claims):
            rec = dict(head, claim_index=i, claim={
                "subject": c.subject, "subject_type": c.subject_type, "predicate": c.predicate,
                "value_kind": c.value.kind.value, "value_raw": c.value.raw, "valid_from_us": _shift(c.valid_from_us, d),
                "modality": c.modality, "confidence_q": c.confidence_q})
            payload = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            out.append((uuid.UUID(memory_id(event, i)).bytes, payload))
    return out


def _worker(args):
    copy, span_us = args
    return copy, copy_records(copy, span_us)


def copy_text(value: bytes) -> bytes:
    """COPY text escaping: JSON has no raw tab/newline/CR (json.dumps escapes control chars); double the backslash."""
    if b"\t" in value or b"\n" in value or b"\r" in value:
        raise ValueError("raw control character in payload")
    return value.replace(b"\\", b"\\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start", type=int)
    ap.add_argument("count", type=int)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-stdout", action="store_true")
    a = ap.parse_args()
    span_us = span()
    keys = np.empty((a.count, 16), np.uint8)
    dig = np.empty((a.count, 16), np.uint8)
    lens = np.empty(a.count, np.uint32)
    out = sys.stdout.buffer
    pos, done, end = 0, 0, a.start + a.count  # pos = stream position of the first record of the next copy
    seen: set[bytes] = set()
    copy = 0
    # skip whole copies before START without building them twice: copy sizes are needed, so build in order
    def windows(pool):  # bounded look-ahead: at most `workers` copies in memory while COPY drains stdout
        c = 0
        while True:
            yield from pool.map(_worker, [(c + i, span_us) for i in range(a.workers)])
            c += a.workers

    with Pool(a.workers) as pool:
        for copy, recs in windows(pool):
            lo, hi = max(a.start, pos), min(end, pos + len(recs))
            for j in range(lo - pos, hi - pos):
                k, payload = recs[j]
                if k in seen:
                    raise SystemExit(f"duplicate key at stream position {pos + j}")
                seen.add(k)
                n = pos + j - a.start
                keys[n] = np.frombuffer(k, np.uint8)
                dig[n] = np.frombuffer(hashlib.sha256(payload).digest()[:16], np.uint8)
                lens[n] = len(payload)
                if not a.no_stdout:
                    out.write(b"%d\t%s\t1\t%s\n" % (pos + j + 1, str(uuid.UUID(bytes=k)).encode(), copy_text(payload)))
                done += 1
            pos += len(recs)
            if pos >= end:
                pool.terminate()
                break
    out.flush()
    if done != a.count:
        raise SystemExit(f"stream ended early: {done} of {a.count}")
    for name, arr in (("keys", keys), ("dig", dig), ("len", lens)):
        with open(f"{a.ref}.{name}", "wb") as f:
            np.save(f, arr)
    print(f"a2_gen: {done} records, copies 0..{copy}, span {span_us} us", file=sys.stderr)


if __name__ == "__main__":
    main()
