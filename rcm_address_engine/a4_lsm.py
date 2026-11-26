"""A4 addressable delta (protocol A4_protocol.md edition 2): tail log with committed / batch states, runs with their own
JLMADR01 addresses, tiered merges (fanout 4) over run logs, major compaction from the table, generation manifest.

Correctness rests on PostgreSQL: every page is only a candidate, the row is accepted after `k = $1` and visibility.
Under the A4 contract (single writer, readers serialised with writer batches) the page of the visible version of a key
is always among the candidates: tail (committed state + current batch), runs newest first, base.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np

import a2_core as C

LOGREC = struct.Struct("<B16sIQ")  # kind | key | page | global operation number  (29 bytes)
I_, U_, D_, C_, R_ = (ord(c) for c in "IUDCR")  # C = batch committed, R = batch rolled back (outcome markers)
ZERO = bytes(16)
FANOUT = 4
SEED_RUN = 0xA4A4_0000
SEED_BASE = 0xA4B4_0000


def r_bits_needed(n_pages: int) -> int:
    return max(1, math.ceil(math.log2(max(n_pages, 2))))


class RefuseOpen(RuntimeError):
    """A batch without an outcome marker or a torn record: the address must be rebuilt from the table (§2.1)."""


# ------------------------------------------------------------------ tail log
class TailLog:
    """JLMDLOG1. In RAM: committed state key -> page | None (deleted), and the current batch key -> page | None."""

    def __init__(self, path: Path, *, create: bool, single_state: bool = False, ignore_open_batch: bool = False) -> None:
        self.path = path
        self.committed: dict[bytes, int | None] = {}
        self.pending: dict[bytes, int | None] = {}
        self.single_state = single_state  # mutant (a)
        self.committed_ops = 0  # committed I/U/D records in this log
        self.last_opseq = 0
        self.last_batch = 0
        self.offset = 0
        if create:
            self.f = open(path, "wb")
            self.f.flush()
        else:
            self._load(ignore_open_batch)
            self.f = open(path, "ab")
        self._buf: list[bytes] = []

    def _load(self, ignore_open_batch: bool) -> None:
        data = self.path.read_bytes()
        if len(data) % LOGREC.size:
            raise RefuseOpen(f"torn record at the end of {self.path.name}")
        batch: dict[bytes, int | None] = {}
        nrec = 0
        for kind, key, page, opseq in LOGREC.iter_unpack(data):
            if kind in (I_, U_, D_):
                batch[key] = None if kind == D_ else page
                nrec += 1
                self.last_opseq = max(self.last_opseq, opseq)
            elif kind == C_:
                self.committed.update(batch)
                self.committed_ops += nrec
                self.last_batch = page
                batch, nrec = {}, 0
            elif kind == R_:
                self.last_batch = page
                batch, nrec = {}, 0
            else:
                raise RefuseOpen(f"unknown record kind {kind}")
        if batch and not ignore_open_batch:
            raise RefuseOpen(f"{len(batch)} records without an outcome marker in {self.path.name}")
        self.offset = len(data)

    def refresh(self) -> int:
        """Reader side: apply records appended since the last read, up to the last outcome marker."""
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        usable = len(data) - len(data) % LOGREC.size
        batch: dict[bytes, int | None] = {}
        nrec = consumed = 0
        for i, (kind, key, page, opseq) in enumerate(LOGREC.iter_unpack(data[:usable])):
            if kind in (I_, U_, D_):
                batch[key] = None if kind == D_ else page
                nrec += 1
            elif kind == C_:
                self.committed.update(batch)
                self.committed_ops += nrec
                batch, nrec = {}, 0
                consumed = (i + 1) * LOGREC.size
            elif kind == R_:
                batch, nrec = {}, 0
                consumed = (i + 1) * LOGREC.size
        self.offset += consumed
        return consumed // LOGREC.size

    def append(self, kind: int, key: bytes, page: int, opseq: int) -> None:
        self._buf.append(LOGREC.pack(kind, key, page, opseq))
        target = self.committed if self.single_state else self.pending
        target[key] = None if kind == D_ else page
        self.last_opseq = opseq

    def _write(self, marker: int, batch_no: int) -> None:
        self._buf.append(LOGREC.pack(marker, ZERO, batch_no, self.last_opseq))
        self.f.write(b"".join(self._buf))
        self._buf = []
        self.f.flush()
        os.fsync(self.f.fileno())

    def commit(self, batch_no: int) -> int:
        ops = len(self._buf)
        self._write(C_, batch_no)
        self.committed.update(self.pending)
        self.pending.clear()
        self.committed_ops += ops
        self.last_batch = batch_no
        return ops

    def rollback(self, batch_no: int) -> None:
        self._write(R_, batch_no)
        self.pending.clear()  # the committed state is untouched (R1); mutant (a) already overwrote it
        self.last_batch = batch_no

    def candidates(self, key: bytes, with_pending: bool) -> list[int]:
        out = []
        if with_pending and key in self.pending and self.pending[key] is not None:
            out.append(self.pending[key])
        p = self.committed.get(key)
        if p is not None:
            out.append(p)
        return out

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()

    def nbytes(self) -> int:
        return self.path.stat().st_size + LOGREC.size * len(self._buf)


# ------------------------------------------------------------------ runs
class Run:
    """Immutable run: sorted log (key 16 | page u32) + JLMADR01 address over the same keys."""

    def __init__(self, d: Path, run_id: int, tier: int) -> None:
        self.dir, self.id, self.tier = d, run_id, tier
        self.adr_path = d / f"run{run_id:06d}.adr"
        self.log_path = d / f"run{run_id:06d}.log"
        self.addr: C.SidecarA | None = None

    @classmethod
    def build(cls, d: Path, run_id: int, tier: int, entries: dict[bytes, int]) -> tuple["Run", dict]:
        r = cls(d, run_id, tier)
        t0 = time.perf_counter()
        keys_sorted = sorted(entries)
        n = len(keys_sorted)
        keys = np.frombuffer(b"".join(keys_sorted), np.uint8).reshape(n, 16) if n else np.zeros((0, 16), np.uint8)
        pages = np.fromiter((entries[k] for k in keys_sorted), np.uint32, n)
        log = np.zeros(n, dtype=[("k", "V16"), ("p", "<u4")])
        if n:
            log["k"] = keys.view("V16").ravel()
            log["p"] = pages
        tmp = r.log_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(log.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, r.log_path)
        r_bits = r_bits_needed(int(pages.max()) + 1 if n else 2)
        ca = C.build_a(keys, pages, SEED_RUN ^ run_id, r_bits) if n else C.build_a(
            np.zeros((1, 16), np.uint8), np.zeros(1, np.uint32), SEED_RUN ^ run_id, 1)
        info = C.write_a(r.adr_path, ca, table_oid=0, relfilenode=0, generation=run_id, n_rows=n,
                         n_pages=int(pages.max()) + 1 if n else 0, watermark=0)
        r.open()
        return r, {"run": run_id, "tier": tier, "keys": n, "adr_bytes": info["bytes"], "log_bytes": r.log_path.stat().st_size,
                   "build_s": time.perf_counter() - t0}

    def open(self) -> None:
        self.addr = C.SidecarA(self.adr_path)
        self.n = self.addr.n_rows

    def entries(self) -> dict[bytes, int]:
        raw = self.log_path.read_bytes()
        log = np.frombuffer(raw, dtype=[("k", "V16"), ("p", "<u4")])
        kb = log["k"].tobytes()
        return {kb[i * 16:(i + 1) * 16]: int(p) for i, p in enumerate(log["p"].tolist())}

    def page_of(self, key: bytes) -> int | None:
        return self.addr.page_of(key) if self.n else None

    def nbytes(self) -> int:
        return self.adr_path.stat().st_size + self.log_path.stat().st_size

    def close(self) -> None:
        if self.addr is not None:
            self.addr.close()
            self.addr = None

    def remove(self) -> None:
        self.close()
        for p in (self.adr_path, self.log_path):
            p.unlink(missing_ok=True)


# ------------------------------------------------------------------ the level set
class AddressLSM:
    def __init__(self, d: Path, *, t_run: int = 30_000, f_major: float = 0.1, mutant: str | None = None) -> None:
        self.dir = d
        self.t_run, self.f_major, self.mutant = t_run, f_major, mutant
        self.base: C.SidecarA | None = None
        self.base_path: Path | None = None
        self.runs: list[Run] = []  # newest first
        self.tail: TailLog | None = None
        self.gen = 0
        self.next_run = 1
        self.opseq = 0
        self.batch_no = 0
        self.ops_since_major = 0
        self.n0 = 0
        self.events: list[dict] = []
        self.peak_coexist_bytes = 0

    # ---- manifest
    @property
    def manifest(self) -> Path:
        return self.dir / "a4.manifest.json"

    def publish(self) -> None:
        self.gen += 1
        m = {"gen": self.gen, "base": self.base_path.name, "runs": [[r.id, r.tier] for r in self.runs],
             "tail": self.tail.path.name, "opseq": self.opseq, "batch_no": self.batch_no,
             "ops_since_major": self.ops_since_major, "n0": self.n0, "next_run": self.next_run}
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(m), encoding="utf-8")
        os.replace(tmp, self.manifest)

    @classmethod
    def open(cls, d: Path, *, ignore_open_batch: bool = False, **kw) -> "AddressLSM":
        m = json.loads((d / "a4.manifest.json").read_text(encoding="utf-8"))
        s = cls(d, **kw)
        s.gen, s.opseq, s.batch_no = m["gen"], m["opseq"], m["batch_no"]
        s.ops_since_major, s.n0, s.next_run = m["ops_since_major"], m["n0"], m["next_run"]
        s.base_path = d / m["base"]
        s.base = C.SidecarA(s.base_path)
        s.runs = []
        for rid, tier in m["runs"]:
            r = Run(d, rid, tier)
            r.open()
            s.runs.append(r)
        s.tail = TailLog(d / m["tail"], create=False, ignore_open_batch=ignore_open_batch)
        s.opseq = max(s.opseq, s.tail.last_opseq)
        s.batch_no = max(s.batch_no, s.tail.last_batch)
        s.ops_since_major = m["ops_since_major"] + s.tail.committed_ops  # manifest value is as of its publish
        return s

    @classmethod
    def create(cls, d: Path, keys: np.ndarray, pages: np.ndarray, n_live: int, **kw) -> "AddressLSM":
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("*.adr")) + list(d.glob("*.log")) + list(d.glob("a4.manifest.*")):
            old.unlink()
        s = cls(d, **kw)
        s.tail = TailLog(d / "tail000001.log", create=True, single_state=s.mutant == "a")
        r_bits = r_bits_needed(int(pages.max()) + 1)
        ca = C.build_a(keys, pages, SEED_BASE ^ 1, r_bits)
        s.base_path = d / "base000001.adr"
        C.write_a(s.base_path, ca, table_oid=0, relfilenode=0, generation=1, n_rows=len(keys),
                  n_pages=int(pages.max()) + 1, watermark=0)
        s.base = C.SidecarA(s.base_path)
        s.n0 = n_live
        s.publish()
        return s

    # ---- lookup
    def candidates(self, key: bytes, writer: bool) -> list[int]:
        out = self.tail.candidates(key, writer)
        for r in self.runs:
            p = r.page_of(key)
            if p is not None:
                out.append(p)
        p = self.base.page_of(key)
        if p is not None:
            out.append(p)
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # ---- maintenance
    def new_tail(self) -> None:
        old = self.tail
        self.tail = TailLog(self.dir / f"tail{self.gen + 1:06d}.log", create=True, single_state=self.mutant == "a")
        return old

    def seal(self) -> dict:
        """Tail -> run (committed state only), then tiered merges; publish one generation (§2.2-§2.5)."""
        t0 = time.perf_counter()
        entries = {k: p for k, p in self.tail.committed.items() if p is not None}
        run, info = Run.build(self.dir, self.next_run, 0, entries)
        self.next_run += 1
        self.runs.insert(0, run)
        merges = []
        tier = 0
        while True:
            same = [r for r in self.runs if r.tier == tier]
            if len(same) < FANOUT:
                break
            merges.append(self.merge(same, tier + 1))
            tier += 1
        old = self.new_tail()
        self.publish()
        old.close()
        coexist = self.nbytes() + old.path.stat().st_size
        old.path.unlink(missing_ok=True)
        self.peak_coexist_bytes = max(self.peak_coexist_bytes, coexist)
        ev = {"kind": "seal", "after_op": self.opseq, "run": info, "merges": merges, "pause_s": time.perf_counter() - t0,
              "levels": len(self.runs) + 2}
        self.events.append(ev)
        return ev

    def merge(self, runs: list[Run], tier: int) -> dict:
        t0 = time.perf_counter()
        order = sorted(runs, key=lambda r: r.id)  # oldest -> newest: the newest entry wins
        if self.mutant == "b":
            order = order[::-1]  # mutant: the oldest entry wins
        entries: dict[bytes, int] = {}
        read = 0
        for r in order:
            entries.update(r.entries())
            read += r.log_path.stat().st_size  # a merge reads run logs only
        new, info = Run.build(self.dir, self.next_run, tier, entries)
        self.next_run += 1
        coexist = self.nbytes() + new.nbytes()
        self.peak_coexist_bytes = max(self.peak_coexist_bytes, coexist)
        ids = {r.id for r in runs}
        pos = min(i for i, r in enumerate(self.runs) if r.id in ids)
        self.runs = [r for r in self.runs if r.id not in ids]
        self.runs.insert(pos, new)
        for r in runs:
            r.remove()
        return {"tier": tier, "inputs": len(runs), "keys": info["keys"], "read_bytes": read,
                "written_bytes": info["adr_bytes"] + info["log_bytes"], "s": time.perf_counter() - t0}

    def major(self, keys: np.ndarray, pages: np.ndarray, n_live: int, read_bytes: int, read_s: float) -> dict:
        """Base rebuilt from the table export (keys, pages); all runs and logs dropped (§2.4)."""
        t0 = time.perf_counter()
        r_bits = r_bits_needed(int(pages.max()) + 1)
        ca = C.build_a(keys, pages, SEED_BASE ^ (self.gen + 1), r_bits)
        path = self.dir / f"base{self.gen + 1:06d}.adr"
        info = C.write_a(path, ca, table_oid=0, relfilenode=0, generation=self.gen + 1, n_rows=len(keys),
                         n_pages=int(pages.max()) + 1, watermark=self.opseq)
        old_base, old_base_path, old_runs = self.base, self.base_path, self.runs
        self.peak_coexist_bytes = max(self.peak_coexist_bytes, self.nbytes() + info["bytes"])
        self.base_path = path
        self.base = C.SidecarA(path)
        self.runs = []
        old_tail = self.new_tail()
        self.ops_since_major = 0
        self.n0 = n_live
        self.publish()
        if old_base is not None:
            old_base.close()
            old_base_path.unlink(missing_ok=True)
        for r in old_runs:
            r.remove()
        if old_tail is not None:
            old_tail.close()
            old_tail.path.unlink(missing_ok=True)
        ev = {"kind": "major", "after_op": self.opseq, "keys": len(keys), "base_bytes": info["bytes"],
              "read_bytes": read_bytes, "read_s": read_s, "build_s": time.perf_counter() - t0,
              "pause_s": read_s + time.perf_counter() - t0}
        self.events.append(ev)
        return ev

    def due(self) -> str | None:
        if self.ops_since_major >= self.f_major * self.n0:
            return "major"
        if self.tail.committed_ops >= self.t_run:
            return "seal"
        return None

    # ---- accounting
    def nbytes(self) -> int:
        total = self.base_path.stat().st_size if self.base_path else 0
        total += sum(r.nbytes() for r in self.runs)
        if self.tail is not None:
            total += self.tail.nbytes()
        if self.manifest.exists():
            total += self.manifest.stat().st_size
        return total

    def close(self) -> None:
        if self.base is not None:
            self.base.close()
        for r in self.runs:
            r.close()
        if self.tail is not None:
            self.tail.close()
