"""A6: the A5 level set over a hash-PARTITIONED PostgreSQL table (protocol A6_protocol.md edition 2).

Differences from a5_lsm (everything else is inherited unchanged):
  * base section j == table partition j; the base section of a key is g(k) = PostgreSQL's hash-partition routing
    (a6_pghash), not the top bits of h. Runs keep their own sections by the top bits of h.
  * the round reads ONE PARTITION per step (COPY of k + ctid page), builds and publishes its base section in the same
    step; no work files. The round state lives in the manifest and a restart RESUMES the round (§2.5).
  * file accounting for S5: every file opened for writing / removed / renamed during a round step is recorded through
    an audit hook (§5).
Mutants (§4; code = protocol letter): e=(д) pre-W runs dropped at the first section; f2=(е′) the last partition is not
rebuilt; m=(м) step s publishes as section (s+1) mod S; n=(н) client partition = top bits of h; o=(о) after a restart W := current
op and every run becomes pre-W; p=(п) run candidates are sent to partition (g+1) mod S; r=(р) a resumed round skips the next
step; inherited: a..d=(а)..(г) of A4, g=(ж), z=(з), i=(и), k=(к), k2=(к′), l=(л) of A5.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from functools import lru_cache
from pathlib import Path

import numpy as np

import a2_core as C
import a5_lsm as L5
import a6_pghash as PH
from a4_lsm import RefuseOpen, TailLog  # noqa: F401  (re-exported)
from a5_lsm import SEED_BASE, SIGMA, MergeJob, Run, bits_for, build_adr, sect_int  # noqa: F401

MANIFEST = "a6.manifest.json"


# ------------------------------------------------------------------ file accounting (S5)
class _Audit:
    active = False
    written: list[str] = []
    removed: list[str] = []
    installed = False

    @classmethod
    def hook(cls, event, args):
        if not cls.active:
            return
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if not isinstance(path, (str, bytes, os.PathLike)):
                return
            w = (mode is not None and any(c in str(mode) for c in "wax+")) or \
                (mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if w:
                cls.written.append(os.fspath(path) if not isinstance(path, bytes) else path.decode())
        elif event == "os.remove":
            cls.removed.append(os.fspath(args[0]))
        elif event == "os.rename":
            cls.removed.append(os.fspath(args[0]))
            cls.written.append(os.fspath(args[1]))

    @classmethod
    def start(cls):
        if not cls.installed:
            sys.addaudithook(cls.hook)
            cls.installed = True
        cls.written, cls.removed = [], []
        cls.active = True

    @classmethod
    def stop(cls) -> tuple[list[str], list[str]]:
        cls.active = False
        return [norm(p) for p in cls.written], [norm(p) for p in cls.removed]


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


class AddressLSM6(L5.AddressLSM):
    def __init__(self, d: Path, **kw) -> None:
        super().__init__(d, **kw)
        self.resumed: dict = {}
        self.trace_mem = False
        self.s5_violations: list[dict] = []
        self.step_hook = None  # R1: witness selection inside the step; the step's pairs never outlive the step
        self.n_live: int | None = None  # R9: exact live-row count kept by the writer (committed INSERT / DELETE)
        self.crash_before_end = False  # scenario hook (Codex 241 R1): stop after the last publication, before round_end
        self._part_cache = lru_cache(maxsize=1 << 18)(self._part_uncached)

    # ---- the partition function (§2.3)
    @property
    def n_parts(self) -> int:
        return 1 << self.bits_base

    def _part_uncached(self, key: bytes) -> int:
        if self.mutant == "n":
            return sect_int(self.hz.one(key), self.bits_base)
        return PH.part_of(key, self.n_parts)

    def part(self, key: bytes) -> int:
        return self._part_cache(key)

    def part_np(self, keys: np.ndarray) -> np.ndarray:
        if self.mutant == "n":
            return L5.sect_np(self.hz.np(keys), self.bits_base)
        return PH.part_of_np(keys, self.n_parts)

    # ---- files
    @property
    def manifest(self) -> Path:
        return self.dir / MANIFEST

    def publish(self) -> None:
        self.gen += 1
        rd = self.round
        m = {"gen": self.gen, "bits_base": self.bits_base, "base": [[g, n] for g, n, _ in self.base],
             "runs": [r.meta() for r in self.runs], "tail": self.tail.path.name, "opseq": self.opseq,
             "batch_no": self.batch_no, "ops_total": self.ops_total, "round_start_ops": self.round_start_ops,
             "n0": self.n0, "next_id": self.next_id, "next_prio": self.next_prio, "tail_no": self.tail_no,
             "base_gen": self.base_gen, "tail_ops": self.tail.committed_ops,
             "job": self.job.id if self.job else None, "n_live": self.n_live,
             "round": None if rd is None else {k: rd[k] for k in ("W", "W_ops", "s", "gen", "t0_op", "steps", "rows",
                                                                 "max_rows", "pre_dropped")}}
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(m), encoding="utf-8")
        os.replace(tmp, self.manifest)
        self.track()

    @classmethod
    def create(cls, d: Path, keys: np.ndarray, pages: np.ndarray, n_live: int, **kw) -> "AddressLSM6":
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
        s = cls(d, **kw)
        s.tail = TailLog(d / f"tail{s.tail_no:06d}.log", create=True, single_state=s.mutant == "a")
        s.bits_base = s.s_base_bits if s.s_base_bits is not None else bits_for(n_live, s.sigma)
        sec = s.part_np(keys)
        order = np.argsort(sec, kind="stable")
        bounds = np.searchsorted(sec[order], np.arange(s.n_parts + 1))
        for j in range(s.n_parts):
            idx = order[bounds[j]:bounds[j + 1]]
            a, _ = build_adr(s.base_path(j, 1), keys[idx], pages[idx], SEED_BASE ^ (1 << 12) ^ j, 1)
            s.base.append([1, int(len(idx)), a])
        s.n0 = n_live
        s.n_live = n_live
        s.publish()
        return s

    @classmethod
    def open(cls, d: Path, *, ignore_open_batch: bool = False, writer: bool = False, **kw) -> "AddressLSM6":
        """Reopen from the manifest (§2.5, R1): an active tier merge is abandoned (inputs stay, unpinned); an
        unfinished round RESUMES at its next unpublished step with the same W and the same pre-W classes. Only the
        writer (writer=True) removes files the manifest does not reference; a reader never deletes anything."""
        m = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
        s = cls(d, **kw)
        for k in ("gen", "bits_base", "opseq", "batch_no", "ops_total", "round_start_ops", "n0", "next_id", "next_prio",
                  "tail_no", "base_gen"):
            setattr(s, k, m[k])
        s.base = [[g, n, C.SidecarA(s.base_path(j, g)) if n else None] for j, (g, n) in enumerate(m["base"])]
        s.runs = [Run.from_meta(d, r) for r in m["runs"]]
        s.tail = TailLog(d / m["tail"], create=False, ignore_open_batch=ignore_open_batch)
        s.opseq = max(s.opseq, s.tail.last_opseq)
        s.batch_no = max(s.batch_no, s.tail.last_batch)
        s.ops_total = m["ops_total"] + max(0, s.tail.committed_ops - m["tail_ops"])
        s.n_live = m.get("n_live")
        s.abandoned = {}
        if m["job"] is not None and writer:  # R2: only the writer cleans up; a reader deletes nothing
            for p in d.glob(f"r{m['job']:06d}*"):
                p.unlink(missing_ok=True)
            s.abandoned["merge"] = m["job"]
        if m["round"] is None:
            for r in s.runs:
                r.pre = False
        else:
            rd = dict(m["round"])
            if s.mutant == "o":  # mutant (o): W replaced by the current op, every run becomes pre-W
                rd["W"], rd["W_ops"] = s.opseq, s.ops_total
                for r in s.runs:
                    r.pre = True
            if s.mutant == "r":  # mutant (r): the resumed round skips its next step
                rd["s"] += 1
            s.round = rd
            s.resumed = {"round_W": rd["W"], "next_step": rd["s"]}
        # files the manifest does not reference (an unpublished merge output or section written before a crash)
        junk = s.stray_files() if writer else []
        for name in junk:
            (d / name).unlink(missing_ok=True)
        if junk:
            s.abandoned["unreferenced_files"] = junk
        return s

    # ---- lookup: (partition, page) targets, newest first
    def targets(self, key: bytes, writer: bool) -> list[tuple[int, int]]:
        g = self.part(key)
        out = [(g, p) for p in self.tail.candidates(key, writer)]
        h = self.hz.one(key)
        job = self.job
        use_out = job is not None and (sect_int(h, job.bits) < job.done or self.mutant == "z")
        gr = (g + 1) % self.n_parts if self.mutant == "p" else g
        emitted = False
        for r in self.runs:
            if job is not None and r.id in job.input_ids:
                if use_out:
                    if not emitted:
                        emitted = True
                        p = job.page_of(key, h)
                        if p is not None:
                            out.append((gr, p))
                    continue
            p = r.page_of(key, h)
            if p is not None:
                out.append((gr, p))
        b = self.base[g][2]
        if b is not None:
            p = b.page_of(key)
            if p is not None:
                out.append((g, p))
        seen, uniq = set(), []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def candidates(self, key: bytes, writer: bool) -> list[int]:
        return [p for _, p in self.targets(key, writer)]

    # ---- the round by partitions (§2.5)
    def round_start(self, p0: int | None = None) -> dict:
        t0 = time.perf_counter()
        ev = {"kind": "round_start", "after_op": self.opseq}
        if self.mutant != "i":
            ev["seal"] = self.seal(forced=True)
        for r in self.runs:
            r.pre = True
        self.round = {"W": self.opseq, "W_ops": self.ops_total, "s": 0, "steps": 0, "rows": 0, "max_rows": 0,
                      "t0_op": self.opseq, "pre_dropped": False,
                      # new section files never reuse a generation already on disk
                      "gen": max(g for g, _, _ in self.base) + 1}
        self.round_pending = False
        self.publish()
        ev["s"] = time.perf_counter() - t0
        return ev

    def allowed_step_files(self, target: int, gen: int, old_gen: int) -> tuple[set[str], set[str]]:
        new = self.base_path(target, gen)
        w = {norm(new), norm(new.with_suffix(".tmp")), norm(self.manifest), norm(self.manifest.with_suffix(".tmp"))}
        rm = {norm(self.base_path(target, old_gen)), norm(new.with_suffix(".tmp")),
              norm(self.manifest.with_suffix(".tmp"))}
        return w, rm

    def round_finish_only(self) -> dict:
        """After a restart between the last publication and round_end: only the round end remains (Codex 241 R1)."""
        t0 = time.perf_counter()
        ev = {"kind": "round_end_only", "round_end": self.audited_end(), "s": 0.0}
        ev["s"] = time.perf_counter() - t0
        return ev

    def audited_end(self) -> dict:
        """round_end under the file audit (Codex 241 R7): it may only publish the manifest and delete pre-W runs."""
        pre = [r for r in self.runs if r.pre] + list(getattr(self, "_dropped", []))
        allowed_rm = {norm(f) for r in pre for f in r.files()} | {norm(self.manifest.with_suffix(".tmp"))}
        if self.mutant == "i":
            allowed_rm.add(norm(self.tail.path))
        allowed_w = {norm(self.manifest), norm(self.manifest.with_suffix(".tmp"))}
        _Audit.start()
        try:
            summary = self.round_end()
        finally:
            written, removed = _Audit.stop()
        if self.mutant == "i":  # the mutant opens a new tail file
            allowed_w.add(norm(self.tail.path))
        bad = sorted((set(written) - allowed_w) | (set(removed) - allowed_rm))
        summary["s5_bad_files"] = bad
        if bad:
            self.s5_violations.append({"round_end": True, "files": bad})
        return summary

    def round_step(self, read_fn) -> dict:
        rd = self.round
        s = rd["s"]
        if s > self.n_parts:
            raise ValueError(f"round step {s} out of range 0..{self.n_parts}")
        if s == self.n_parts:
            return self.round_finish_only()
        mem0 = 0
        if self.trace_mem:
            tracemalloc.reset_peak()
            mem0 = tracemalloc.get_traced_memory()[0]
        _Audit.start()
        try:
            t0 = time.perf_counter()
            keys, pages, info = read_fn(s)
            t_read = time.perf_counter() - t0
            t1 = time.perf_counter()
            target = (s + 1) % self.n_parts if self.mutant == "m" else s
            gen = rd["gen"]
            a, nb = build_adr(self.base_path(target, gen), keys, pages, SEED_BASE ^ (gen << 12) ^ target, gen)
            old = self.base[target]
            self.base[target] = [gen, int(len(keys)), a]
            rd["s"] += 1
            rd["steps"] += 1
            rd["rows"] += int(len(keys))
            rd["max_rows"] = max(rd["max_rows"], int(len(keys)))
            last = rd["s"] >= self.n_parts or (self.mutant == "f2" and rd["s"] == self.n_parts - 1)
            if self.mutant == "e" and not rd["pre_dropped"]:
                self.drop_pre()  # mutant (e) = protocol (д): pre-W runs dropped at the FIRST section
            self.publish()
            if old[2] is not None:
                old[2].close()
            if old[0] != gen:
                self.base_path(target, old[0]).unlink(missing_ok=True)
            t_build = time.perf_counter() - t1
        finally:
            written, removed = _Audit.stop()
        n_rows = int(len(keys))
        live_max = int(pages.max()) if n_rows else -1
        if self.trace_mem:
            mem_peak = tracemalloc.get_traced_memory()[1] - mem0
        # witnesses are picked here (not by the stand later): the pairs are released when the step returns
        wit = self.step_hook(s, keys, pages) if self.step_hook is not None else []
        del keys, pages
        ev = {"kind": "round_step", "part": s, "section": target, "rows": n_rows, "adr_bytes": nb,
              "pair_bytes": n_rows * 20, "max_live_page": live_max, "s_read": t_read, "s_build": t_build, "last": last,
              "files_written": sorted(set(written)), "files_removed": sorted(set(removed)), "_wit": wit} | info
        aw, ar = self.allowed_step_files(target, gen, old[0])
        bad = sorted((set(written) - aw) | (set(removed) - ar))
        if bad:
            self.s5_violations.append({"part": s, "files": bad})
        ev["s5_bad_files"] = bad
        if self.trace_mem:
            ev["mem_peak_bytes"] = mem_peak
        if last:
            if self.crash_before_end:
                raise RuntimeError("injected stop after the last publication, before round_end")
            ev["round_end"] = self.audited_end()
        ev["s"] = time.perf_counter() - t0
        return ev

    def round_end(self) -> dict:
        t0 = time.perf_counter()
        rd = self.round
        if not rd["pre_dropped"]:
            self.drop_pre()
        dropped = getattr(self, "_dropped", [])
        self._dropped = []
        old_tail = None
        if self.mutant == "i":
            old_tail = self.new_tail()  # mutant (i): the tail "before W" is dropped together with everything after W
        self.base_gen = rd["gen"]
        self.round_start_ops = rd["W_ops"]
        base_sum = sum(n for _, n, _ in self.base)
        self.n0 = self.n_live if self.n_live is not None else base_sum
        summary = {k: rd[k] for k in ("W", "rows", "max_rows", "steps")}
        summary["n0"] = self.n0
        summary["base_keys_sum"] = base_sum
        summary["window_ops"] = self.opseq - rd["t0_op"]
        self.round = None
        self.publish()
        for r in dropped:
            r.remove()
        if old_tail is not None:
            old_tail.close()
            old_tail.path.unlink(missing_ok=True)
        summary["s"] = time.perf_counter() - t0
        return summary

    # ---- one batch boundary: seal, then at most one merge step, then at most one round step
    def boundary(self, read_fn, p0_fn=None) -> list[dict]:
        evs = []
        if self.tail.committed_ops >= self.t_run:
            evs.append(self.seal())
        if self.round_due():
            self.round_pending = True
        if self.job is not None or self.maybe_start_merge():
            evs.append(self.merge_step())
        if self.round_pending and self.round is None and self.job is None:
            evs.append(self.round_start())
        elif self.round is not None:
            evs.append(self.round_step(read_fn))
        for e in evs:
            e.setdefault("after_op", self.opseq)
        return evs

    def run_round_now(self, read_fn, p0_fn=None) -> list[dict]:
        """Final round: steps back to back without operations (§3)."""
        self.round_pending = self.round is None
        evs = []
        guard = 0
        while self.round_pending or self.round is not None or self.job is not None:
            evs.extend(self.boundary(read_fn))
            guard += 1
            if guard > 100_000:
                raise RuntimeError("round did not finish")
        return evs

    # ---- accounting
    def spill_bytes(self) -> int:
        return 0

    def stray_files(self) -> list[str]:
        """S5: files in the address directory that are not part of the published structure, the tail, the manifest
        or the partial output of the active tier merge."""
        ok = {self.manifest.name, self.tail.path.name}
        ok |= {self.base_path(j, g).name for j, (g, n, _) in enumerate(self.base) if n}
        for r in self.runs:
            ok |= {p.name for p in r.files()}
        if self.job is not None:
            ok.add(self.job.part.name)
            ok |= {self.job.out.adr_path(j).name for j in range(self.job.done)}
        return sorted(p.name for p in self.dir.iterdir() if p.name not in ok)

    def close(self) -> None:
        for r in getattr(self, "_dropped", []):
            r.close()
        super().close()
