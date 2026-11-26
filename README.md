# RCM Path: a computed address layer for PostgreSQL

**RCM Address Engine** finds the heap page of a row by *computing* it from the row's key instead of walking a B-tree.
Each key costs about **3–5 bytes**: a minimal-perfect-hash cascade, an 8-bit fingerprint and a page number. A
PostgreSQL B-tree on the same key costs about 31.5 bytes. The engine runs as a **sidecar** next to an ordinary
PostgreSQL table: no database of its own, no server patch. A lookup of a present key reads one heap page with a Tid
Range Scan and checks the key there.

This repository is the complete, reproducible record of experiments **A2–A8** (November 2026). They tested the idea
on **1 million and 10 million rows**, both static and under a stream of inserts, updates and deletes. It contains:

- the code;
- the protocols, frozen by sha256 before the measurement runs;
- the raw results and the reports.

All numbers come from one modest desktop machine. The results are preliminary. They are published so that others can
**re-run the experiments on stronger hardware** and at larger N, and extend them; see
[What we ask from testers](#what-we-ask-from-testers).

Author: **Abu ([@tymakbayev](https://github.com/tymakbayev))**. License: [MIT](LICENSE).
Russian version: [README.ru.md](README.ru.md).

## How it works

```
key (uuid, 16 B) ──hash64(key, seed)──► h                         G  generated, not stored
h ──► P: cascade of bitmaps "the key landed alone in this cell"     P  ≈3 bits/key, rank in O(1)  ──► slot
slot ──► D[slot] = fingerprint (8 bit) | heap page (⌈log2 pages⌉ bit)   R  the residual per key
SELECT … FROM t WHERE ctid >= '(page,0)' AND ctid < '(page+1,0)' AND k = $key    reads that heap page
```

- **The static address stores neither keys nor a search tree.** The address file (format `JLMADR01`) is a set of flat
  arrays, and the time to open it does not depend on N.
- **A returned row is always the requested key.** The key is compared in the row. A fingerprint collision costs one
  extra page read, never a wrong row. Absent keys are almost always rejected by the fingerprint before any SQL.
  Finding every live row relies on the address being kept in step with the table under the
  [operating contract](#operating-contract-of-the-prototype).
- **Changing data (A4–A6).**
  - New rows go to a tail log. Sealed batches become small addresses ("runs"), and runs are merged in tiers, like an
    LSM tree of addresses.
  - These variants also keep run logs with full keys (16-byte key + 4-byte page) for the merges. The logs stay off
    the lookup path.
  - The big merge runs in bounded steps: per hash section of the address (A5), or per hash partition of the table
    (A6).
- **Point lookups by key only.** Range scans and ordering still need a B-tree.
- **It is still an index**: a static function "key → page", not a "zero-byte index". When row placement does not
  depend on the key, an exact key → page map needs about log2(pages) bits per row; an exact row position needs about
  log2(n) − 1.44 bits (`rcm_address_engine/STAGE1.md` §2.1). The page field above is that unavoidable part. The
  cascade (≈3 bits) and the fingerprint (8 bits) are the overhead.

The idea comes from the RCM Path model "generator + trajectory + residual", which the author first developed for
compact long-term memory: [`theory/`](theory/README.md).

## Results

Reference stand (below), PostgreSQL 18.7 with default settings (`shared_buffers` 128 MB), warm OS cache. Numbers are
**measured** unless marked *derived*. "1M / 10M" means 1 and 10 million rows.

| | Result | Source |
|---|---|---|
| Static address size, bytes per row, 1M / 10M | **3.57 / 3.94** vs B-tree 31.56 / 31.54 and hash index 33.57 / 26.85: about **8–9× smaller than the B-tree**, 6.8–9.4× smaller than the hash index | A2 |
| Heap-buffer accesses per lookup, static | present key: exactly 1.000 (B-tree 4.000 buffer accesses in total, hash index 2.0–2.24); absent key: 0.005–0.006 on average, because 99.5% of absent keys are rejected without SQL | A2 |
| PostgreSQL buffer accesses per live-key lookup under updates | **1.00–1.04 vs 4.00–4.06** for the B-tree control, lower at every checkpoint on 1M and 10M. Only PostgreSQL buffers are counted; reading the address file is not | A4, A5, A6 |
| Published address bytes per live key under updates | 3.1–5.8 at the measured checkpoints and batch boundaries (target ≤ 8). More on disk while merges run: A4 8.5–9.5 (1M) / 8.78 (10M) when old and new generations coexist; A5 up to 24.8 / 24.6 including its work file (A6 removed the work file) | A4, A5, A6 |
| Opening the address | static 2.98 / 2.66 ms; A4 about 11 ms; A5 13.9 / 47.6 ms; A6 12.7 / 47.3 ms (grows with the number of section files) | A2, A4, A5, A6 |
| Correctness | no wrong answers in the prescribed correctness samples, in the witnesses checked after every maintenance step, and after restarts. Every required non-equivalent mutant was detected by its designated check; one mutant in A5/A6 was proven equivalent before the runs and replaced | A2–A7 |
| Estimated steady-state address maintenance per operation, 1M / 10M (*derived*) | A4 26.4 / 22.2 µs; A5 44.0 / 33.1 µs; A6 33.0 / 28.7 µs: lower at 10M than at 1M in each variant. The A5 and A6 10M values rest on one forced round; the natural SQL maintenance cycle at 10M was not run | A4, A5, A6 |
| Longest maintenance pause, 1M / 10M | A4 5.0 / 18.5 s (10M: the final big merge) → A5 1.13 / 0.40 s → **A6 0.62 / 0.36 s** (longest working step) | A4, A5, A6 |
| Client lookup latency, p50 | **slower than the B-tree**: static 177 vs 101 µs (1M) and 174 vs 110 µs (10M); under updates 179–322 vs 100–149 µs. The server part alone is slower too (Tid Range Scan 47 µs vs Index Scan 28 µs at 1M). The Python client adds about 22 µs to compute a static address, and about 12.7 µs per level under updates | A2, A4 |
| Full cost per operation vs B-tree, 1M / 10M (*derived*: operations + VACUUM + address maintenance) | A4 ×1.23 / ×1.04 and A5 ×1.39 / ×1.04 (steady state). A6 ×1.115 / ×0.995 steady state (×1.09 / ×1.05 actual) vs a B-tree on the same partitioned table, and ×1.31–1.36 actual vs a single-table B-tree of another run | A4, A5, A6 |
| Price of hash partitioning (one run with three tables) | 1M, two runs: operation time through the parent table ×1.251 / ×1.228 vs a single table; A6 full cost ×1.36 / ×1.33 vs the single-table B-tree. 10M, one run with three 6.4 GB tables on one server: the sign flipped, operation time ×0.807 and A6 full cost ×0.853. The cause: existing-key lookups on the single table took 544–548 µs vs 223–256. Why is **not established** | A7 |

### What is established and what is not

Established on 1M and 10M rows, on one machine:

- the static computed address is about 8–9× smaller than PostgreSQL's B-tree on the same key. A present-key lookup
  reads exactly one heap page, and absent keys almost never reach the table;
- under a mixed stream of 500 000 writes the address keeps ≈1 PostgreSQL buffer access per live-key lookup against
  4, with no wrong answers in the prescribed checks;
- there is no monolithic rebuild pause: the base address is refreshed section by section (A5) or partition by
  partition (A6). The measured pauses and the estimated maintenance cost per operation were lower at 10M than at 1M.

Not established, or negative:

- **No latency win.** Fewer buffer accesses did not make queries faster with a warm cache. Both the server part and
  the Python client are slower than a B-tree index scan. A C implementation of the lookup (for example a PostgreSQL
  index access method) has not been built.
- **Full cost per operation is not better than a B-tree in A4–A6**: higher at 1M (×1.09–1.39), about equal at 10M
  (×0.995–1.05). A7 measured ×0.853 at 10M, but only in the three-table co-run, where the single table slowed down
  for an unknown reason. The isolated comparison is open; A8 was designed for it. Its protocol, code and self-check
  are ready, but the 10M run was not done because the stand ran out of disk.
- **Scale.** Only two sizes were tested. A4–A6 have one run per size; A7 has two runs at 1M and one at 10M. Nothing
  was run at 100M. A6 would then need repartitioning (about 1 024 partitions, derived), which was not tested.
- **Negative protocol verdicts** are kept as they are:
  - A2 failed its criterion A4b. Variant A exceeded the B-tree control's buffer accesses at two checkpoints (4.019 /
    4.004 vs 4.013 / 4.000); variant Б exceeded it at all 12 (4.25–5.12 vs 4.00).
  - A5 failed its own 1.0 s pause threshold at 1M (1.13 s).

### Operating contract of the prototype

The guarantees above hold only inside this contract:

- one writer; readers are serialized with write batches; maintenance runs while clients wait (that is the pause
  measured above);
- no long-running snapshots, and no `VACUUM FULL` or `CLUSTER`: they move rows and would invalidate the addresses;
- only consistent restarts between batches. A crash of the OS is outside the contract.
- It is a sidecar, not a PostgreSQL index access method. PostgreSQL's WAL and crash recovery do not cover the address
  files.

What is counted, and what is not:

- "Address bytes" means the published address. Temporary merge files and process memory are reported separately: the
  peak RSS of the Python client was 1.17 / 5.18 GB in A6, including the reference model used for checking, and
  134 / 893 MB for the replay without SQL.
- "Buffer accesses" counts PostgreSQL buffers only. It does not include reading the address or Python work, and it is
  not the number of physical SSD reads.

## Experiments

| | Question | Verdict by its frozen protocol | Report |
|---|---|---|---|
| A2 | Sidecar address for a PostgreSQL table: variant A (perfect hash + page) and Б (hash-bucket table layout) | H-A and H-Б **not passed**, by criterion A4b (buffer accesses under updates, see above). Static criteria passed | [a2_report.md](rcm_address_engine/a2_report.md) |
| A3 | Frequent full rebuild (T = 30 000) | **passed** at 36 checkpoints; but the rebuild cost per operation grows ∝ N | [a3_report.md](rcm_address_engine/a3_report.md) |
| A4 | Addressable delta: levels of small addresses, tier merges | **passed** on 1M and 10M; the big merge pause grows with N | [a4_report.md](rcm_address_engine/a4_report.md) |
| A5 | Big merge in steps over a hash-sectioned address | **not passed**, by the pause threshold only (1.13 s > 1.0 s at 1M); the pause no longer grows with N | [a5_report.md](rcm_address_engine/a5_report.md) |
| A6 | Hash-partitioned table, one partition per step, no work file | **passed**, all conditions S0–S5 | [a6_report.md](rcm_address_engine/a6_report.md) |
| A7 | Price of hash partitioning: single table vs partitioned vs A6, in one run | descriptive, no pass/fail | [a7_report.md](rcm_address_engine/a7_report.md) |
| A8 | Why single-table lookups were 2× slower at 10M in A7 | protocol, code and self-check ready; the 10M run is open | [A8_protocol.md](rcm_address_engine/A8_protocol.md) |

How the work was done:

- Every experiment has a protocol frozen by sha256 before its measurement runs; facts found later go into dated
  addenda. The hashes are in `rcm_address_engine/results/*_protocol.sha256`, and each runner refuses to start if its
  protocol file changed.
- Numbers in the reports are labelled ЗАМЕР (measured), ВЫВОД (derived) or ИСТОЧНИК (taken from another run).
- The research hypothesis is reported separately from the protocol verdicts.

The author directed the work and made every decision. Code, protocols and reports were written with AI assistance
(Claude Code). Before the runs, a second AI model (Codex) reviewed every protocol, and from A6 on also the code. The
review rounds are recorded in the protocols and addenda.

The reports and protocols are in Russian.

## Repository layout

```
README.md, README.ru.md, LICENSE, CITATION.cff, requirements.txt
theory/                        RCM Path model (memory_math.md) and results of the project it came from
rcm_address_engine/
  STAGE1*.md                   audit and design of the address scheme (stage 1)
  A*_protocol.md, A*_addendum  frozen protocols and dated addenda
  a*_report.md                 reports with verdicts
  a2_core.py                   hash, JLMADR01 address (variant A), bucket layout (variant Б)
  a2_pg.py                     PostgreSQL layer: server control, COPY load, lookup clients, EXPLAIN, pg_statio
  a4_lsm.py, a5_lsm.py, a6_lsm.py   address LSM: tail log, runs, tier merges, stepwise big merge
  a6_pghash.py                 client copy of PostgreSQL hash-partition routing (checked against the server)
  a2*.py, a4_run.py … a8_run.py     experiment runners (self-checks with mutants, dynamics, replay, open probes)
  a2_gen.py, corpus.py         deterministic record generator
  verify_corpus.py             regenerates a stream and compares it with results/data_manifest.sha256
  hoststats.py                 process memory and disk counters (Windows and Linux)
  results/                     raw JSON results and logs of the reference runs
  probes/                      stage 1 probes (perfect-hash sizing, SQLite baseline)
```

The first commit holds the files exactly as they ran. The later commits make them portable without changing any
algorithm: configuration from environment variables, a standalone generator, Linux counters and safety checks. See
their messages.

**Reproducible data.** `corpus.py` is a verbatim copy of the author's generator. On the reference stand, the arrays
of keys, payload digests and payload lengths it produces match those of the published runs byte for byte for 100k,
1M and 10M records (`verify_corpus.py`). The COPY text stream fed to PostgreSQL also matched the original generator
byte for byte on the windows that were compared (see the commit messages). This was checked on Windows only.

## Run it yourself

### Requirements

- **PostgreSQL 18** (tested with 18.7): a **dedicated local test cluster** that you control.
  - The runners restart it with `pg_ctl` for cold-start checks. `pg_ctl` must run under the OS account that owns the
    cluster, not as root. A PostgreSQL superuser role does not grant that.
  - They create and drop tables in database `a2` and create the extensions `pageinspect`, `pg_buffercache` and
    `pg_prewarm`, so the role must be a superuser.
  - Before any measurement they check that the SQL connection reaches the cluster in `RCM_PG_DATA`.

  Do not point them at a server with data you care about.
- **Python 3.10+** with `pip install -r requirements.txt` (numpy 2.2.6 and psycopg 3.2.10, the versions of the
  reference runs; numpy pins the generator's random streams).
- **Disk** on the data volume. A7 and A8 at 10M check for at least 24 GB free before starting (three tables of
  6.4 GB). Approximately: 1M runs need a few GB, and the A2 static 10M tables took about 13 GB.
- **Memory**: the reference stand has 32 GB. The peak RSS of the Python process was 4.4 GB (A7, 10M) and 5.18 GB
  (A6, 10M, with the reference model).
- **Time**, approximate, from the log timestamps of the reference runs: a self-check takes 5–10 min, a 1M dynamics
  run about 15 min, a 10M dynamics run 1–2 h, and A2 static 10M 30 min.

### Configure the stand

| Variable | Meaning | Default |
|---|---|---|
| `RCM_PG_DATA` (or `PGDATA`) | data directory of the test cluster (restarts, free-space checks, cluster check) | none |
| `RCM_PG_BIN` | directory with `pg_ctl` | directory of `pg_ctl` on `PATH` |
| `RCM_PG_LOG` | server log used by `pg_ctl -l` (keep it off the data volume for long runs if space is tight) | `$RCM_PG_DATA/rcm_server.log` |
| `PGHOST`, `PGPORT`, `PGUSER` | connection to the same cluster | `localhost`, `5432`, `postgres` |
| `PGPASSWORD` or `RCM_PG_PWFILE` | password, or a file holding it (libpq `~/.pgpass` also works) | none |
| `RCM_DB` | database for the experiment tables | `a2` |

The reference stand used port 5433 and a superuser role `bench`. Windows device counters (A8) use the drive letter of
`RCM_PG_DATA`. A data directory on a volume mounted into a folder, behind a junction, or on a network path is not
resolved to its own volume.

### Steps

```sh
python -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd rcm_address_engine

# no database needed
python verify_corpus.py small         # reference arrays reproduced byte for byte (also: n1, n10)
python a2_selftest_core.py            # address core
python a5_selftest_lsm.py             # address LSM with stepwise merge, restarts mid-round
python a6_selftest_lsm.py             # partition-per-step variant

# with the test cluster configured (see the protocol of each experiment for what is measured)
python a2.py selfcheck ; python a2.py static N1 ; python a2.py static N10 ; python a2.py dynamics
python a2.py dynamics-a3
python a4_run.py selfcheck ; python a4_run.py dynamics N1 ; python a4_run.py dynamics N10 ; python a4_run.py replay N1
python a5_run.py selfcheck ; python a5_run.py dynamics N1 ; python a5_run.py dynamics N10 ; python a5_run.py replay N1
python a6_run.py selfcheck ; python a6_run.py dynamics N1 ; python a6_run.py dynamics N10 ; python a6_run.py replay N10
python a7_run.py selfcheck ; python a7_run.py dynamics N1 r1 ; python a7_run.py dynamics N10 r1
python a8_run.py selfcheck ; python a8_run.py run
```

- Run experiments **one at a time**: several of them restart the server.
- A6–A8 refuse to run dynamics until a self-check of exactly the same code has passed on your stand. The self-check
  files in `results/` from the reference stand do not count, because the code hash changed in the portability commits.
- Results are written to `rcm_address_engine/results/` over ours, with the same file names. Work on a branch or copy
  the folder first; `git diff` then shows your numbers against ours.
- The Linux code paths were written but have not yet been run against PostgreSQL.

## What we ask from testers

Our hardware is a single desktop, so these questions are open:

1. **A8 at 10M**: why are single-table existing-key lookups 2× slower when three 6.4 GB tables share one server? It
   needs ≥ 24 GB free disk. More RAM or faster NVMe storage is welcome; report it with the result.
2. **A6 and A7 at 10M and at larger N** (for example 100M): do the maintenance cost per operation and the longest
   pause stay flat? Is the price of partitioning the same when each system runs alone?
3. **Different hardware and settings**: Linux, NVMe, larger `shared_buffers`.
4. **A C lookup path** (an index access method or a C extension): can the one-page address beat the B-tree on
   latency, not only on bytes and buffer accesses?

To share results, open an issue with the **Results** template and include:

- the JSON and log files from `results/`;
- CPU, RAM and storage;
- OS, and the PostgreSQL version and settings. The A2, A7 and A8 results record the settings themselves; for the
  others, attach the output of `SHOW ALL`.

## Reference stand

| | |
|---|---|
| CPU | AMD Ryzen 5 3600X (6 cores / 12 threads) |
| RAM | 32 GB DDR4 at 2400 MT/s |
| Storage | SSD (system volume, which also held the PostgreSQL cluster) |
| OS | Windows 10 Pro |
| Database | PostgreSQL 18.7 (EDB zip build), default settings: `shared_buffers` 128 MB, `effective_cache_size` 4 GB, `io_method` worker |
| Python | 3.10.10, numpy 2.2.6, psycopg 3.2.10 |

## Citation

See [CITATION.cff](CITATION.cff). Please cite the repository and keep the copyright notice when reusing the code (MIT).
