---
name: Results
about: Share the results of a run on your hardware
title: "Results: <experiment> <size> on <CPU / storage>"
labels: results
---

**Experiment and command** (for example `a8_run.py run`, `a6_run.py dynamics N10`):

**Commit** (`git rev-parse HEAD`):

**Hardware**: CPU, RAM, storage (model, NVMe/SATA, free space on the data volume):

**OS**:

**PostgreSQL**: version, and any settings you changed from the defaults (or attach `SHOW ALL`):

**Self-check**: did `<runner> selfcheck` pass on this stand? (`all_ok` in the self-check JSON)

**Files**: attach the JSON and log files written to `rcm_address_engine/results/` by this run.

**Notes**: anything unusual (other load on the machine, errors, restarts):
