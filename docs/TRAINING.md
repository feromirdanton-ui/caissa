# TRAINING — the runbook

> Goal (GOAL.md §10): an engineer with 8×A100 launches Act I with **one command**.
> Every step has a checkpoint and a fallback. Nothing here is a promise of results —
> the build phase prepares the run; it does not run it.

## 0. Prerequisites

- A machine with one or more CUDA GPUs (8×A100 is the reference target).
- Python 3.10+ and a working `pip`.
- Disk: see the data manifest for exact volumes (never trust memory — GOAL.md §7).
- Optional: a Stockfish binary for the arena and the counterfactual harness.

## 1. One-command launch (the Definition of Awesome)

```bash
git clone <repo> && cd caissa
make install
make train CONFIG=configs/nano.yaml
```

`make train` dispatches to `python -m caissa.training.act1 --config <CONFIG>`. The config selects
model scale, data shards, batch size, and checkpoint cadence.

## 2. Step-by-step (with checkpoints and fallbacks)

| Step | Command | Checkpoint | Fallback |
|---|---|---|---|
| 1. Install | `make install` | `pip show caissa` | use a venv; pin Python 3.11 |
| 2. Verify env | `make lint test` | all green | fix env before proceeding (Article 7) |
| 3. CPU smoke | `make smoke` | loss decreasing, 1000 moves 100% legal | reduce `steps` in the config |
| 4. Fetch data | `make data-sample` | manifests written, resume works | re-run; completed shards are skipped |
| 5. Process | `make data` then shard CLI | shards + QA report in `LOG/` | re-run from the last completed shard |
| 6. Train Act I | `make train CONFIG=configs/nano.yaml` | checkpoints every N steps | resume from the last checkpoint |
| 7. Evaluate | `make eval` | arena + blunder-rate report | mock engine if Stockfish is absent |

## 3. Resumability (Article 4)

- Downloads: `.part` files + HTTP `Range`; a completed shard is recorded in the manifest and never
  re-fetched.
- Processing: shards are written atomically; a crash leaves the previous shards intact.
- Training: checkpoints are written on a fixed cadence; `--resume <ckpt>` continues exactly.

## 4. Scale configs

| Config | Params | Hardware | Purpose |
|---|---|---|---|
| `configs/tiny.yaml` | <10M | CPU | smoke test, CI |
| `configs/nano.yaml` | 100–300M | 1–8 GPUs | first real Act I run |
| `configs/main.yaml` | 1–3B | 8×A100+ | the full being |

## 5. Distributed (stubs)

DDP/FSDP entry points are stubbed in `src/caissa/training/` with documentation. They are **not**
verified on a cluster in the build phase — they are labeled as stubs, not as working recipes.

## 6. Honesty

- No training results are produced or claimed in this repository.
- Any mock (engine, data, model) is labeled `MOCK` in code and in its output.
- If a step cannot be completed, it is parked with a note (GOAL.md §6.5) — never faked.

## 7. Phase 2 — the processing pipeline (raw → shards → QA)

The pipeline is one command and never loads the archive into memory:

```bash
# stream a PGN (plain or .zst) -> filter (ADR-0010) -> dedup -> shard -> QA report
make processing PGN=data/raw/lichess/lichess_day_2013-01.pgn.zst \
                OUT=data/shards QA=data/qa/processing.json

# blunder labeling (real Stockfish when present, else MOCK — labeled)
make label PGN=data/raw/lichess/lichess_day_2013-01.pgn.zst LIMIT=10000 ENGINE=auto
```

### The 1M-game DoD — the full procedure

The build environment cannot hold 1M games honestly, so the acceptance run processed the
**day-prefix (56 962 games)**. On a machine with the disk, the full procedure is:

```bash
# 1. pull a full month of Lichess bulk (units of GB; static files, download calmly)
python -m caissa.acquisition.lichess_bulk --month 2013-01 --out data/raw/lichess

# 2. stream the whole month through the pipeline (O(1) memory in the number of games)
make processing PGN=data/raw/lichess/lichess_2013-01.pgn.zst \
                OUT=data/shards QA=data/qa/processing.json

# 3. the QA report carries the real counts and every shard's sha256
```

**A Lichess archive is a concatenation of zstd frames.** The parser reads every frame
(`_ZstdRaw._feed` rolls over via `unused_data`); a stream that ends mid-frame is
*truncated* and is named as such, never read as an empty complete archive.

### Resume

Shards are written to `<name>.part` and renamed into place only after their bytes are
flushed, so a crash leaves the previous shards intact. Re-running the pipeline over the
same input reproduces byte-identical shards (same input → same sha256).

### Honest scale

A small honest dataset beats a silent truncation, and both beat a fabricated volume
(Article 5). If the disk cannot hold the month, process the day-prefix and say so in the
LOG — do not shrink the volume silently.

