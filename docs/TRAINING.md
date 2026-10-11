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
