# Session 0008 - deep lane #2: the scaled vertical and its two ablations

**Status: CONSTRUCTION COMPLETE, DISPATCH PENDING.** This log records what was built and what the
project's own gates caught. It carries **no metric**: the run has not been dispatched from the clean
runner, so no result exists to record. Per `LOG/session-0008-preregistration.md` the verdict is
recorded here verbatim - whichever way it goes - only after the run, never before.

## The question, and where its criterion lives

Deep lane #2 asks whether raising capacity (d_model 128 -> 256, 8 heads) AND giving the model move
history (K=8 boards) raises top-1 move accuracy on the SAME day-prefix v1 used, and then attributes
the change to an organ with two micro-ablations. The thresholds (SUCCESS >= 0.30; DEAD <= the
most-common-move baseline; INCONCLUSIVE in between; INVALID if sampling legality < 100%) were
committed in the pre-registration BEFORE any run. This session did not move them.

## What was built

- `configs/smoke-train-v2.yaml` - the scaled vertical (K=8, d_model 256, 8 heads, geometry on) plus
  the two ablations, and the honesty notes about identical data and split seed.
- `src/caissa/model/numpy_history.py` - the history model: a K-board window mixed by a SHARED
  encoder, a multi-head bilinear (from, to) head, and a geometry bias behind a flag whose "off"
  contribution is exactly zero (so the ablation is one config line, not a code change).
- `src/caissa/training/dataset.py` - `HistorySample` + `build_history_samples`: windows never cross a
  game boundary, and a game too short for a window contributes nothing rather than being padded.
- `src/caissa/training/smoke_train_v2.py` - trains the main configuration, then each ablation from
  the SAME shards and the SAME split seed; writes `metrics-v2.json` and `ablations-v2.json`.
- `.github/workflows/smoke-train-v2.yml` - the clean-runner acceptance machine (data -> shards ->
  train v2 -> metrics + ablations), with a clone guard that keeps data out of git.
- `src/caissa/processing/labeling.py` + `cli.py` - the labeling economy (below).
- `tests/unit/test_numpy_history.py`, `test_labeling_economy.py`, `test_smoke_train_v2.py`.

## Two defects, caught by the project's own gates

**1. The gradient check found a split objective in the history model.** `train_step` inlined its own
forward pass that OMITTED the geometric bias while `logits()` added it. With the bias on, the model
therefore TRAINED one objective and was EVALUATED on another - every v2 loss number would have been
internally inconsistent, and the geometry ablation would have compared two things that were never the
same function. `tests/unit/test_numpy_history.py` caught it: the analytic gradient matched a central
finite difference only when `geometry_bias=False`, and diverged when it was True. Fixed by giving the
model ONE `_forward` used by `logits`, `loss` and `train_step` alike; a regression test now pins it.

**2. The ADR-0009 publish -> verify check found a silent publication corruption.** Commit 05bf8f79
published `tests/unit/test_labeling_economy.py` with a blob sha (d34a2329) that did NOT match the
local bytes (af336cd0). Cause, pinned with `cmp -l`: the local copy carried FIVE literal backslash-u
escape sequences (3 em-dashes, 1 en-dash, 1 arrow) as the six-character ESCAPE TEXT, while the
published blob held the real UTF-8 characters (3 bytes each; 5850 - 5820 = 15 = 5 x 3). The update
transport normalises those escape sequences, so a file containing them silently lands in git as
mojibake with a different hash. The CODE was identical - only docstring prose differed, which is
exactly why nothing downstream would ever have caught it. Fixed by making the file pure ASCII
(0 bytes >= 0x80, 0 backslash-u sequences) and re-publishing; the blob sha then matched. The lesson:
publish, then verify every blob sha against the local bytes - never assume a clean push.

## The labeling economy (fast lane)

Session-0007 left the real-Stockfish labeling job running past a CI budget. This session's answer is
in `labeling.py`: an `EvalCache` that makes one engine call per DISTINCT position (the "after" of ply
i is the "before" of ply i+1, so the cache nearly halves the work and REPORTS its hit/miss counts), a
timeout that yields a PARTIAL report rather than a silent truncation, progress checkpoints, and an ETA
derived from a MEASURED rate (None - not a guess - before any measurement exists).

## Publication

Published to `main` through the GitHub API (ADR-0009; this sandbox has no `.git`):

- `978a0536` v2 scaffolding + pre-registered criterion + silence-guard step
- `9cec4f29` history model + labeling economy + runner wiring
- `05bf8f79` the three test modules
- `acdfbb7b` the ASCII fix (defect #2 above)

All 14 files' published blob shas were verified byte-for-byte against the local source after the
final commit.

## The run: PENDING

`python scripts/run.py smoke-train-v2 --config configs/smoke-train-v2.yaml` (or the `smoke-train-v2`
GitHub workflow) trains the main configuration and both ablations on the same day-prefix shards. When
it has run, its verdict is recorded here verbatim with the metrics artifact and the run URL - a
negative result with the same care as a positive one (pre-registration; GOAL.md Article 5).

## What is NOT claimed

No top-1 accuracy, no ablation delta and no verdict is claimed by this session. The local tests
exercise the machinery on a small fixture whose accuracy is meaningless as a result; the fixture is
separable by construction, so the real number can only come from the real shards.
