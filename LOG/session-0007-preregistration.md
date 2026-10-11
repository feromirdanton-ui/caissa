# Session 0007 — PRE-REGISTRATION of the smoke-train falsification criterion

**Written BEFORE the run.** Committed before the first `smoke-train.yml` dispatch, so the git
history itself proves the criterion preceded the numbers (ADR-0011 §5; GOAL.md Article 5).

The vertical (representation + tiny model) has never been exercised on **real data**. This is the
first real training the project performs. The criterion is fixed here, in advance, precisely so it
cannot be moved after the fact.

---

## The experiment

`smoke-train.yml` (workflow_dispatch, clean runner, ≤60 min CPU):

```
data-sample (day-prefix of one Lichess month)  →  processing (filters, dedup, jsonl.zst shards)
        →  position dataset (real games)  →  train the tiny model  →  metrics artifact
```

Model: `caissa.model.numpy_tiny.NumpyTinyModel` — the REAL tiny model (SGD, masked softmax,
CPU-only), not a mock. Data: real positions from real rated standard Lichess games.

## The criterion — written in advance, not negotiable

The vertical is **ALIVE** if and only if **both** hold on the validation split:

1. **VAL LOSS DECREASES.** `val_loss_final < val_loss_first`, with a margin that exceeds the
   seed-to-seed spread measured in the same run (the run reports the first/last val loss and the
   curve, so the margin is visible rather than asserted).
2. **TOP-1 MOVE ACCURACY BEATS THE BASELINE.** Baseline = the frequency of the single most common
   (from, to) move in the **training** split, measured on the **validation** split. The model's
   top-1 accuracy must be **strictly greater** than that baseline.

The vertical is **DEAD** if either condition fails: a plateau, a rising val loss, or model accuracy
at or below the baseline. "Dead" is a *result*, not a failure — it says the composition as wired
does not learn, which is exactly what this run exists to find out.

## The gate — validity, not a criterion

- **Sampling legality must be 100%** (1000 sampled moves, Article 1). If it is not, the run is
  **INVALID** rather than "dead": a legality leak is a defect to fix, not a finding to record.
- **Honesty gate (Article 5):** no fabricated numbers. If a stage cannot run in the runner's
  budget, its scale is reduced and the reduction is stated — never a silent truncation and never
  an invented figure.
- **Disposable weights (Article 8):** the trained weights are a run artifact, never a git object.
  The goal is to prove or refute the vertical, not to ship a model.

## Retouching is forbidden

Whatever the outcome, it is recorded verbatim in `LOG/session-0007.md` with the run URL and the
metrics artifact. A negative result is recorded with the same care as a positive one and is worth
more than a retouched positive (§9, Article 5). The criterion above may not be edited after
dispatch; if it ever needs to change, that is a new pre-registration for a new run.

---

*Pre-registered by the Caïssa-1 construction agent, session 0007, before dispatch.*
