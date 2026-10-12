# Session 0009 — «Выстрел» (deep lane: fire and verdict)

**Precondition:** none required — the milestone is unsigned and this session does not need a
signature. It dispatches the session-0008 experiment and records the verdict.

**The criterion was fixed before the shot** in `LOG/session-0008-preregistration.md` and is not
restated or adjusted here. The preflight (unified eval set + interpretation note) is in
`LOG/session-0009-preflight.md`, committed before the first dispatch. The append-only shot registry
is `LOG/session-0009-shots.md`.

---

## Step 0 — Preflight (done before dispatch)

1. **Unified eval set.** It was **not** unified; the confounder is fixed. `_train_once` built the
   history window with the configuration's own `K`, so the main run (`K=8`) was scored on `ply>=8`
   while `no_history` (`K=0`) was scored on `ply>=0` — a larger, different set. The window is now
   built **once** with the main run's `K` (`window_history`), and a smaller-`K` model reads only the
   suffix of that same window. Pinned by
   `test_eval_set_is_identical_across_main_and_ablations`, mutation-verified (reverting the fix makes
   it red). **Confirmed on the runner:** all three configurations report `val_samples=2000`,
   `val_games=149` — the same set.
2. **Interpretation note.** The v2 eval set is a strict subset of v1's and **harder** (opening plies
   excluded), so any v2 improvement over v1's 0.1839 is **understated, not overstated**. Thresholds
   untouched.

## Step 1 — The shot

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38186843291
- **head_sha:** `584517fd` (the preflight commit) · **conclusion:** `success` · all steps green.
- A first dispatch (run `38186826762`) started on the **wrong commit** (`ff7e2813`, the session-0008
  HEAD) because of a ref-update/dispatch race; it was cancelled and is recorded as **VOID** in the
  registry. A run on the wrong commit is not a result.

## Step 2 — The verdict (verbatim, no retouching)

> **INCONCLUSIVE** — `"top-1 0.0925 beats the baseline but is below 0.30"`

This is the pre-registered rule's own output, recorded as it stands. It is a **legal outcome** of the
experiment, not a failure to hide.

### The milestone table

| config | top-1 | Δ vs main | val loss (first → last) | legality | val samples | verdict |
|---|---|---|---|---|---|---|
| **v1** (`numpy_tiny`, K=0) | 0.1839 | — | 3.2603 → 3.1326 | 1000/1000 | — | ALIVE |
| **v2 main** (K=8, d_model 256, geom on) | **0.0925** | — | 3.4014 → **NaN** | 1000/1000 | 2000 | INCONCLUSIVE |
| ablation `no_geometry_bias` | 0.0925 | **0.0000** | 3.2053 → **NaN** | 1000/1000 | 2000 | INCONCLUSIVE |
| ablation `no_history` (K=0) | 0.0925 | **0.0000** | 3.4001 → **NaN** | 1000/1000 | 2000 | INCONCLUSIVE |

- **Metrics artifact:** `smoke-train-v2-metrics` (id `11719948779`) on run `38186843291`.
- **CI (green):** https://github.com/feromirdanton-ui/caissa/actions/runs/38187454120 (`ec784ea3`).

### Interpretation — the number is not a measurement

**The run diverged.** `train_loss_curve` is `[NaN, NaN]` and `val_loss_last` is `NaN` for the main
run **and both ablations**; `loss_decreasing` is `False` everywhere. The three configurations report
an **identical** top-1 (0.0925) because a diverged model's policy is the same degenerate object in
all three — the deltas of 0.0000 are not "the organ contributes nothing", they are "the model was
destroyed before it could use any organ". **The top-1 of 0.0925 is therefore uninformative**, and the
ablation deltas are void. The rule still computes INCONCLUSIVE (0.013 < 0.0925 < 0.30), and that is
what is recorded — but the honest reading is that this shot measured a divergence, not a model.

### Diagnosis — the cause, reproduced

`scripts/repro_nan.py` trains the same model on a real subset at several learning rates:

```
lr=0.05   first=3.4450  last_finite=27.6310  nan_count=0   ← loss EXPLODES (3.44 → 27.6, rising)
lr=0.01   first=3.4450  last_finite=3.3042   nan_count=0   ← decreasing
lr=0.005  first=3.4450  last_finite=3.1182   nan_count=0   ← decreasing
lr=0.001  first=3.4450  last_finite=3.0429   nan_count=0   ← decreasing
```

**Cause: the learning rate (0.05) is too high for this model.** At `lr=0.05` the loss climbs
monotonically and overflows to NaN within the full 40 000-position run; at `lr<=0.01` it decreases.
The v1 model tolerated `lr=0.05` because it is linear and shallow; the v2 model's bilinear multi-head
policy plus the history mixer has a much larger effective step, and 0.05 is past its stability edge.

### A gap in the verdict rule (recorded, not silently patched)

The pre-registered rule has no branch for a **diverged** run: a NaN loss falls through to
INCONCLUSIVE, which reads as "interesting but not convincing" when the truth is "the model was
destroyed". That is a defect in the *rule*, not a threshold to move. The fix — treat a non-finite
loss as a defect (like INVALID) — is a **new pre-registration** for the next shot, not an edit to
this one. It is written down here so it cannot be quietly forgotten.

## Step 3 — Fast lane: labeling economy (separate acceptance)

Real Stockfish 19 over **10 000 positions from the actual shards**: 168.6 s, **59.32 positions/s**,
9 464 engine calls, **10 536 cache hits** (52.7% served free), `timed_out=False`. The Phase-2 MOCK
debt is closed. Full report: `LOG/session-0009-labeling.md`.

## Step 4 — Milestone package

One package: the CI run above, the shot run above, the metrics artifact, this log, and the WEEK
digest. **One human signature** on the milestone; **tag v0.4.0 only after it.** The signature is not
written by the agent — never, in any form.

### The next question (for the next deep lane)

Re-run v2 at a stable learning rate (`lr=0.005`), with a **NaN guard** that fails the run loudly on a
non-finite loss, under a **new pre-registration** that adds the divergence branch to the rule. Only
then is a v1→v2 comparison meaningful.

---

*Recorded by the Caïssa-1 construction agent, session 0009. The verdict is the rule's own output,
written verbatim; the divergence is recorded with the same care as a success would have been.*
