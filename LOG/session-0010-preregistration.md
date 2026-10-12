# Session 0010 -- PRE-REGISTRATION of the reloaded v2 criterion (RELOAD)

**Written BEFORE any run of `smoke-train-v2` in session 0010.** This file is committed before the
first dispatch, so the repository history itself proves the thresholds preceded the numbers
(ADR-0011 sec.5; GOAL.md Article 5). It is **not** edited after the shot, whatever the shot says.

Session 0009 fired the session-0008 criterion once and the run **diverged** (`train_loss_curve`
`[NaN, NaN]` for the main run and both ablations, all three reporting an identical top-1). The
number was not a measurement. This reload fixes the *instrument* and the *rule*, then fires again --
it does **not** move a threshold to manufacture a result.

**Provenance chain:** session-0008 (the experiment + its criterion) -> session-0009 shot 1 (VOID,
wrong commit) -> session-0009 shot 2 (the shot; diverged) -> **session-0010 (this reload)**. The shot
registry (`LOG/session-0010-shots.md`) is **append-only**: every attempt stays forever, including
the 0009 rows, which are never edited.

---

## 1. Thresholds -- UNCHANGED

SUCCESS / DEAD / INCONCLUSIVE keep the numbers written in session 0008 and are **not** renegotiated:

| outcome | condition |
|---|---|
| **SUCCESS** | top-1 accuracy **>= 0.30** |
| **DEAD** | top-1 accuracy **<= the most-common-move baseline** |
| **INCONCLUSIVE** | strictly between the two above |

SUCCESS stays at **0.30** (~1.6x v1's historical 0.1839): reachable by a working capacity increase,
not by a broken one. Moving it because session 0009 was inconclusive would be exactly the
retouching ADR-0011 sec.5 forbids.

## 2. NEW branch -- INVALID (the instrument is destroyed)

Added because session 0009 lacked it: a diverged run fell through to INCONCLUSIVE, which reads as
"interesting but not convincing" when the truth is "the model was destroyed". The rule now says:

> **INVALID** iff **any** non-finite `train`/`val` loss **or** any non-finite metric (top-1, baseline,
> loss curve entry) iff **"the instrument is destroyed"**, **regardless of top-1**. Sampling legality
> **< 100 %** remains INVALID as before.

INVALID is a **defect, not a finding**: the run is **RED** (exit non-zero), a FAILURE artifact is
written in place of the metrics, and no verdict word is available. This is enforced in code, at two
levels, so no future loop can forget it:

1. **In the training loop** -- `smoke_train_v2_loop.train_configuration` checks the loss is finite
   on *every* step (a non-finite loss raises `DivergenceError`) and checks the WEIGHTS every
   `guard_every` steps (`parameters_finite`), because a finite loss this step does not prove finite
   weights -- a large update can push a weight to `inf` and the next step is the first to notice.
   On divergence it writes `FAILURE-<label>.json` (schema `caissa-smoke-train-v2-failure/1`)
   carrying the cause and the partial curve, and re-raises `SystemExit(3)`.
2. **In the writer** -- `smoke_metrics.assert_finite` walks the metrics structure and refuses to
   persist a non-finite number, so the artifact that lands is either honest or absent.

A green run with a dead model is thereby architecturally impossible -- the last member of the
"green but dead" family is closed.

## 3. Baseline -- recomputed on the ACTUAL eval set (set-locality)

The DEAD threshold now compares top-1 against the **most-common move of the eval set itself**
(`set_local_baseline`), not the training set's most-common move measured on the validation set.

**Motivation.** A constant policy cannot beat the most-common move of its *own* set -- that move's
frequency is the strict upper bound for any constant predictor. Session 0009 compared top-1
**0.0925** against a baseline of **0.0132** that was computed as "the training set's most common
move, scored on the validation set". Those are two different quantities measured with two different
rulers: a model can "beat" 0.0132 by playing a move the eval set barely contains. The set-local
baseline makes the model and the baseline share **one ruler**.

The cross-ruler number is **kept in the artifact for transparency only** (`baseline.cross_ruler_*`)
so a reader can see both rulers and why the DEAD threshold moved to the set-local one -- but it is
**not** used for any verdict.

Pinned by a test (`tests/unit/test_baseline_set_locality.py`): a set whose most-common move differs
from the training set's must produce a **different** set-local baseline, and the DEAD branch must be
decidable against it.

## 4. Learning rate -- fixed A PRIORI, one number: **lr = 0.01**

Chosen **before the shot** from the session-0009 reproduction already in the repo
(`scripts/repro_nan.py`): `lr=0.05` diverges (loss 3.44 -> 27.6, rising); `lr<=0.01` decreases.
This session takes the **top of the stable range, 0.01** -- the more demanding of the two safe
values, not the gentlest. It is fixed here and **not moved after the shot** (task rule: "thresholds and lr do not move after the shot").

Interpretation rule: **an lr is part of an architecture's price.** v1 keeps its own stable rate
(`v1_lr = 0.05`); v2 runs at its own stable rate (`0.01`). Neither model is handicapped to flatter
the other -- a comparison of two architectures, each at a rate it can actually train at.

## 5. Ruler control -- v1 retrained in the SAME run, scored on BOTH sets

v2 is measured on the **ply >= K** set (K=8; the session-0009 preflight's controlled surface). v1's
historical 0.1839 was measured on the **ply >= 0** set. Comparing v2's number to that historical v1
number would mix the organ effect with a **change of ruler**, so this session retrains v1 (the tiny
linear model) **inside the same run**, on the **same split seed and the same train games**, and
scores it on **both** eval sets, producing two columns:

- `v1@v1-set` -- the historical reading, reproduced (sanity check on the reproduction);
- **`v1@v2-set` -- v1 on the SAME positions v2 is scored on. This is the honest v1->v2 comparison,
  because only the model differs, not the ruler.**

The relocation of the eval set is one-directional and known: dropping the opening half-moves makes
the ply >= K set **harder**, so any improvement of v2 over v1's *historical* number is, if anything,
**understated** -- but the `v1@v2-set` column is what the milestone table is built on, not that
argument.

## 6. Append-only registry

`LOG/session-0010-shots.md` records every dispatch **at the moment of dispatch** with its run URL
and `head_sha`, so the chain `pre-registration -> shot -> verdict` has no gap. A shot that starts on
the wrong commit is a **mis-targeted shot**: it is cancelled and marked **VOID** and **left in the
file**. The session-0009 rows are copied in unchanged and never edited. There are no silent retries.

## 7. Retouching is forbidden

Whatever the outcome -- SUCCESS, DEAD, INCONCLUSIVE, or INVALID -- it is recorded **verbatim** in
`LOG/session-0010.md` with the run URL and the metrics artifact. A negative result is recorded with
the same care as a positive one and is worth more than a retouched positive (sec.9; Article 5). This
criterion may not be edited after dispatch.

---

*Pre-registered by the Caissa-1 construction agent, session 0010, before dispatch. Weights are a
disposable run artifact, never a git object (Article 8).*
