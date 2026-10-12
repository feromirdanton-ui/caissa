# Session 0013 — divergence forensics (step 1)

**Descriptive, not a duel.** This run exists only to record *where* the v2 divergence starts. It is
an observation, not a verdict, and it does not decide anything about v1.

## The run

A short diagnostic (d256@K8, Adam `lr=0.005`, `forensics: true`, 4000 steps, 3-epoch cap). It is
recorded in the append-only registry as **0013-4** (the run that carried the trace) with its run URL.
The same trace is also produced inside the stability run, so the published artifact carries the same
numbers.

## What was recorded

At every step: the L2 norm of the gradient **per parameter group** (`encoder`, `mixer`, `head`,
`geometry`), the L2 norm of each learned tensor (embedding norms), and `max|logit|`. Full per-step
series in the artifact; the numbers below are read from
`forensics/metrics-forensics.json` (artifact `smoke-train-stability-metrics`, id `11754434343`).

## The result — the hypothesis is CONFIRMED

The pre-stated hypothesis was: *the bilinear `(from, to)` head explodes first, because its
multiplicative interaction produces unbounded logits; the history mixer is the second suspect.*

**The head exploded first — at step 3.** The encoder and the mixer followed at step 4, and the
geometry scalar never exploded at all.

| group | grad norm first | grad norm max | first explosion |
|---|---|---|---|
| **head** | 0.5344 | **2217.64** | **step 3** |
| encoder | 0.4133 | 2171.55 | step 4 |
| mixer | 0.5520 | 915.18 | step 4 |
| geometry | 1.6202 | 3.4034 | — (never) |

**`max|logit|` went 2.1207 → 2408.89, peaking at 4261.97.** The logits are unbounded, exactly as the
hypothesis said. The embedding norms ran away in step: `w_mix` 21.4818 → 182.3440, `w_enc`
19.5768 → 95.7407, `head_from` 11.1574 → 48.9513, `head_to` 11.1505 → 48.8306 — while `geom_scale`
*shrank* (1.0 → 0.9425).

Validation loss: **3.3722 → 25.1684** — a 7.46× blow-up, and the DIVERGED status fires (final > 2×
best). The failure is fast: the whole thing is visible within the first ~100 steps.

## The reading

The divergence is **not** a learning-rate accident and **not** a geometry-bias accident. It is the
bilinear head's **unbounded logits**: a multiplicative `(from, to)` product has no ceiling, so a
large update to `head_from`/`head_to` feeds straight into the logits and then back into the gradient.
The encoder and mixer explode one step later — they are pulled by the head's gradient, not the
cause.

**Consequence for the fix.** A **symptom dampener** (gradient clipping, F1) should not be enough,
because it caps the *step*, not the *quantity that can grow without bound*. The **cause fix** is to
bound the logits themselves — F2's cosine (unit-normalised) projections, whose logits cannot exceed
`scale × n_heads` however large the weights grow. The factorial (step 2) tests exactly that.

This is where the session's decision came from: the session asks *whether any configuration
converges*, and the forensics says the answer depends on whether the cause is removed or only
damped.
