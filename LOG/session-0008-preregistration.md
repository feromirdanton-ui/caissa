# Session 0008 — PRE-REGISTRATION of the v2 criterion and the ablation questions

**Written BEFORE any run of `smoke-train-v2`.** This file is committed before the first dispatch,
so the repository history itself proves the thresholds preceded the numbers (ADR-0011 §5; GOAL.md
Article 5). It is not edited after the run, whatever the run says.

Three hypotheses are fixed here in advance: one for the main v2 configuration, and one for each of
the two micro-ablations. Every threshold was chosen **from the v1 measurement** — v1 top-1 was
**0.1839** on the same day-prefix — not from a wish.

---

## H1 — the scaled vertical (main run)

`configs/smoke-train-v2.yaml`: K=8 history, d_model 256, 8 heads, geometry bias on, 40 000 train
positions, 2 epochs, ≤90 min CPU. Data and split seed are **identical** to v1 (same day-prefix
shards, `seed: 0`, 95/5 by game), so the only differences from v1 are capacity, history and budget.

| outcome | condition | reading |
|---|---|---|
| **SUCCESS** | top-1 accuracy **≥ 0.30** | the scaled vertical earns its keep; go on building capacity |
| **DEAD** | top-1 accuracy **≤ the most-common-move baseline** | capacity did not buy learning; the wiring is wrong, not the size |
| **INCONCLUSIVE** | baseline **<** top-1 **< 0.30** | learning is real but the scale bet did not pay as hoped — a result, recorded as one |
| **INVALID** | sampling legality **< 100 %** | a defect to fix, never a finding (Article 1) |

SUCCESS was set at **0.30**, ~1.6× v1's 0.1839: a number that a working capacity increase can
reach and a broken one cannot. It is deliberately **above** anything the ablations could fluke into.

The val-loss condition from v1 is reported (first/last and the full curve) but is **not** part of
the verdict here: a rising loss with rising accuracy would be a real and interesting shape, and the
v2 question is about accuracy, so accuracy decides. Recording the curve without making it a gate is
the honest choice — a criterion invented after seeing the curve would not be a criterion.

## H2 — ablation #1: does the geometric prior pay for itself?

**Removed:** the 64×64 Chebyshev bias on the policy logits (`model.geometric_bias: false`). Nothing
else changes; the bias contribution is exactly zero when off, so this is a controlled subtraction.

**Question, asked in advance:** does a bias built from board geometry buy top-1 accuracy that the
bilinear head would otherwise have to learn from data?

| outcome | condition | reading |
|---|---|---|
| **KEEPS IT** | main − ablation **≥ 0.02** top-1 | the prior pays; keep it on by default |
| **NEUTRAL** | \|main − ablation\| **< 0.02** | the head learned the geometry anyway; the prior is redundant and its cost (a fixed prior that can be wrong) is unpaid for |
| **HURTS** | ablation − main **≥ 0.02** | the prior constrains more than it helps; remove it from the default |

The 0.02 band is ~11 % of v1's accuracy — smaller than that and the gap is inside run-to-run noise,
which the project will not dress up as a finding.

## H3 — ablation #2: does history context pay for itself?

**Removed:** the K=8 history window (`history: 0`), so the model sees one board — the v1 information
set. Everything else — d_model 256, 8 heads, geometry bias on — stays at the main values.

**Question, asked in advance:** does giving the model the preceding 8 boards buy top-1 accuracy,
i.e. is a chess move a function of the game's trajectory rather than only the current frame?

| outcome | condition | reading |
|---|---|---|
| **KEEPS IT** | main − ablation **≥ 0.03** top-1 | sequence context is a real organ; deepen it next |
| **NEUTRAL** | \|main − ablation\| **< 0.03** | at this scale the extra input changes nothing; revisit only when capacity is far larger |
| **HURTS** | ablation − main **≥ 0.03** | the extra parameters cost more than the context is worth at this scale |

H3's band is **0.03** (wider than H2's): history adds `(K+1)×FEATURE_DIM×d_model` parameters, so its
noise floor is higher and a smaller gap would not be distinguishable from it. Fixing a wider band
for the noisier comparison *in advance* is what keeps "no difference" from being read as "hurt".

## What is falsified, and what a null means

- A **null** on H2 or H3 is not a failure of the session: it is the honest measurement of an organ's
  contribution at this scale, and it is what the map of contributions is built from.
- The **main H1 verdict stands regardless** of the ablations. If H1 is DEAD, that is recorded as the
  headline and the ablations are reported as measured.
- Ablations run at **40 000 positions / 1 epoch**; the main run at 40 000 / 2 epochs. This is stated
  so no reader compares an ablation's absolute number to the main run's as if the budgets matched —
  only the *differences* within the matching budget are claims.

## Retouching is forbidden

Whatever the outcome, it is recorded verbatim in `LOG/session-0008.md` with the run URLs and the
metrics artifacts. A negative result is recorded with the same care and is worth more than a
retouched positive (§9; Article 5). This criterion may not be edited after dispatch.

---

*Pre-registered by the Caïssa-1 construction agent, session 0008, before dispatch.*
