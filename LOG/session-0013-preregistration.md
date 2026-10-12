# Session 0013 «Стабильность» — pre-registration

**Written BEFORE any code of this session ran.** The criterion below is frozen: retouching it after
seeing the numbers is forbidden (ADR-0011 §5, GOAL.md §9 Article 5). A negative result is recorded
and signed with the same care as a positive one.

## Precondition — the «Сила-2» signature

Read and quoted verbatim, with its URL, as §6.7 requires:

> «Сила-2» — ACCEPTED», Feromir Danton 16:46 12.10.2026

- URL: https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6125480077
- id `6125480077` · **author_association: OWNER** · 2026-10-12T13:42:06Z

The milestone «Сила-2» is closed by that signature. The agent did not write it, in any form. Tag
`v0.4.0` is created only after that signature — and it is not created here.

## The question

**Does ANY configuration of the v2 family converge at all?**

This is deliberately narrower than session-0011's question. Session-0012 established *that* the v2
family diverges (val loss 3.3453 → 25.1846, grad norm 4.82 → 648, max 3390) but not *why*. Until a
configuration converges, a duel with v1 is meaningless — a diverged contender has no top-1 to
compare. So this session asks only the convergence question, and the duel is **not** run.

## The F0–F4 factorial — config flags, not code branches

Every fix is a flag in `configs/smoke-train-stability.yaml`. The same training loop, the same model
class, the same eval set, the same split seed. That is what makes this a controlled comparison
rather than five programs.

| cell | fix | mechanism | hypothesis |
|---|---|---|---|
| **F0** | base | none — the control | expected to DIVERGE (reproduces session-0012) |
| **F1** | grad clip | global-norm clipping, `max_grad_norm: 1.0` | a symptom dampener; may not be enough |
| **F2** | cosine logits | unit-normalised from/to projections, fixed scale 8.0 | removes the CAUSE: unbounded bilinear logits |
| **F3** | optimizer | SGD at `lr=0.01` (the stable range session-0009 found) | the 0.05/0.005 gap is a conditioning price |
| **F4** | F1 + F2 | clip AND cosine logits | both together |

**The hypothesis under test** (stated, not assumed): the divergence's cause is the bilinear
`(from, to)` head — its multiplicative interaction produces unbounded logits, so a large update to
`head_from`/`head_to` feeds back into the logits and then into the gradient. If that is right, **F2
converges and F1 does not** (clipping damps the symptom; cosine logits remove the mechanism). The
forensic run of step 1 tests the *where*; this factorial tests the *fix*.

## The decision rule — pre-registered, frozen

Applied per configuration, to that configuration's own curve:

- **CONVERGES** = final val loss **<** first val loss **AND** grad norm never exceeds **10×** its
  first value **AND** legality is **100%**.
- **DIVERGES** = final val loss **> 2×** the **best** val loss.
- **anything between** = recorded as facts, not a verdict.

The comparison is against the **best**, not the first: v1's own curve rises 3.1663 → 3.5408 (1.12×)
because it overfits, and calling that a divergence would be wrong. The factor 2.0 sits between the
two real curves of session-0012 (v1 1.12×, v2 7.53×).

**The duel with v1 is NOT run this session.** A configuration earns the right to a rematch by
converging first, under its own pre-registration (0014).

## The instrument changes (also frozen before the shot)

1. **PlateauTracker gains a DIVERGED status.** A curve whose final val loss exceeds 2× its best is a
   divergence even though every number is finite. Session-0012's run reported `PLATEAUED` on a curve
   that went 3.35 → 25.2 — that must never be readable as a result again.
2. **One function decides PLATEAUED vs DIVERGED**, shared by the detector and the run guard. Two
   code paths for "what happened to this curve" is how session-0012's disagreement got in.
3. **The run guard is extended**: DIVERGED ⇒ RED run + FAILURE artifact, exactly as NaN/Inf does.
4. **Mutation control on both real 0012 curves**: the v2 curve must classify DIVERGED, the v1 curve
   must not. The test carries the actual numbers from the artifact, not a synthetic curve.
5. **NaN guard and gradient-norm logging** (from session-0010) are retained unchanged.

## The shot registry is append-only

Every dispatch attempt stays in `LOG/session-0013-shots.md` forever. Attempt **0011-0** ("build
without a shot") remains recorded as a fact. There are no silent retries.

## What would make this session INVALID

- Any non-finite train/val loss or metric ⇒ INVALID, regardless of top-1 (session-0010's branch).
- Legality < 100% ⇒ INVALID.
- A run that did not start, or started on the wrong commit ⇒ VOID, recorded, not a result.

## Thresholds and the optimizer are frozen

`max_grad_norm: 1.0`, `logit_scale: 8.0`, `lr: 0.01` (F3), the 2× and 10× factors, patience 5, eps
1e-4 — all recorded here, before the shot. None moves after the numbers are seen.
