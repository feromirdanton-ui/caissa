# Session 0013 «Стабильность» — deep lane: divergence diagnosis

**Question:** does ANY configuration of the v2 family converge at all? Session-0012 established
*that* the v2 family diverges (val loss 3.3453 → 25.1846, grad norm 4.82 → 648) but not *why*. This
session asks the narrower, answerable question, and does **not** run the duel with v1 — a
configuration earns the right to a rematch by converging first, under its own pre-registration
(0014).

## Precondition — the «Сила-2» signature

Read and quoted verbatim, with its URL, as §6.7 requires:

> «Сила-2» — ACCEPTED», Feromir Danton 16:46 12.10.2026

- URL: https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6125480077
- id `6125480077` · **author_association: OWNER** · 2026-10-12T13:42:06Z

The milestone «Сила-2» is closed by that signature. The agent did not write it, in any form. Tag
`v0.4.0` is created only after that signature — and it is not created here.

## Step 0 — the milestone book

| milestone | closed by | state |
|---|---|---|
| «Вертикаль» | signature `MILESTONE VERTICAL — ACCEPTED`, issue #17 comment `6114946330` (OWNER, 2026-10-11T23:10:19Z) | **closed** |
| «Сила и голос» (0008–0010) | — | **CLOSED BY SUPERSESSION** in «Сила-2» |
| «Сила-2» | signature `«Сила-2» — ACCEPTED», issue #17 comment `6125480077` (OWNER, 2026-10-12T13:42:06Z) | **closed** |
| tag `v0.4.0` | — | **not created** — only after the «Сила-2» signature |

**«Сила и голос» — CLOSED BY SUPERSESSION in «Сила-2»:** its central question (v2 versus v1 at
*equal steps*) was annulled by a confounder — the two families were judged at the same step count
but different learning rates, so the comparison measured the budget, not the architecture — and the
question was re-addressed by the convergence duel of «Сила-2». The count reconciles: three
milestones named, two closed by a human signature, one closed by supersession, one tag withheld.

## Step 1 — divergence forensics (descriptive, not a duel)

Full report: `LOG/session-0013-forensics.md`. The short diagnostic (d256@K8, Adam `lr=0.005`,
`forensics: true`) recorded the per-group gradient norm, the embedding norms and `max|logit|` at
every step.

**The hypothesis is CONFIRMED: the bilinear head explodes first, at step 3.** The encoder and mixer
follow at step 4; the geometry scalar never explodes.

| group | grad norm first | grad norm max | first explosion |
|---|---|---|---|
| **head** | 0.5344 | **2217.64** | **step 3** |
| encoder | 0.4133 | 2171.55 | step 4 |
| mixer | 0.5520 | 915.18 | step 4 |
| geometry | 1.6202 | 3.4034 | — (never) |

`max|logit|` went **2.1207 → 2408.89, peaking at 4261.97** — the logits are unbounded, exactly as the
hypothesis said. The reading: the divergence is the head's **unbounded logits**, not a learning-rate
or geometry accident. A symptom dampener (clipping) should not be enough; the cause fix is to bound
the logits themselves.

## Step 2 — pre-registration «Стабильность»

`LOG/session-0013-preregistration.md`, written before any code. The F0–F4 factorial as config flags;
the CONVERGES / DIVERGES / BETWEEN rule with its thresholds (2× best, 10× grad, legality 100%)
recorded before the shot. The duel with v1 is not run.

## Step 3 — the plateau detector and the verdict rule

1. **`PlateauTracker` gains a DIVERGED status**: a curve whose final val loss exceeds 2× its **best**
   is a divergence even though every number is finite. The comparison is against the best, not the
   first — v1's own curve rises 3.1663 → 3.5408 (1.12×) because it overfits, and calling that a
   divergence would be wrong.
2. **One function decides PLATEAUED vs DIVERGED** (`classify_curve`), shared by the detector and the
   run guard. Session-0012's defect was that the two disagreed about the same curve.
3. **Mutation control on both real 0012 curves**: the v2 curve must classify DIVERGED, the v1 curve
   must not. The test carries the actual numbers from the artifact.
4. **The run guard is extended**: DIVERGED ⇒ RED run + FAILURE artifact, exactly as NaN/Inf does.
5. **The 0012-3 verdict stays DEAD** — a frozen rule is not rewritten after the fact. Interpretive
   caveat recorded: *the duel was run against a non-converging contender*, so DEAD means "v2 did not
   beat v1 **while diverging**", not "v2 cannot beat v1".

## Step 4 — McNemar within the family

The within-family comparison (K8 vs K0 at each width) is implemented in the driver and reported in
the artifact's `within_family` block, with its own paired z and p. **It could not be computed from
the session-0012 artifact**: that artifact carries only the marginal 2×2 against v1, not the paired
per-position correctness vectors, so a K8-vs-K0 test cannot be reconstructed from it. Rather than
fabricate a number, the comparison is computed by the run that has the vectors — this session's
factorial — and read from its artifact. Deltas are written from unrounded artifact values.

## Step 5 — fast lane

1. **Transport (ADR-0012 addendum).** The payload store is now **GitHub Releases** (persistent),
   with **litterbox.catbox.moe (72 h)** as the fallback; the key lives in a repo secret, never in
   workflow text; rotation is required if the repository's visibility changes. See
   `DECISIONS/adr-0012-publish-transport.md`.
2. **WEEK versus ADR-0011 — decided explicitly: the WEEK digest is KEPT, not abolished.** ADR-0011
   §3 abolishes *ritual without a question*; the WEEK digest answers a real question — "what changed
   in the project's truth this week, and what is the next stone?" — and it is the only artifact that
   carries the *narrative* across sessions. It is therefore a fast-lane **check** (a report), which
   ADR-0011 §1 explicitly admits. No silent drift: the decision is recorded here and in the ADR
   addendum.
3. **Which checkpoint was scored in 0012 — the FINAL one, not the best.** The artifact's
   `convergence.best_eval` is `1`, i.e. the best evaluation was the *first*, and the scored
   checkpoint was the last (the destroyed one). This is why 0012's top-1 (0.0810) is a measurement of
   a blown-up model, and it is a second, independent reason the 0012 duel is not a fair comparison.

## The shot

The shot is recorded in the append-only registry `LOG/session-0013-shots.md` with its run URL and
`head_sha`, verified against the polled `refs/heads/main` before the run was read. The verdict is
read from the metrics artifact, never from memory.

## Risks

- The v2 family's divergence is now attributed (unbounded bilinear logits) and a cause fix exists
  (cosine logits), but whether it converges at scale is the factorial's question, not this session's
  assumption.
- The transport depends on a public 72 h host as a fallback; GitHub Releases is the primary store.
- The 0012 duel was run against a non-converging contender; its DEAD verdict is frozen but carries
  that caveat.
