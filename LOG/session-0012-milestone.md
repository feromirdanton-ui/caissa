# Milestone «Сила-2» — the package

**Question (frozen in `LOG/session-0011-preregistration.md`, before any code):** does any
configuration of the v2 family beat v1 at v1's own convergence limit, on the same eval set, by a
statistically significant margin?

**Answer:** **No.** Verdict **DEAD** — no v2-family configuration's top-1 exceeds the v1 reference on
the shared eval set. (The machine printed `SUCCESS`; that was a polarity defect in the rule's
implementation, corrected in `LOG/session-0012.md`. The raw artifact is preserved unmodified.)

## The table

All configurations scored on the **same** eval set (9 231 positions, `ply >= 8`), same split seed.
Deltas are top-1 differences against the v1 reference on that set; every delta carries a McNemar
test on the paired correctness vectors.

| configuration | top-1 | Δ vs v1@set | McNemar z | p | method | favours | significant | steps | plateau | val loss first → last | legality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1 reference** (tiny, sgd 0.05) | **0.1167** | — | — | — | — | — | — | 5 ep | yes | 3.1663 → 3.5408 | — |
| d256_k8 (main) | 0.0810 | **−0.0356** | −9.066 | 1.238e-19 | normal | b (v1) | yes | 10000 | yes | 3.3453 → 25.1846 | 1000/1000 |
| d128_k8 | 0.0801 | **−0.0366** | −9.085 | 1.032e-19 | normal | b (v1) | yes | 10000 | yes | 3.3440 → 22.4378 | 1000/1000 |
| d256_k0 | 0.0714 | **−0.0453** | −11.776 | 5.201e-32 | normal | b (v1) | yes | 10000 | yes | 3.3444 → 25.4160 | 1000/1000 |
| d128_k0 | 0.0731 | **−0.0435** | −11.031 | 2.699e-28 | normal | b (v1) | yes | 10000 | yes | 3.3455 → 22.2576 | 1000/1000 |

**Every delta is negative and every one is significant — significantly *worse* than v1.** The
significance is real; its direction is against the v2 family.

Set-local baseline (most frequent move on this set): 0.0113. All configurations clear it, so the
models are learning something — just less than v1.

## The frozen 2×2 — the capacity axis

| | K=0 | K=8 |
|---|---|---|
| **d=128** | 0.0731 | 0.0801 |
| **d=256** | 0.0714 | 0.0810 |

Reading: history (K=8) helps at both widths (+0.0087 at d=128, +0.0096 at d=256); width (d=256)
helps only with history (+0.0009 at K=8, −0.0017 at K=0). The largest cell is d256_k8 — still
0.0356 below v1.

## The v1@v2-set column (the slice effect)

v1 was retrained in the same run and scored on **both** sets: **0.1324** on its own set (10 000
positions, `ply >= 0`) and **0.1167** on the v2 set (9 231 positions, `ply >= 8`). The v2 set is
harder, so v1 loses 0.0157 to the slice — which is why the comparison uses v1@v2-set, not v1's own
number. This closes the confounder session-0009's preflight identified.

## Runs and artifacts

| what | URL / id |
|---|---|
| the shot | https://github.com/feromirdanton-ui/caissa/actions/runs/38251372886 (`head_sha` `bf379d66`) |
| metrics artifact | `smoke-train-v2-metrics`, id `11748652041`, sha256 `430e46b86a692807f1189845299cc91b6d27b7fc945355107b2aab17aaec1927` |
| materialize (publication) | https://github.com/feromirdanton-ui/caissa/actions/runs/38251100097 |
| materialize (void, platform limit) | https://github.com/feromirdanton-ui/caissa/actions/runs/38250168401 |
| session-0010 shot (prior) | https://github.com/feromirdanton-ui/caissa/actions/runs/38213524325 |

## Labeling bank (0009/0010)

Real Stockfish 19 on 10 000 positions from the shards: 168.6 s, 59.32 pos/s, 10 536 cache hits,
`timed_out=False`. The MOCK debt of Phase 2 is closed. See `LOG/session-0009-labeling.md`.

## What this milestone establishes

1. **The instrument works end to end** — data → shards → history windows → plateau-bounded training
   → McNemar on paired vectors → a verdict, all on real games, all reproducible.
2. **The v2 family does not beat v1 at this scale.** A negative result, recorded as a result.
3. **The v2 family diverges** at both tested learning rates (val loss 3.35 → 25.2, grad norm
   4.8 → 648). The open question is the architecture's **stability**, not its capacity.
4. **The plateau detector cannot tell a plateau from a divergence** — it reported `PLATEAUED` on a
   run that blew up. A defect for the next deep lane.

## Signature

**One human signature closes this milestone.** The agent does not write it, in any form. Tag
`v0.4.0` is created only after that signature.
