# Session 0011 pre-registration — «Сила-2» (the convergence duel)

**Written BEFORE any code of this session.** This file is the falsification criterion of the run it
precedes. Once the shot is fired, nothing in it moves: not the thresholds, not the optimizer mode,
not the budget. The shot registry is append-only.

**Repo:** https://github.com/feromirdanton-ui/caissa · **Branch:** `main`
**Session starts from HEAD:** `193a581f` (session-0010 verdict commit)

---

## 0. Precondition — the milestone signature (read, quoted, never written by the agent)

GOAL.md §5 makes a milestone close only on a green run + tag + **human signature** (Article 13), and
ADR-0011 makes it **one human signature per milestone**. Session 0011 may only start once the
«Vertical» milestone carries that signature. The precondition is therefore a read, not an inference.

Signature, **verbatim** from issue #17:

> «MILESTONE VERTICAL — ACCEPTED»

Source (URL in LOG as the task requires):
https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6114946330

The most recent owner comment on the same issue is a forward instruction, also quoted verbatim
because it names the *next* signature and the tag that may follow it:

> Затем — «PHASE/MILESTONE — ACCEPTED», тег v0.4.0.

https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6123519894

**Gate result: OPEN.** The milestone signature exists, so §6.7 (post-phases evergreen mode, stop) is
**not** triggered. The `v0.4.0` tag is **not** written by this session — only one human signature per
milestone, and the agent never writes a signature or the tag that follows it.

---

## 1. The question

> Does **any** configuration of the v2 family beat **v1 at v1's own convergence limit**, measured on
> the **same** eval set, by a statistically significant margin?

Session-0010 ran v2 and both ablations for a **fixed** number of steps, at one learning rate, and the
verdict was INCONCLUSIVE (top-1 0.1150 vs a set-local baseline of 0.0150, below the 0.30 marker).
Session-0011 changes the *measurement instrument*, not the model family:

- **Equal steps judge the stopwatch, not the players.** Two models trained for the same step count
  but at different effective progress are not being compared; the faster-converging one is merely
  cut off earlier. Session-0011 therefore trains every configuration **to its own plateau**.
- The `lr=0.05` (v1, stable) vs `lr=0.005` (v2, stable) gap recorded in session 0009 is a symptom of
  **conditioning, not capacity**. One recorded optimizer choice is made below to address that.

## 2. Budget — a plateau, not a step count

| item | value | why this number |
|---|---|---|
| `plateau_patience` | **5** evaluations | stop after 5 consecutive val-loss evaluations with no improvement |
| `plateau_eps` | **1e-4** | an "improvement" must beat the best-so-far by more than this absolute margin, so float jitter cannot buy patience forever |
| `max_epochs` | **20** | the **hard ceiling**; a run that hits it without plateauing is recorded as INCONCLUSIVE ("ceiling before plateau"), never silently truncated |
| `min_epochs` | **6** | patience cannot fire before 6 evaluations, so a lucky epoch-1 minimum is not mistaken for convergence |
| `eval_every_epochs` | 1 | val loss is measured every epoch on the **10k eval set** |
| `max_positions_to_load` | 200 000 | ~10k positions land in val at a 5 % by-game split |
| `max_train_positions` | 60 000 | training positions per epoch, declared rather than hidden |
| `max_val_positions` | **10 000** | the task's eval size (was 400 — a 400-position eval cannot resolve a 2-point McNemar delta) |
| `budget_minutes` | **90** | the runner constraint, stated |

**Steps consumed are recorded per configuration** in the metrics artifact (`steps`, `epochs_run`,
`plateau_reached`), because "trained to plateau" is only meaningful if the plateau is visible.

**Cost, measured on the sandbox before this file was committed** (`OMP_NUM_THREADS=1`, steady state,
real windows; the default 64-thread setting thrashes small matmuls and inflates K=8,d=256 from
0.53 ms to 163 ms per step — a measurement artefact, not architecture cost):

| configuration | ms/step | 60k steps (one epoch) |
|---|---|---|
| K=8, d_model=256 | 0.53 | ~32 s |
| K=8, d_model=128 | 0.36 | ~22 s |
| K=0, d_model=128 | 0.14 | ~9 s |
| K=0, d_model=256 | 0.22 | ~13 s |

Worst case (every configuration exhausting `max_epochs=20` at K=8,d=256 plus a 10k eval per epoch)
is ≈ 25 min of CPU — inside the 90-minute budget with room for the v1 reference and the McNemar pass.

## 3. Optimizer — one recorded choice, made now

**SGD → Adam.** Recorded **before** the run, as the task requires.

- `optimizer: adam`
- `lr: 0.005`
- `betas: [0.9, 0.999]`, `eps: 1e-8` (Adam's defaults, recorded because they become part of the run)

**Motive, stated so it cannot be revised later:** the session-0009 gap (v1 stable at `lr=0.05`, v2
diverging above `lr=0.01`) is the price of a badly conditioned objective, not evidence of lower
capacity. Adam normalizes each parameter's step, so a single `lr` is meaningful across the two
architectures. The `0.05/0.005` chasm is the thing being removed, not a result being flattered.

**v1 keeps its own optimizer/lr.** The v1 reference is retrained with its historical stable setting
(`sgd`, `lr=0.05`) — a learning rate is part of an architecture's price, so neither model is
handicapped to make the other look good. Both are trained to the **same plateau rule**.

## 4. Eval — 10k positions and a significance test on every delta

1. **One eval set for everything.** The set is built ONCE with the main run's `K` and shared by the
   main run, every ablation and the v1 reference (the session-0009 preflight fix, retained). This is
   what makes a delta a controlled difference rather than a change of ruler.
2. **10 000 positions** (was 400). A McNemar test on a 400-position set cannot resolve a small delta;
   the larger set is what makes "significant" an honest word.
3. **McNemar for ALL deltas.** For every pairwise comparison against the v1 reference (and against
   the main configuration) the artifact carries the full 2×2 discordant table plus the test
   statistic `z` and the two-sided `p`:

   | | v1 wrong | v1 right |
   |---|---|---|
   | **v2 right** | `only_v2_right` | `both_right` |
   | **v2 wrong** | `both_wrong` | `only_v1_right` |

   `z = (only_v2_right − only_v1_right) / sqrt(only_v2_right + only_v1_right)`, exact binomial
   `p` when the discordant total is < 25, normal approximation otherwise. Every cell is recorded,
   so a reader can recompute the p-value without the code.
4. **Cross-run comparisons** (against v1's historical 0.1839, or session-0010's numbers) are made in
   the interpretative note only, never in the table — different eval sets, different rulers.

## 5. Capacity axis — the factorial closes inside the v2 family

The task's axis item: add **`d128@K8`**. To close `{128,256} × {0,8}` inside the family, **`d128@K0`**
is added as well (the `256×K0` cell already exists as the `no_history` ablation, and `256×K8` is the
main run). Four frozen cells of one 2×2:

| cell | config | role |
|---|---|---|
| `d256_k8` | main (K=8, d=256, geom on) | the scaled vertical |
| `d128_k8` | K=8, d=128, geom on | **capacity axis at fixed history** |
| `d256_k0` | K=0, window K=8 | **history axis at fixed capacity** |
| `d128_k0` | K=0, d=128, window K=8 | closes the factorial |

All four share the split seed, the eval set and the plateau rule. Geometry bias stays **on** in all
four (it was ablated in session 0010; re-ablating it would confound this session's axis).

**Optional, hand of data:** the task offers "+2 day-prefix — a starvation test". It is **not** taken
this session, and the reason is recorded rather than skipped silently: the run already spends its
90-minute budget on four configurations × plateau × a 10k eval, and a second day-prefix would change
the eval set at the same time as the data volume, confounding the axis it is meant to test. It is
proposed as the next session's experiment, where it can be isolated.

## 6. Thresholds — recorded, then frozen

Per the task, for **this** session:

| outcome | rule |
|---|---|
| **SUCCESS** | at least one v2-family configuration beats the v1 reference **on the same eval set** with McNemar **p < 0.05** in its favour |
| **DEAD** | **no** v2-family configuration beats v1 (no configuration's top-1 exceeds v1@set) |
| **INCONCLUSIVE** | the ceiling (`max_epochs`) was reached before any plateau — the comparison is not yet meaningful |
| **INVALID** | any non-finite loss/metric (the instrument is destroyed) or legality < 100 % — a defect, not a science result (retained from session 0010) |

**`top-1 ≥ 0.30` is a progress marker, not a gate.** It is reported as `progress_marker_reached`
and never decides the verdict.

**Plateau on the v1 hyper-parameters is itself a result**, and is recorded with the same care as a
win: a plateau at a low number says the *family* is where the ceiling is, not the schedule.

## 7. Instruments carried over (non-negotiable)

- **NaN guard** (session 0010): a non-finite loss or parameter aborts the run **RED** and writes a
  `FAILURE-*.json`; the metrics writer refuses to persist a non-finite number.
- **Gradient norm** is logged every N steps (this session's fast-lane addition): a diverging run
  shows the norm climbing before the loss reaches NaN.
- **Legality gate**: 1000 distinct positions, the policy's top-1 move, 100 % legal (Article 1).
- **Append-only shot registry**: every dispatch — including one that never starts — stays in
  `LOG/session-0011-shots.md` forever. A silent retry does not exist.
- **Weights are a disposable run artifact**, never a git object (Article 8).

## 8. What this session does NOT do

- It does not write the human signature, and it does not create `v0.4.0`.
- It does not move a threshold, an `lr`, or the budget after the shot.
- It does not call a green run a result if its metrics are non-finite (session 0009's failure mode).

---

*A pre-registration is a promise made to a future reader who has already seen the numbers. It is
worth exactly as much as the discipline that leaves it unedited.*
