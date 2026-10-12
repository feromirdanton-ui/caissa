# Session 0009 — PREFLIGHT (written BEFORE the shot)

**Written before `smoke-train-v2` is dispatched.** This file records the two preflight checks the
session-0009 brief requires, and it is committed before the run so the repository history proves
the checks preceded the numbers. The pre-registered thresholds in
`LOG/session-0008-preregistration.md` are **not touched** — this is a confounder fix and an
interpretation note, not a change of criterion.

---

## 1. The eval set is unified across the main run and both ablations

**Finding: it was NOT, and the confounder is now fixed.**

`_train_once` built the history window with the *configuration's own* `K`. The main run uses `K=8`,
so it is scored on positions with `ply >= 8` (a window needs 8 predecessors). The `no_history`
ablation sets `K=0`, so it was scored on positions with `ply >= 0` — a **larger, different set**
that includes the opening plies. A top-1 delta between them would therefore have mixed the
component's effect with a change of eval set: a confounder, not an ablation.

**Fix.** The window is now built **once** with the main run's `K` (`window_history`, defaulting to
`history`), and a model with a smaller `K` reads only the **suffix** of that same window
(`_hist_matrix` takes the last `history + 1` boards). So:

- the **main run is unchanged** — `window_history` defaults to `history = 8`, so its eval set is
  exactly the one the pre-registration describes;
- `no_geometry_bias` is unchanged (same `K`, same set);
- `no_history` (`K=0`) now reads a single board **of the same window**, so it is scored on the
  **same positions** as the main run — only its input shrank.

**Pinned by a test.** `tests/unit/test_smoke_train_v2.py::test_eval_set_is_identical_across_main_and_ablations`
asserts `val_samples` and `val_games` are identical for the main run and both ablations. Mutation
check (recorded in `LOG/session-0009.md`): reverting the fix makes the test **red** — the `K=0`
ablation then reports 34 val samples against the main run's different count.

## 2. Interpretation note — the v2 eval set relative to v1, and its direction

**The v2 eval set is a strict subset of v1's, and it is HARDER. The v1 → v2 comparison is therefore
conservative.**

- **v1** (`numpy_tiny`, `K=0`) was evaluated on **all** positions of the validation games
  (`ply >= 0`), including the opening.
- **v2** (main, `K=8`) is evaluated only on positions with `ply >= 8` — the first eight plies of
  every game are excluded, because a history window cannot be filled there.

**Direction.** Opening plies are the most predictable moves in chess: they are few, heavily
repeated, and dominated by a small set of book moves. Including them **inflates** top-1 accuracy.
By excluding them, v2 is measured on a **harder** set than v1 was. The consequence is stated
plainly: **any v2 improvement over v1's 0.1839 is understated, not overstated** — v2 has to earn its
number on a set where the easy opening moves have been removed. This is the honest direction to
record, and it is why the pre-registered SUCCESS bar (0.30) is not lowered to "compensate": a
criterion adjusted for a known-harder set after the fact would not be a criterion.

**What this note does NOT do.** It does not move a threshold, and it does not re-open the
pre-registration. The four outcomes (SUCCESS / INCONCLUSIVE / DEAD / INVALID) and their numbers are
exactly as committed in `LOG/session-0008-preregistration.md`.

---

*Preflight recorded by the Caïssa-1 construction agent, session 0009, before dispatch.*
