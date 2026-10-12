# Session 0010 — SHOT REGISTRY (append-only)

**This file is append-only.** Every dispatch of the session-0010 experiment is recorded here at the
moment of dispatch, with its run URL and `head_sha`, so the provenance chain
`pre-registration -> shot -> verdict` has no gap. A silent retry does not exist: a failed or
mis-targeted attempt stays in this file forever, marked as such.

The criterion is fixed in `LOG/session-0010-preregistration.md` (already on `main`, commit
`f6feb881`) and is **not** restated or adjusted here. The four branches are: SUCCESS top-1 >= 0.30,
DEAD top-1 <= the SET-LOCAL baseline, INCONCLUSIVE in between, and **INVALID** for any non-finite
loss/metric or legality < 100% (the branch session 0009 lacked).

---

## Carried forward — session 0009 (§6.7: the registry is append-only; nothing is deleted)

Recorded in `LOG/session-0009-shots.md` and left in place there; restated here so the 0010 chain is
readable in one file.

- **0009 shot 1 — VOID.** Run https://github.com/feromirdanton-ui/caissa/actions/runs/38186826762
  · `head_sha=ff7e2813` (the session-0008 HEAD, not the preflight commit). A ref-update/dispatch
  race; cancelled. A run on the wrong commit is not a result.
- **0009 shot 2 — THE SHOT.** Run https://github.com/feromirdanton-ui/caissa/actions/runs/38186843291
  · `head_sha=584517fd` · **verdict INCONCLUSIVE** — *"top-1 0.0925 beats the baseline but is below
  0.30"*. The run had **diverged** (`train_loss_curve` = `[NaN, NaN]` for the main run and both
  ablations; identical top-1 across all three), so the number was not a measurement. Kept forever.

---

## Session 0010

### Shot 1 — THE SHOT (correct commit; race-free dispatch verified)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38212904808
- **Dispatched:** 2026-10-12T06:31:03Z
- **Workflow:** `.github/workflows/smoke-train-v2.yml` (`workflow_dispatch`, ref `main`)
- **inputs:** `lichess_month=2013-01`, `train_positions=2000`, `shard_size=10000`
- **head_sha:** `0914f99d22771d8ea2ca2298ac064c6ced1d6e51` — verified equal to the merged commit
  produced by PR #18 (squash of `46323362`). The ref-update/dispatch race of 0009 did **not** recur:
  the dispatch resolved `main` **after** the merge, and the run's `head_sha` matches the intended
  commit. No VOID entry is needed for 0010.
- **What changed vs 0009:** `lr` is now fixed by prior (`0.01`, the top of the stable range, from
  `scripts/repro_nan.py`); the NaN guard aborts a diverged run RED; the baseline is **set-local**;
  v1 is retrained in-run and scored on both eval sets.
- **Status at dispatch:** in_progress
- **Verdict:** PENDING — to be recorded verbatim in `LOG/session-0010.md` when the run completes.
  A run without a URL is not a report; this line carries the URL.

---

*Registry opened by the Caissa-1 construction agent, session 0010. Entries are never edited or
removed; a superseded shot is marked VOID and left in place.*
