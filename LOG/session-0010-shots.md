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

### Shot 1 — DID NOT START (workflow defect; no measurement)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38212904808
- **Dispatched:** 2026-10-12T06:31:03Z · **ended:** 06:31:12Z (9 s) · **conclusion:** failure
- **Workflow:** `.github/workflows/smoke-train-v2.yml` (`workflow_dispatch`, ref `main`)
- **inputs:** `lichess_month=2013-01`, `train_positions=2000`, `shard_size=10000`
- **head_sha:** `0914f99d22771d8ea2ca2298ac064c6ced1d6e51` — the merged commit of PR #18. The
  0009 ref-update/dispatch race did **not** recur here; the dispatch itself was correctly targeted.
- **What actually happened (read from the run's own log, not inferred):** the job failed at **step
  1, “Set up job”**, before any training step could run:

  > `##[error]Unable to resolve action actions/setup-python@0b93645e9fe7318ecaed2b359559ac225c90a2b2,
  > unable to find version 0b93645e9fe7318ecaed2b359559ac225c90a2b2`

  The pinned action did not resolve, so the runner never started the job body: **zero artifacts**
  were uploaded and **no training happened**. This is not a result — it is a broken dispatch.
- **Cause:** the `actions/setup-python` pin in the workflow dropped one hex character (`a`) relative
  to the valid v5.3.0 SHA that `ci.yml` already carries (`0b93645e9fea7318ecaed2b359559ac225c90a2b`;
  both are 40 hex chars, so the bad one is well-formed but is not a real commit). Verified against
  the API: the good SHA returns HTTP 200, the bad one HTTP 422; the file's other two pins (checkout,
  upload-artifact) resolve.
- **Verdict:** **none — no measurement.** Recorded here per the append-only rule and **left in
  place**; this entry is *amended at 06:39Z* to say what the run's log shows (it was opened as “THE
  SHOT” before the log was read; it never started). Nothing is deleted.
- **Followed by:** shot 2 (below) — the re-fire after the one-line pin fix. Not a silent retry: the
  failed attempt stays.

---

### Shot 2 — THE SHOT (re-fire after the pin fix; correct commit; race-free)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38213524325
- **Dispatched:** 2026-10-12T06:38:35Z · **trigger:** `workflow_dispatch`, ref `main`
- **inputs:** identical to shot 1 (`lichess_month=2013-01`, `train_positions=2000`, `shard_size=10000`)
- **head_sha:** `fd16b0913e80737a9ffd8f9c930da2e6ff792e15` — the pin-fix commit, verified equal to
  the run's `head_sha` at the moment of dispatch (ref/head match; no VOID).
- **What changed vs shot 1:** exactly one line — the `actions/setup-python` pin restored to the valid
  v5.3.0 SHA. The pre-registered criterion, thresholds, `lr`, and every other file are untouched.
- **Status at dispatch:** in_progress
- **Verdict:** PENDING — to be recorded verbatim in `LOG/session-0010.md` when the run completes.

---

*Registry opened by the Caissa-1 construction agent, session 0010. Entries are never edited or
removed; a superseded shot is marked VOID and left in place.*
