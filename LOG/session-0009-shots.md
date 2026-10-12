# Session 0009 — SHOT REGISTRY (append-only)

**This file is append-only.** Every dispatch of the session-0009 experiment is recorded here at the
moment of dispatch, with its run URL, so the provenance chain
`pre-registration → shot → verdict` has no gap. A silent retry does not exist: a failed or
mis-targeted attempt stays in this file forever, marked as such.

The criterion is fixed in `LOG/session-0008-preregistration.md` and is **not** restated or adjusted
here. The preflight (unified eval set + interpretation note) is in
`LOG/session-0009-preflight.md`, committed before the first shot.

---

## Shot 1 — VOID (wrong commit; superseded, not deleted)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38186826762
- **Dispatched:** 2026-10-12T00:07:41Z
- **head_sha:** `ff7e2813` — **the session-0008 HEAD, NOT THE PREFLIGHT COMMIT.**
- **Why void:** a race between `GITHUB_UPDATE_A_REFERENCE` and the dispatch. The dispatch resolved
  `main` before the new commit `584517fd` was visible to the workflow resolver, so the run started
  on the OLD tree — it would have measured the *confounded* eval set the preflight exists to fix.
  A run on the wrong commit is not a result; it is a mis-targeted shot.
- **Action:** cancelled (`GITHUB_CANCEL_WORKFLOW_RUN` → 202 Accepted). Kept here, not deleted —
  the registry is append-only.

## Shot 2 — THE SHOT (correct commit)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38186843291
- **Dispatched:** 2026-10-12T00:07:58Z
- **head_sha:** `584517fd` — the preflight commit (unified eval set). **This is the shot.**
- **Workflow:** `.github/workflows/smoke-train-v2.yml` (`workflow_dispatch`, ref `main`)
- **Status at dispatch:** pending → in_progress
- **Verdict:** PENDING — to be recorded verbatim in `LOG/session-0009.md` when the run completes.

---

*Registry opened by the Caïssa-1 construction agent, session 0009. Entries are never edited or
removed; a superseded shot is marked VOID and left in place.*

---

## Shot 2 — VERDICT RECORDED (appended after the run completed)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38186843291
- **conclusion:** `success` (all steps green) · **head_sha:** `584517fd`
- **Verdict (verbatim, the rule's own output):** **INCONCLUSIVE** —
  *"top-1 0.0925 beats the baseline but is below 0.30"*
- **Honest reading:** the run **diverged** (`train_loss_curve` = `[NaN, NaN]` for the main run and
  both ablations; identical top-1 0.0925 across all three). The number is not a measurement; the
  ablation deltas are void. Cause reproduced: `lr=0.05` is past this model's stability edge
  (`scripts/repro_nan.py`). Full analysis: `LOG/session-0009.md`.
- **Metrics artifact:** `smoke-train-v2-metrics` (id `11719948779`).
