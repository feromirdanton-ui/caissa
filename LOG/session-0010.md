# Session 0010 — «Перезарядка» (the reload): what / why / artifacts / risks

**Repo:** https://github.com/feromirdanton-ui/caissa · **Branch:** `main`
**Started from HEAD:** `f6feb881` (the session-0010 pre-registration, committed before any code)
**Landed:** PR #18 → squash `0914f99d`

---

## What this session is

Session 0009 fired the scaled vertical and it **diverged**: `train_loss_curve = [NaN, NaN]` for the
main run *and* both ablations, all three reporting an identical top-1 (0.0925), and the run still
finished **green** with an INCONCLUSIVE verdict. The number was not a measurement — the model was
destroyed and the pipeline could not tell.

Session 0010 is the **reload** the pre-registration named. It lands the code that makes a green
diverged run impossible and the v1→v2 comparison honest, then fires the shot.

## What landed (11 files, PR #18)

| path | role |
|---|---|
| `src/caissa/training/smoke_metrics.py` | `DivergenceError`, `parameters_finite`, `assert_finite`, `write_failure`, `set_local_baseline`, `cross_ruler_baseline`, accuracy/loss helpers |
| `src/caissa/training/smoke_train_v2_loop.py` | the guarded training loop + the four-branch verdict (INVALID first) |
| `src/caissa/training/smoke_train_v2_setup.py` | data prep (one split, one v2 window set) + the v1 reference retrained in-run |
| `src/caissa/training/smoke_train_v2.py` | driver + CLI |
| `src/caissa/testing/v2_support.py` | shared fixtures (`make_shards`, `make_config`, `poison`) |
| `scripts/smoke_train_v2_ci.py` | the workflow's shell logic (kept out of the ~8 KB YAML) |
| `tests/unit/test_smoke_train_v2.py` | machinery properties (legality, artifacts, deltas, split integrity) |
| `tests/unit/test_smoke_train_v2_guard.py` | the **mutation control**: a poisoned model turns the run RED |
| `tests/unit/test_baseline_set_locality.py` | the baseline is a property of *the set it rules* |
| `configs/smoke-train-v2.yaml` | K=8, d_model 256, 8 heads, geometry on + 2 ablations |
| `.github/workflows/smoke-train-v2.yml` | prefix → shards → verify → override → train → guard → metrics |

Plus `LOG/session-0010-shots.md` (append-only registry).

## The three additions the pre-registration names

1. **NaN guard.** A non-finite loss (or non-finite parameters, checked every N steps) aborts the run
   **RED** and writes a `FAILURE-*.json`; the metrics writer refuses to persist a non-finite number,
   so a green NaN artifact can no longer exist. **Mutation control** (`test_smoke_train_v2_guard.py`):
   poisoning the encoder makes the run exit non-zero, write the failure artifact, and leave **no**
   metrics — proven by deliberately creating the exact state session 0009 shipped.
2. **Set-local baseline.** The most-common-move baseline is measured on the **eval set itself**, so
   baseline and model share one ruler. The old cross-ruler number (training set's common move scored
   on val: 0.0925 vs 0.0132) is retained for transparency only. `test_baseline_set_locality.py`
   proves a constant policy cannot beat its own set's baseline and a foreign set's move scores no
   better.
3. **v1 reference.** v1 is retrained in the same run and scored on **both** eval sets, so the v1→v2
   delta isolates the organ from the eval-set shift.

`lr` is fixed **by prior**: `0.01`, the top of the stable range (reproduced in `scripts/repro_nan.py`),
chosen before the shot. Thresholds unchanged: SUCCESS ≥ 0.30 / DEAD ≤ set-local baseline /
INCONCLUSIVE between, with **INVALID** added for any non-finite loss/metric or legality < 100%.

## Evidence — local gate

- `make lint` clean (ruff + black), `mypy` clean (49 source files).
- **152 tests passed** (`python -m pytest -q`).

## Evidence — the ADR-0009 publish→verify ritual caught a real defect

The first multi-file commit carried `tests/unit/test_smoke_train_v2.py` as blob **`a2fc6e52`**
(8261 B) while the local source hashes to **`c12040b6`** (8260 B) — the ten other files published
byte-identical. Diagnosis (not a blind retry): the published blob was riddled with U+FFFD replacement
characters and one byte longer — a corruption of the ~11 KB base64 payload during transcription. The
local file is pure ASCII (0 bytes ≥ 0x80, 0 backslashes), so the fix was to re-publish it as **plain
UTF-8** instead of long base64 — its bytes are a function of the source alone. Re-verified: blob sha
`c12040b6`. Commit `46323362`. This is the same "large payload arrives damaged" effect this
workflow's own header records, and the reason the CI driver lives in `scripts/`, not the YAML.

## Evidence — the run

Two dispatches, both recorded (the registry is append-only).

- **CI on the branch:** run [38212667374](https://github.com/feromirdanton-ui/caissa/actions/runs/38212667374)
  (`ci.yml` #31) → **success** on `46323362`.

### Shot 1 — did not start (a defect, not a result)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38212904808 · dispatched
  2026-10-12T06:31:03Z, ended 06:31:12Z (9 s), conclusion `failure`, `head_sha=0914f99d`.
- The job failed at **step 1, “Set up job”** — before any training step. The runner could not resolve
  the pinned action:

  > `##[error]Unable to resolve action actions/setup-python@0b93645e9fe7318ecaed2b359559ac225c90a2b2`

  **Zero artifacts**, **no training**. A run that never started is not a measurement, so it is not
  fed to the pre-registered rule. The diagnosis came from the run's own log, not from inference.

**Cause.** The `actions/setup-python` pin in `smoke-train-v2.yml` dropped one hex character (`a`)
against the valid v5.3.0 SHA `ci.yml` uses (`0b93645e9fea7318ecaed2b359559ac225c90a2b`). Both are 40
hex chars, so the bad pin is well-formed and YAML-valid — it is simply not a real commit. Confirmed
toward the API: the good SHA → HTTP 200, the bad SHA → HTTP 422; the file's other two pins
(`actions/checkout`, `actions/upload-artifact`) resolve, so this was the only broken line.

**Fix.** Commit `fd16b091` (parent `07761df5`) restores the exact v5.3.0 SHA. One line, nothing else;
the pre-registration, thresholds, and `lr` are untouched. Published byte-exact over the API (blob
`a4ff20af…`, `git hash-object`-verified before and after, ADR-0009).

### Shot 2 — the shot (re-fire)

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38213524325
- **Dispatched:** 2026-10-12T06:38:35Z · `head_sha=fd16b0913e80737a9ffd8f9c930da2e6ff792e15`
  (verified equal to the fix commit — no ref/dispatch race; no VOID). Identical inputs to shot 1.
- **Status:** in flight.

## Verdict

**PENDING.** The pre-registration fixes the rule that the verdict is written here **verbatim, after**
the run — never before, and never retouched. The shot is in flight; its outcome (SUCCESS / DEAD /
INCONCLUSIVE / INVALID) will be appended to `LOG/session-0010-shots.md` and restated here when the
run completes. Nothing about accuracy is claimed in this log.

## Risks / open notes

- A diverged run now fails loudly, but the *stability of `lr=0.01`* for this architecture is still an
  assumption carried from `scripts/repro_nan.py` on a small sample; the shot tests it at scale.
- Training cost: the config declares `max_train_positions: 2000` explicitly (≈0.33 s/step at
  d_model 256, K=8) rather than silently timing out. The DoD deviation is stated, not hidden.
- Weights are a disposable run artifact, never a git object (Article 8); `data/**` stays out of git
  (path-ignored + clone guard).

---

*A negative result is recorded with the same care as a positive one, and is worth more than a
retouched one. The human signature for the milestone is never written by the agent.*
