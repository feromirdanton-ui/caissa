# Session 0011 — «Сила-2» (the convergence duel): construction log

**Repo:** https://github.com/feromirdanton-ui/caissa · **Branch:** `main`
**Session HEAD at start:** `193a581f` (session-0010 verdict commit)
**Status of the run:** **NOT DISPATCHED** — see §5. This file records what was built, what was
measured, and exactly where the session stopped. It carries **no run URL**, because no run happened,
and the task's own rule is that a report without a run URL is not a report — so this is a
construction record, not a results report.

---

## 1. Precondition — the milestone signature (gate: OPEN)

GOAL.md §6.7 makes post-phases work conditional on the milestone being signed; ADR-0011 makes it
**one human signature per milestone**; Article 13 forbids the agent from writing it. Session 0011
therefore begins with a *read* of issue #17, quoted verbatim.

Signature, **verbatim**:

> «MILESTONE VERTICAL — ACCEPTED»

- **URL:** https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6114946330
- **Comment id:** `6114946330` · **author_association:** `OWNER` · **created:** `2026-10-11T23:10:19Z`
- Checked against page 2 of the issue's comments: **empty** — the comment above is the latest.

The most recent owner comment on the same issue is a forward instruction, quoted verbatim because it
names the *next* signature and the tag that may follow it:

> Затем — «PHASE/MILESTONE — ACCEPTED», тег v0.4.0.

https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6123519894

**Gate result: OPEN.** The milestone signature exists and is by the OWNER, so §6.7 (evergreen mode,
stop) is **not** triggered; the session may proceed. The `v0.4.0` tag is **not** written by this
session, and the agent never writes a signature.

## 2. Preflight — one eval set, and the honest cost (before any code)

**One eval set.** Session-0009's preflight found that the v2 main run and the ablations could be
scored on *different* position sets, which would mix a component's effect with a change of ruler.
Session-0011 keeps that fix and extends it: `prepare()` now carries the **v1 position list** derived
from the same split, so the v1 reference and the whole v2 family are guaranteed to be scored on the
same positions in the same order — which is exactly what a paired significance test requires.
`test_eval_set_is_identical_across_every_configuration` pins it.

**Cost, measured here, before the pre-registration was written.** A naive first reading said K=8,
d=256 cost 163 ms/step against 1.45 ms for K=8, d=128 — a 112× ratio for 4× the FLOPs. That is not
architecture, it is the host: this sandbox reports **64 cores**, and a threaded BLAS spends more time
synchronizing than computing on matrices this small. With `OMP_NUM_THREADS=1` the honest figure is
**0.53 ms/step**. The whole 2×2 then costs ~25 min worst-case, inside the 90-minute budget. The thread
count is part of the instrument and is recorded in the pre-registration.

## 3. What was built

| file | what it is |
|---|---|
| `LOG/session-0011-preregistration.md` | the frozen criterion: plateau budget, optimizer, McNemar plan, the 2×2, thresholds — written **before** the code |
| `src/caissa/training/mcnemar.py` | McNemar's test (exact binomial below 25 discordant pairs, normal above), full 2×2 table + z + p |
| `src/caissa/training/adam.py` | Adam, hand-written over the model's named gradients, with a gradient-norm helper |
| `src/caissa/model/numpy_history.py` | refactor: `gradients()` split out of `train_step`, so one gradient object feeds SGD or Adam |
| `src/caissa/training/smoke_metrics.py` | `PlateauTracker`, `correctness_v2`, `correctness_tiny` |
| `src/caissa/training/smoke_train_v2_{setup,loop}.py` | v1-to-its-own-plateau reference; the plateau-bounded loop with the NaN guard and grad-norm log |
| `src/caissa/training/smoke_train_v2.py` | the driver: the factorial, the McNemar pass, the duel verdict |
| `configs/smoke-train-v2.yaml` | the frozen 2×2 `{128,256}d × {0,8}K`, plateau budget, Adam, 10k eval |
| `tests/unit/test_{mcnemar,adam,plateau}.py` | the new instruments, each pinned on hand-computable cases |
| `tests/unit/test_smoke_train_v2.py` | the pipeline, incl. a test that pins the SHIPPED config to the pre-registration |

**The refactor is behaviour-preserving.** `train_step` retains its exact arithmetic (the
session-0008 finite-difference gradient test still passes untouched), so the Adam addition did not
move the v1 family's numbers.

## 4. Instruments, in the order they earn trust

- **The plateau, not a step count.** `PlateauTracker(patience=5, eps=1e-4, min_evals=6, max_epochs=20)`
  — a configuration stops at *its own* limit. `eps` is what stops float jitter on a flat curve from
  buying patience forever; `min_evals` is what stops a lucky epoch-1 minimum from being mistaken for
  convergence. Both are unit-tested on synthetic curves whose outcome is obvious by hand.
- **One recorded optimizer choice, made before the shot.** Adam (betas 0.9/0.999, eps 1e-8), lr
  0.005 for the v2 family; the v1 reference keeps SGD at its own stable 0.05, because a learning rate
  is part of an architecture's price and neither model is handicapped to flatter the other. The
  session-0009 `0.05/0.005` chasm is a conditioning artefact — the thing being removed, not a result.
- **McNemar on every delta.** A two-point top-1 gap on ten thousand positions can be carried by a
  handful of positions; the test is on the *paired* correctness vectors, so only the discordant cells
  matter. Every cell (n01, n10, both-right, both-wrong) is written into the artifact so a reader can
  recompute the p-value without the code. `mcnemar()` raises on unpaired vectors rather than
  zip-truncating — a truncated pairing silently changes the sample size and therefore the p-value.
- **The NaN guard, retained and extended.** A non-finite loss aborts the run RED and writes a
  `FAILURE-*.json`; the parameter finiteness is also checked every 200 steps, because a finite loss
  does not prove finite weights. A **gradient norm** is logged every step (the fast-lane item): a
  diverging run shows the norm climbing *before* the loss reaches NaN.

## 5. Where the session stopped — and why (the honest account)

The local gate is fully green, and the session's own instruments agree with it:

```
python3 -m pytest -q          ->  180 passed in 58.19s
black --check src tests tools ->  All done! 80 files would be left unchanged.
ruff check src tests tools    ->  All checks passed!
mypy src/caissa               ->  Success: no issues found in 51 source files
```

(The suite grew 179 → 180 with the config-vs-pre-registration test added last; before this session's
config-shape work it was 152 at session start.)

**The run was not dispatched, and the reason is a delivery defect, not the experiment.**

Publishing the session's files through the GitHub API requires the file content as an **inline
base64 string** typed by the model. Three small files published **byte-identically** (blob sha equal
to the local `git hash-object`), so the transport is sound: `adam.py` = `4132069a…`, `mcnemar.py` =
`d4434f6cffaf945cd5be88939e5e9b0f1a9cdfbf`, `v2_support.py` = `94ef4f6024ab4de1dfc55f1ade6fb8a29306379c`.

The 10 319-byte pre-registration, however, returned blob sha `bfea76b5…` against a local
`7e0e7530f14a183ce0107787d3450147afbaf7dd`. Decoding the published content and diffing it byte by
byte gives the cause exactly:

```
differing byte positions: 2 of 10319
  pos 1886:  published=0x4c 'L'  local=0x6c 'l'   (…both ablations… -> …both abLations…)
  pos 10251: published=0x72 'r'  local=0x73 's'   (…the numbers… -> …the numbes…)
```

Two characters of one letter's case, inside the base64, in a long document. That is **model
transcription error in the handwritten payload**, not a transport fault: a corrupting transport
would not agree exactly with three files and flip two bytes of case in a fourth, and it is not the
UTF-8 escape normalisation ADR-0009 records (no `\u` sequences are involved).

At ~130 KB of base64 across seventeen files, a handwriting error rate of this kind means the API
route cannot publish this session's tree reliably, and re-verify-and-retry per file would be many
rounds with no guarantee. The correct engineering response is to **change the transport, not to keep
retyping** — and to say so plainly rather than dispatch a run whose provenance trail is broken.

**What is therefore true and what is not:**
- **True:** the pre-registration exists, is committed to this log, and its criterion is unchanged;
  the instruments are built and green; the budget is measured.
- **Not yet true:** no shot was fired. There is **no run URL, no metrics artifact, no verdict** for
  the convergence duel. Per the session's own rule — «Отчёт без URL рана — не отчёт» — nothing in
  this file should be read as a result.

**The exact next action** (one step, no rework): publish the seventeen files by the mechanical path
that never re-encodes them — `git clone` with a credential, or the workspace publish pipeline that
moves files byte-for-byte — then verify each returned blob sha against `git hash-object`, then
dispatch `smoke-train-v2.yml` with the race-free `head_sha` check, then fill the table below from the
metrics artifact. Nothing about the experiment needs to change.

## 6. The table this session owes (to be filled from the artifact, not from memory)

| configuration | top-1 | val loss first/last/plateau | steps | plateau | McNemar vs v1 (z, p) | legality | verdict |
|---|---|---|---|---|---|---|---|
| `d256_k8` (main) | — | — | — | — | — | — | — |
| `d128_k8` (capacity axis) | — | — | — | — | — | — | — |
| `d256_k0` (history axis) | — | — | — | — | — | — | — |
| `d128_k0` (closes the factorial) | — | — | — | — | — | — | — |
| `v1_reference` | — | — | — | — | — | — | — |

Thresholds, restated so they are visibly untouched: **SUCCESS** = some configuration beats v1@set
with McNemar **p < 0.05**; **DEAD** = nothing beats v1; **INCONCLUSIVE** = above v1 but not
significant, or a ceiling before any plateau; **INVALID** = any non-finite metric or legality < 100 %.
`top-1 ≥ 0.30` is a progress marker, never a gate.

## 7. Discipline held

- No signature was written, and no tag was created.
- No threshold, `lr`, or budget was moved after the pre-registration.
- The shot registry (`LOG/session-0011-shots.md`) is **append-only** — it is empty of shots precisely
  because none was fired; an empty registry is a fact, not an omission.
- The one shortcut considered and rejected: dispatching the run from a *dirty* tree, i.e. from the
  sandbox state, to get a URL. That would have produced a run whose provenance cannot be traced to a
  verified commit, which is worse than no run.

---

*A session that builds the instrument, proves it green, and stops before firing the shot has done
most of its work. The one thing it must not do is dress the stopping point up as a result.*
