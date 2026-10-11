# ISSUES 0002 — acceptance audit of Phase 0

> Register from session 0002 (acceptance audit, GOAL.md §6.1 format: goal, context, DoD, artifacts,
> failure criterion). The audit ran against the **delivered artifact**, not the author's tree:
>
> - file: `caissa-phase0-foundation.zip`, 64 969 bytes
> - sha256: `a35639848d09a5383247d58ba5dc8c0b1ccdaf961197a63d41177f7b49af7ee7`
> - 60 files; byte-identical to the Desktop working copy (60/60 SHA256 match)
>
> Clean room: Windows 10.0.19045, Python 3.14.6, no `make`, no `git`, no `uv`, no CI runner.
> Raw command → output evidence: `LOG/session-0002.md`.

## Summary

| id | P | title | status |
|---|---|---|---|
| P0-1 | P0 | Test fixture `tests/fixtures/mini.pgn` is gitignored, so a real clone fails 3 tests | **fixed** |
| P0-2 | P0 | A server that ignores `Range` silently corrupts a shard; the manifest certifies it | **fixed** |
| P0-3 | P0 | Phase 0 was reported complete, but no CI run has ever happened (no `.git` in the delivery) | parked (needs a repository) |
| P1-1 | P1 | `tools/` is empty → `make lint` fails on any clone (`Path 'tools' does not exist`) | **fixed** |
| P1-2 | P1 | sha256 is stored but never verified or re-verified | **fixed** |
| P1-3 | P1 | A stale/oversized `.part` (HTTP 416) deadlocks the shard forever | **fixed** |
| P1-4 | P1 | Resume had zero test coverage, while the module docstring claimed a test existed | **fixed** |
| P1-5 | P1 | `arena.py` implies a real Stockfish path; both sides actually move randomly | partial (docstring fixed, implementation parked) |
| P1-6 | P1 | `smoke.py` counted finished positions as "legal moves", inflating the 100 % claim | **fixed** |
| P1-7 | P1 | Requiem-1 FEN was a duplicated literal instead of being read from GOAL.md §8 | **fixed** |
| P1-8 | P1 | Documentation claims without reproducible evidence | partial |
| P1-9 | P1 | `make data-sample` is a stub; puzzle/TWIC loaders do not exist (Phase 1 DoD unreachable) | parked (Phase 1 work) |
| P2-1 | P2 | Manifest root is CWD-relative → idempotency breaks across working directories | parked (workaround) |
| P2-2 | P2 | Empty directories (`data/`, `tools/`, `tests/integration/`) cannot survive a clone | **fixed** |
| P2-3 | P2 | `fetched_at` empty in Lichess manifests; `_remote_size` was dead code | **fixed** |
| P2-4 | P2 | Unpinned action tags and dev dependencies (drift risk vs pinned pre-commit) | parked |
| P2-5 | P2 | `nightly.yml` uploads `LOG/`, but nothing writes a report there | parked |
| P2-6 | P2 | Makefile is POSIX-only (`SHELL := /bin/bash`), no Docker path for Windows | parked (workaround) |
| P2-7 | P2 | Nightly property step claims 10k positions, exercises ~800; GOAL §5 Phase 3 wants 10k | parked |
| P2-8 | P2 | `-m "not network"` deselection is vacuous: no test carries the marker | parked |
| P3-1 | P3 | ADR-0002 option 1 is a one-line strawman without tradeoffs | open |
| P3-2 | P3 | ADR-0001 points at the wrong "mask" file and promises a promo axis the model lacks | open |

---

## P0-1 — the fixture never reaches the repository

- **Labels:** `P0`, `phase-0`, `phase-2`
- **Goal:** `git clone && make lint test` is green from the delivered artifact.
- **Context:** GOAL.md §2 (Definition of Awesome) is a claim about a *clone*. `.gitignore:7` ignores
  `*.pgn` globally with no negation; `.pre-commit-config.yaml:34` also blocks a force-add. The test
  suite reads `tests/fixtures/mini.pgn` (`tests/unit/test_processing.py:11,15,26,32`).
- **Evidence:** simulated clone (drop ignored files + empty dirs) → `3 failed, 25 passed` with
  `FileNotFoundError: ...\tests\fixtures\mini.pgn`; the *only* dropped file was that fixture.
- **DoD:** clone simulation passes `pytest -q`; the fixture is committable.
- **Artifacts:** `.gitignore`, `.pre-commit-config.yaml`.
- **Failure criterion:** any file required by the suite is absent from a clean clone.
- **Status:** fixed in session 0002 (`!tests/fixtures/*.pgn`, pre-commit exemption for
  `tests/fixtures/`), verified by re-running the clone simulation.

## P0-2 — Range-ignoring server corrupts the artifact silently

- **Labels:** `P0`, `phase-1`
- **Goal:** a partially downloaded shard is never silently corrupted, and a corrupt shard is never
  certified as complete.
- **Context:** Article 4 (everything resumable) and Article 5 (honesty of artifacts). `download()`
  appended to `<target>.part` whenever a partial existed, without checking whether the server
  honoured `Range`.
- **Evidence (local HTTP fixture, no internet):** payload 4 194 304 B. Range-ignoring server →
  final file 5 242 880 B, sha256 mismatch, `.part` renamed into place, and the manifest written with
  `bytes=5242880` — a corrupt multi-GB shard certified as complete. Also: a truncated body was
  renamed into place; a stale `.part` produced HTTP 416 and then deadlocked every later run.
- **DoD:** `tests/unit/test_lichess_bulk.py` covers resume (206 + exact offset), Range-ignoring
  server, stale part/416, truncated body, and hash re-verification.
- **Artifacts:** `src/caissa/acquisition/lichess_bulk.py`, `tests/unit/test_lichess_bulk.py`.
- **Failure criterion:** any path where a file that is not byte-exact is renamed to its final name.
- **Status:** fixed — the range response is validated (`206` + `Content-Range` start must equal the
  resume offset), the partial file is discarded and the download restarts on any mismatch, the final
  size is checked against the advertised length, and an incomplete body raises instead of publishing.

## P0-3 — no CI run has ever happened

- **Labels:** `P0`, `phase-0`, `parked`
- **Goal:** the Phase 0 DoD line "CI triggers on push" is either satisfied or withdrawn.
- **Context:** `.github/workflows/ci.yml` and `nightly.yml` exist and look sane, but the delivery
  contains no `.git`. `git` is not installed in the acceptance environment, so no workflow could
  ever have executed; `git ls-files` (checklist B7) is impossible.
- **Evidence:** `git ls-files` → `git: command not found`; no `.git` directory in the artifact;
  `docs/BACKLOG.md:21` nevertheless claimed a "CI run" artifact.
- **DoD:** a repository exists, a workflow run is linked in `LOG/`, and the run is from the tagged
  commit.
- **Artifacts:** to be created (repository + first green run).
- **Failure criterion:** a phase marked done while its CI has never run.
- **Status:** parked (cannot be fixed without a remote). Workaround: acceptance used the underlying
  commands plus a `.gitignore`-semantics clone simulation; `BACKLOG.md` no longer claims a CI run.

## P1-1 — `tools/` cannot exist in a clone → lint job red

- **Evidence:** clone simulation `black --check src tests tools` →
  `Error: Invalid value for 'SRC ...': Path 'tools' does not exist.`; `ruff` → `E902`. Same paths in
  `Makefile:20,21,25,26` and `ci.yml:29,31`.
- **DoD:** `make lint` passes from a clone.
- **Status:** fixed — `tools/README.md` documents the three packaged console scripts and keeps the
  path alive.

## P1-2 — sha256 was recorded but never verified

- **Evidence:** a completed shard whose bytes were replaced was skipped with
  "already complete — skipping (Article 4)" while the manifest still held the old hash; no code path
  compared them.
- **Status:** fixed — `fetch_month` now re-verifies size + sha256 before skipping, and re-downloads on
  mismatch.

## P1-3 — stale `.part` (416) deadlocked a shard permanently

- **Status:** fixed — a 416 discards the partial file and restarts from zero.

## P1-4 — resume had no tests, and the docstring said it did

- **Evidence:** the module docstring claimed "exercised by unit tests against a local HTTP fixture"
  and "verifies sha256 when the server provides one"; no such test existed and no such code existed.
- **Status:** fixed — `tests/unit/test_lichess_bulk.py` (6 tests) and a corrected docstring that
  describes what the code actually does.

## P1-5 — arena implies a real Stockfish path

- **Evidence:** `arena.py:3-4` said "When a Stockfish binary is present, plays real games"; both sides
  play random legal moves, `opponent` is never consulted, `mock=True` is hardcoded, and no blunder-rate
  metric exists (`BACKLOG.md:132` unchecked).
- **Status:** partial — the docstring now states the stub honestly; implementing the real path is
  parked as Phase 6 work (GOAL.md §5).

## P1-6 — smoke inflated its legality claim

- **Evidence (instrumented run):** `sample_move` called 998 times for `sample_moves: 1000`; two
  finished positions were counted as "legal" without sampling anything, so
  `legal moves: 1000/1000 (100.0%)` overstated coverage. The other claims held:
  `train_step(lr>0) = 100`, all 998 sampled moves legal, loss 3.3330 → 2.7146, elapsed 0.3 s.
- **Status:** fixed — finished positions are skipped, so the printed figure counts real samples.

## P1-7 — Requiem-1 FEN was duplicated instead of read from the constitution

- **Evidence:** `tests/conftest.py` hardcoded `REQUIEM_1_FEN`; the audit had to parse GOAL.md §8 to
  check that the two agree (they do today).
- **Status:** fixed — `conftest.py` extracts the FEN from `GOAL.md` §8 and fails loudly if the
  constitution's shape changes.

## P1-8 — documentation claims without evidence

- **Evidence / disposition:** README "Act I starts" (now states the torch precondition),
  `nightly.yml` "10k random positions" (now unqualified), `MANIFEST.md` manifest format `.yaml` →
  `.json` (code writes JSON), `BACKLOG.md:21` "CI run" (now records that no run exists),
  `docs/TRAINING.md:33` "QA report in `LOG/`" (still aspirational — parked, P2-5),
  ADR-0001 mask location + promo axis (P3-2).
- **Status:** partial — the load-bearing claims are corrected; the remainder are listed here.

## P2-1 — CWD-relative manifest root

- **Evidence:** manifests land in `<cwd>/data/manifests`; the same shard is unknown from another
  working directory → full re-download plus a duplicate manifest.
- **Status:** parked. Workaround: always invoke the loader from the repository root (the new test
  pins the CWD explicitly to make this visible rather than implicit).

## P2-2 — empty directories do not survive a clone

- **Status:** fixed — `data/.gitkeep`, `tools/README.md`, `tests/integration/.gitkeep`.

## P2-3 — `fetched_at` empty; `_remote_size` dead

- **Status:** fixed — the Lichess manifest is written through `write_manifest` (timestamp populated),
  and `_remote_size` is now used to validate the final length.

## P2-4 / P2-5 / P2-6 / P2-7 / P2-8

Unpinned actions and dev deps; `nightly.yml` uploading an empty `LOG/`; POSIX-only Makefile without a
container path; the nightly property step's 10k claim; the vacuous `not network` deselection. All
parked with notes; none of them can make a passing build fail.

## P3-1 / P3-2 — ADR quality

ADR-0001 and ADR-0002 both have substantive alternatives (ADR-0001 even records a concession for the
chosen option). ADR-0002's option 1 is a one-line strawman; ADR-0001 cites
`src/caissa/guards/legality_mask.py` as the mask while the implementation is
`src/caissa/representation/mask.py`, and promises a `(from, to, promo)` softmax while both models are
`(from, to)`. Cosmetic, but they are exactly the kind of drift Article 10 exists to prevent.

---

## Verified clean (kept, not changed)

- **Requiem-1 reproduces, for the right reason.** FEN parsed out of GOAL.md §8; `Ne5` → ALARM with
  three material threats including the real `g4d1` (SEE +6, the queen on d1 after the knight leaves);
  `Be2` and `Qe2` → PASS. Negative control: 15 of 38 legal moves alarm, so the guard is a filter, not
  a wall.
- **Legality mask property holds.** 48 214 positions from 800 random legal games: `mask ==
  board.legal_moves` in both directions, zero counterexamples, plus 1 023 castlings blocked by an
  attacked transit square, 235 legal en passants, 5 pseudo-legal-but-illegal en passants (the
  discovered-check edge case) and 1 468 capture-promotions. Mask is applied before the softmax
  (`np.where(mask, logits, -inf)` then `exp`, in both `numpy_tiny.py` and `tiny.py`).
- **No data or weights in the artifact.** The only data-shaped file is the 600-byte
  `tests/fixtures/mini.pgn` fixture.
- **Mocks are labelled** (`arena.py` "MOCK — not a result", `counterfactual.MockEngine`,
  `training/distributed.py` `STUB_NOTICE`) and `make train` **refuses to fabricate**: exit code 2 with
  "torch not installed — this is the build phase, not a training run."
- **Test suite has no hollow tests.** 26 test functions, every one with an assertion; no empty test
  files.
- **Clean extraction is green:** `pytest -q` → 28 passed; `ruff` → all checks passed;
  `black --check` → clean; `mypy` → no issues in 32 source files; `make smoke` → PASS.
