# session-0002 — acceptance audit of Phase 0 + fixes

- **Date:** 2026-10-11
- **Phase:** 0 — Foundation (acceptance), Phase 1 (gated)
- **Agent:** Caïssa-1 acceptance agent (session 0002)
- **Branch:** none available — the delivery contains no `.git` (see P0-3)

**Verdict: Phase 0 is NOT accepted as delivered (v1).** The artifact reproduced green *as a zip* but
the `git clone` promised by the Definition of Awesome (GOAL.md §2) was red, and the resumable
downloader could silently corrupt a multi-GB shard.

The **v2 artifact** built at the end of this session does reproduce clean from an empty room, with a
clone simulation that drops no file at all (see §v2 below). So the phase was fixable in one session —
but it is not *formally* accepted until someone who did not write the fix reproduces it (ADR-0003,
open question 3). The artifact hash is recorded in the delivery note beside the archive, not inside it.

## Artifact under audit

| item | value |
|---|---|
| file | `caissa-phase0-foundation.zip` |
| size | 64 969 bytes |
| sha256 | `a35639848d09a5383247d58ba5dc8c0b1ccdaf961197a63d41177f7b49af7ee7` |
| files | 60 |
| relation to working copy | byte-identical, 60/60 SHA256 match |

**Clean room:** artifact unpacked into an empty directory; Windows 10.0.19045, Python 3.14.6.
The environment has **no `make`, no `git`, no `uv`, no CI runner**, so the Makefile steps were run as
their underlying commands (this is stated rather than glossed: checklist item A in its literal form
`make lint && make test && make smoke` cannot execute on this machine).

## Raw evidence — clean extraction

| # | command | observed |
|---|---|---|
| A1 | `python -m pytest -q` | `28 passed, 1 warning in 2.82s` |
| A2 | `python -m ruff check src tests tools` | `All checks passed!` |
| A3 | `python -m black --check src tests tools` | clean, exit 0 |
| A4 | `python -m mypy src/caissa` | `Success: no issues found in 32 source files` |
| A5 | `python -m caissa.training.smoke --config configs/tiny.yaml` | `RESULT: PASS`, `loss_first=3.3330 loss_last=2.7146 decreasing=True`, `legal moves: 1000/1000`, `elapsed=0.3s` |
| A6 | `python -m caissa.acquisition.manifest --status` | `no manifests yet — run make data-sample (Phase 1)` |
| A7 | `python -m caissa.acquisition.sample --config configs/data-sample.yaml` | prints the config, then `BUILD PHASE: loaders are wired but not executed here.` |
| A8 | `python -m caissa.training.act1 --config configs/nano.yaml` | **exit 2** — `torch not installed — this is the build phase, not a training run.` (honest refusal, Article 5) |
| A9 | `python -m caissa.eval.arena --config configs/eval.yaml` | `# Arena report (MOCK — not a result)`, `W/D/L: 0/19/1` |

## Raw evidence — simulated `git clone` (the test the DoD actually promises)

Git is not installed, so clone semantics were simulated: every `.gitignore`-matched file and every
empty directory removed, then the suite run from the clone with its own `src` on the path.

| # | command | observed |
|---|---|---|
| C1 | `pytest -q` (clone) | **`3 failed, 25 passed`** — `FileNotFoundError: ...\tests\fixtures\mini.pgn` |
| C2 | `ruff check src tests tools` (clone) | **exit 1** — `E902 … --> tools:1:1` |
| C3 | `black --check src tests tools` (clone) | **exit 2** — `Error: Invalid value for 'SRC ...': Path 'tools' does not exist.` |

Exactly one file was dropped by the simulation: `tests/fixtures/mini.pgn` (matched by `.gitignore:7`
`*.pgn`). `LOG/session-0001.md:29` claims "Tests: … with a PGN fixture" — the author believed it
shipped.

## Independent checks (audit-owned code, not the repo's own tests)

| # | check | observed |
|---|---|---|
| B1 | Requiem-1, FEN parsed out of `GOAL.md` §8 | PASS — `Ne5` → ALARM with `c6e5` (SEE +3), `d5d1` (+2) and the real **`g4d1` (+6, Bxd1)**; `Be2`/`Qe2` → PASS; negative control 15/38 legal moves alarm |
| B2 | mask property over random games | PASS — 48 214 positions, `mask == legal_moves` both directions, **0 counterexamples**; coverage: 1 023 castlings blocked by an attacked transit square, 235 legal en passants, **5 pseudo-legal-but-illegal en passants** (discovered check), 1 468 capture-promotions |
| B3 | mask applied before softmax | confirmed in `representation/mask.py`, `model/numpy_tiny.py`, `model/tiny.py` (`np.where(mask, logits, -inf)` / `masked_fill` then `exp`) |
| B4 | smoke is real training (instrumented) | `train_step(lr>0) = 100/100`; `sample_move` called **998** times for `sample_moves: 1000`; 0 illegal samples; loss 3.3330 → 2.7146; **2 finished positions were counted as "legal" without sampling** (P1-6) |
| B5 | resume under `kill -9` (local HTTP fixture) | resume PROVEN (Range `bytes=1048576-`, `206`, byte-exact, atomic rename); but a Range-ignoring server produced 5 242 880 bytes instead of 4 194 304, hash mismatch, **and a manifest certifying it** (P0-2) |
| B6 | CI alive | files exist, **no CI run has ever happened** (no `.git`); clone test job provably red; `-m "not network"` is vacuous (no test carries the marker) |
| B7 | artifact honesty | no data/weights (only a 600-byte test fixture); mocks labelled; `make train` refuses to fabricate; `git ls-files` impossible (no git) |
| B8 | ADRs contain alternatives | PASS — ADR-0001 substantive (3 options with pros/cons, including the chosen one); ADR-0002 substantive but option 1 is a one-line strawman (P3-1) |

## What could not be verified (and why)

- **A real `git clone` and `git ls-files`** — no git on the machine; substituted by a documented
  `.gitignore`-semantics simulation.
- **GitHub Actions behaviour** — no repository, no runner, no network push; substituted by static
  analysis of the workflow files plus the clone simulation.
- **Real Lichess/TWIC servers** — no download was attempted (GB scale, and the build phase must not
  download); substituted by a local HTTP fixture, which proves the protocol but not Lichess's Range
  behaviour.
- **`make` itself** — absent on Windows; the POSIX Makefile was validated by running its commands.

## Findings

Full register: [`docs/ISSUES-0002.md`](../docs/ISSUES-0002.md). P0s:

1. **P0-1** — `tests/fixtures/mini.pgn` is gitignored (and force-add blocked by pre-commit), so a real
   clone fails 3 tests.
2. **P0-2** — a server that ignores `Range` silently corrupts the shard and the manifest certifies it.
3. **P0-3** — the phase was reported complete although no CI run has ever existed.

## Fixes applied and verified

23 changes, originals backed up outside the repo. The protocol used is now ADR-0003.

| fix | verification |
|---|---|
| P0-1 `!tests/fixtures/*.pgn` + pre-commit exemption | clone simulation now drops 0 source files; `tests/fixtures/mini.pgn present: True` |
| P0-2 Range validation, size check, `IncompleteDownload` | `tests/unit/test_lichess_bulk.py` (7 tests): honest resume, **Range-ignoring server**, stale part/416, short body, cut connection, hash re-verification, verified skip |
| P1-1 `tools/README.md` | clone `ruff` → `All checks passed!`; clone `black --check` → exit 0 |
| P1-2 sha256 re-verified before skipping | corrupt-the-file test forces a re-download |
| P1-3 416 discards the partial file | test asserts a 416 is observed and the shard recovers |
| P1-6 finished positions no longer counted as legal | smoke prints real samples only |
| P1-7 Requiem-1 FEN read from `GOAL.md` §8 | fixture fails loudly if the constitution's shape changes |
| P2-2/P2-3 `.gitkeep`s, `fetched_at`, `_remote_size` used | `fetched_at` asserted non-empty in the new test |
| P1-8 doc claims | README/arena/nightly/MANIFEST/BACKLOG corrected |

**Post-fix verification (patched working copy, installed with `pip install -e .`):**

| step | result |
|---|---|
| `pytest -q` | `35 passed, 1 warning in 6.68s` |
| `ruff check src tests tools` | `All checks passed!` |
| `black --check src tests tools` | clean |
| `mypy src/caissa` | `Success: no issues found in 32 source files` |
| `make smoke` equivalent | `RESULT: PASS`, `legal moves: 1000/1000`, `elapsed=0.6s` |
| Requiem-1 (audit verifier) | PASS with the real `g4d1` capture |

**Same suite from the simulated fresh clone (with its own `pip install -e .`):**
`35 passed` — pytest, ruff, black and smoke all exit 0, and the clone still contains
`tests/fixtures/mini.pgn` and `tools/README.md`.

## Why this session took the shape it did

The architect's instruction was that a report is not an acceptance. That turned out to be
load-bearing rather than ceremonial: the two worst defects (P0-1, P0-2) are both invisible from the
author's tree — one needs a *clone*, the other needs a *hostile server*. Both were found only because
the artifact was unpacked elsewhere and the claims were re-derived instead of read.

Two mistakes of my own are recorded here for honesty: my first mask-property harness reported six
"violations" that were bugs in the harness (a queen on f3 attacks d1 as well as f1, and Black's rook
on e1 makes `e1g1` an ordinary move), and the first patch script died on a PowerShell parse error
before touching anything. Neither was a repo defect; both are noted so the next auditor does not
re-derive them.

## v2 re-acceptance

v2 was built after the fixes and audited with the same protocol, in a fresh empty room.

| item | value |
|---|---|
| file | `caissa-phase0-foundation-v2.zip` |
| files | 68 |
| dropped by the clone simulation | **nothing** — zero files, zero directories |

| step | result |
|---|---|
| `pip install -e .` from the zip | exit 0 |
| `pytest -q` | `35 passed, 1 warning in 6.67s` |
| `ruff check src tests tools` | `All checks passed!` |
| `black --check src tests tools` | clean |
| `mypy src/caissa` | `Success: no issues found in 32 source files` |
| smoke | `RESULT: PASS`, `legal moves: 1000/1000`, `elapsed=0.5s` |
| `make train` (no torch) | **exit 2** — the intended honest refusal |
| Requiem-1 verifier | PASS (real `g4d1` capture) |
| mask property verifier | 48 214 positions, 0 counterexamples |
| **same suite from the simulated clone** | `35 passed`, ruff clean, black clean, smoke PASS |

Every step is green except `make train`, which must be: the build phase has no torch and refuses to
pretend. That is the one exit code in this log that is a *pass*.

## Risks / open questions

- **The fixes are not yet in an artifact.** They live in the working copy; a v2 zip was built at the
  end of this session. Nothing is accepted until v2 is reproduced the same way (ADR-0003).
- **The mask's protection is only as good as its callers.** The property test proves the mask is
  correct; it does not prove every future sampling site applies it before the softmax.
- **Phase 1 is gated.** `make data-sample` is a stub: no puzzle or TWIC loader exists, and
  `configs/data-sample.yaml:resume: true` is read by nothing. Building Phase 1 on top of a broken
  downloader would have produced TBs of silently corrupt data — which is exactly what P0-2 was.
- **Self-acceptance is a conflict of interest.** This session both fixed and verified. ADR-0003's open
  questions flag that the next acceptance should be reproduced by someone who did not write the fix.

## Next stones

1. Produce the v2 artifact, then re-run this acceptance against it (ADR-0003) and tag only then.
2. Create a real repository so CI stops being a claim: first green run on `main` (P0-3).
3. Phase 1 for real: puzzle + TWIC loaders, a manifest root that is not CWD-relative (P2-1), and
   `make data-sample` doing what its name says.
4. Raise the property test to the 10 000 positions GOAL.md §5 Phase 3 asks for (P2-7).
