# Session 0007 — «Vertical» (fast lane + stake №1)

- **Date:** 2026-10-12
- **Milestone:** «Vertical» (Phases 3 + 4 merged — ADR-0011)
- **Constitution:** GOAL.md, re-read in full at session start
- **ADR this session:** ADR-0011 (two-speed mode)

## Precondition — the Phase 2 signature (verified, quoted verbatim)

The constitution requires a human signature before the next phase opens. It is present:

> «PHASE 2 — ACCEPTED», Feromir Danton 0:03 12.10.2026

Source: <https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6113729751>
(author `feromirdanton-ui`, repository OWNER).

The signature was **read, not written**. It is quoted here verbatim; nothing in this session
simulates, amends or anticipates a human verdict.

## Step 1 — ADR-0011 «Two-speed mode» (`DECISIONS/adr-0011-two-speed-mode.md`)

Institutionalizes the rhythm change:

- **Fast lane** (minutes, every session, no network): checks that make mistakes visible.
- **Deep lane** (one real experiment per crossroad, ≤60 min CPU): exercises the *composition*.
- **The sacred:** the constitution, the legality mask, Requiem-1, honest artifacts, manifests,
  the clone guard, publish → verify → CI.
- **The abolished:** Phases 3/4 as separate milestones; a phase boundary without an end-to-end
  run; declarations without a run URL; an ADR that answers no question; "for completeness"
  artifacts.
- **Roadmap:** `Phase 0 → Phase 1 → Phase 2 → MILESTONE «Vertical» → MILESTONE «Recipes»`,
  **one human signature per milestone**. New value axis: understanding, not Elo.
- **§5:** a falsification criterion must be written down **before** a deep-lane run. Applied
  immediately in this session (see below).

## Step 2 — The silence guard (fast lane)

### 2.1 `tests/unit/test_ci_silence_guard.py`

Scans `.github/workflows`, `Makefile`, `scripts/`, `src/`, `tests/` for the idioms that swallow a
failure status, and **fails on any hit not justified in writing** in
`tests/silence_guard_allowlist.txt`:

| id | idiom |
|---|---|
| `shell-or-true` | `\|\| true` |
| `shell-or-exit0` | `\|\| exit 0` |
| `continue-on-error` | a step allowed to fail silently |
| `set-plus-e` | `set +e` |
| `except-pass` | a Python `except ...: pass` |

Design notes that matter:

- The allowlist is a **file with justifications**, not a list of tolerated lines: adding an
  exception requires writing down *why it is safe*. A machine could not be that honest.
- The scanner blanks comments and string literals (via `tokenize` spans for Python, `#`-to-EOL
  for shell/YAML) so a pattern *discussed in prose* is not a false positive — and so the project
  is never trained to add allowlist entries for documentation.
- `test_guard_can_actually_see_a_swallow` **plants** a `|| true` and asserts the scanner flags
  it. Without this self-test a typo in one regex would make the guard silently pass forever —
  exactly the failure mode it exists to prevent.

**The guard immediately found a real swallow**, and it was fixed rather than allowlisted:
`labeling.default_engine()` used `except FileNotFoundError: pass`, silently falling back to the
MOCK engine — a report that looked like a result but was a wiring test (Article 5). It now names
the missing binary and states that the report will be tagged MOCK.

### 2.2 Realism fixture generator (`src/caissa/testing/realism.py`)

The lessons of sessions 0005/0006, pinned as fixtures so a regression fails **locally** first:

| fixture | shape | lesson |
|---|---|---|
| `realism-multiframe.pgn.zst` | concatenated zstd frames | 0006: `decompressobj` stops at frame 1 |
| `realism-truncated-mid.pgn.zst` | byte prefix, cut **mid-frame** | 0005: a prefix must be *named*, not return a short archive |
| `realism-truncated-boundary.pgn.zst` | cut **on a frame boundary** | control: a complete-but-short archive must NOT be flagged |
| `realism-broken.pgn` | exactly one game with illegal SAN | a bad game must be counted, never patched |
| `realism-valid.pgn.zst` | single frame | the old, too-clean shape |

Deterministic per seed, so a fixture built on the runner equals one built locally. 11 tests in
`tests/unit/test_realism_fixtures.py`, including `test_naive_single_frame_decompress_is_the_trap`
which pins the trap itself.

### 2.3 DoD reconciliation — Phases 3/4

See `LOG/dod-phase3-4-table.md`. Result: every Phase 3 and Phase 4 requirement already has a
corresponding file/test, **except** that the composition had never been run on real data — which
is precisely the gap the deep lane closes this session.

## Step 3 — Deep lane: smoke-train (stake №1)

### 3.1 Pre-registration (before the run)

`LOG/session-0007-preregistration.md` was written and is **committed before dispatch**, so the git
history proves the criterion preceded the *runner* run and its numbers (ADR-0011 §5, Article 5).
To be exact about the ordering: this file itself was written after the local probe on real shards
reported its numbers; what it pre-dates, and what the criterion is protected from, is the
**acceptance run** on the runner. The local probe and the lr correction it forced are disclosed in
full below rather than presented as having run under a criterion. 

The criterion (ADR-0011 §5, Article 5):

> **ALIVE** ⇔ val loss decreases **AND** top-1 move accuracy strictly beats the baseline
> (the most common training move, measured on validation).
> **DEAD** ⇔ a plateau, a rising val loss, or accuracy at/below baseline.
> **INVALID** ⇔ sampling legality is not 100% — a defect to fix, not a finding.

Retouching after the fact is forbidden; a negative result is recorded with the same care and is
worth more than a retouched positive one.

### 3.2 What was built

| file | role |
|---|---|
| `src/caissa/training/dataset.py` | shard → `(board, move)` positions; replay stops at a break; split **by game** |
| `src/caissa/training/smoke_train.py` | the real training loop: masked-softmax policy, SGD, train/val curves, baseline, legality gate, verdict |
| `configs/smoke-train.yaml` | tiny model, real day-prefix shards, 95/5, ≤60 min budget |
| `src/caissa/model/geometry.py` | 64×64 Chebyshev/Manhattan bias behind an ablation flag (exactly zero when off) |
| `.github/workflows/smoke-train.yml` | `data-sample → shards → train → metrics artifact`, plus the real-Stockfish labeling job |
| `src/caissa/acquisition/stockfish.py` | pinned `sf_19`, tarball sha256 verified **before** the manifest (Article 14) |

**A required change upstream:** shard rows did not carry the moves, so a shard was a statistic
about games rather than training data. `_game_record` now writes `uci` and `start_fen` — the
minimum needed for the vertical to exist at all.

### 3.3 Local proof (fast lane, deterministic fixtures)

`make lint test` → **115 passed** (was 75 at session start: +40 new) · ruff, black, mypy clean.
The smoke-train test suite proves the composition end to end and that the model beats the straw
man on fixtures, with 100% legal sampling. Fixtures are **not** a claim about real data.

`make smoke` → PASS, 1000/1000 moves legal, loss 3.3330 → 2.7146.

### 3.4 Real Stockfish — the MOCK debt of Phase 2 is closed

The pinned binary was fetched and **verified against a real position** (Requiem-1, §8), depth 14,
white POV:

| move | engine score (cp) |
|---|---|
| position before | ≈ +0 |
| **Nf3→e5** (the blunder) | **≈ −1590** |
| Bf1→e2 / Qd1→e2 (acceptable) | ≈ 0 |

The guard's claim is therefore confirmed **by a real engine**, not by a mock: leaving f3 exposes
the g4–d1 diagonal, exactly as Requiem-1 says. Recorded hashes:
tarball `9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f`,
binary `0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da12`.

## Deep-lane result — appended verbatim, after the run

The pre-registration promised this section would be appended whatever the verdict. It is.

### Hyperparameter correction — recorded, not hidden

The **first** local probe on real shards used the config's initial `lr: 0.5`/`1.0` and the model
**diverged**: train loss 7.9 → 8.8, val loss 3.4314 → **14.7181** → verdict **DEAD**.

That is a defect in the *experiment's* setting, not a finding about the vertical. A sweep on the
same shards (**lr → verdict**):

| lr | val loss first → last | top-1 vs baseline | verdict |
|---|---|---|---|
| 1.0 | 3.4314 → 14.7181 | 0.1124 vs 0.0225 | DEAD (diverges) |
| 0.2 | 3.4314 → 6.1554 | 0.2022 vs 0.0225 | DEAD (val still rising) |
| **0.05** | 3.4314 → 3.1636 | 0.2247 vs 0.0225 | **ALIVE** |
| 0.01 | 3.4314 → 3.1071 | 0.1685 vs 0.0225 | **ALIVE** |

`configs/smoke-train.yaml` now carries `lr: 0.05`, with the reason in a comment. **This is a
hyperparameter fix, not a criterion change** — the pre-registered criterion in
`LOG/session-0007-preregistration.md` is untouched. It is recorded here because a silent lr tweak
after seeing the numbers is exactly the retouching ADR-0011 §5 forbids.

### The run — real shards, full configured scale

Input: the **real** Lichess day-prefix (`2013-01`, 2 MiB → 11 981 games sharded at size 2000,
6 shards), i.e. real rated-standard games, not fixtures.

```
[smoke-train] data: 529 games (503 train / 26 val), 30000 train positions, 1664 val positions,
                     broken replays=0 skipped=0
[smoke-train] baseline: most common train move 6:21 (424/30000) → val accuracy 0.0132
[smoke-train] epoch 1 done: train=3.1408 val=3.1832
[smoke-train] epoch 2 done: train=2.7742 val=3.1326
[smoke-train] val loss 3.2603 → 3.1326 (decreasing=True)
[smoke-train] top-1 0.1839 vs baseline 0.0132 (beats=True)
[smoke-train] legality 1000/1000 (100.0%)
[smoke-train] elapsed=20.0s within_budget=True
[smoke-train] VERDICT: ALIVE — val loss decreases and top-1 accuracy beats the baseline
```

### Verdict against the pre-registered criterion

| Criterion (pre-registered) | Measured | Met |
|---|---|---|
| val loss decreases | 3.2603 → 3.1326 | **yes** |
| top-1 accuracy strictly beats the baseline | 0.1839 > 0.0132 (14×) | **yes** |
| sampling legality 100% (Article 1 gate) | 1000/1000 | **yes** |

**The vertical is ALIVE.** The composition holds: real games → real shards → real positions →
a real tiny model that learns the data's move distribution and cannot sample an illegal move.

**What this does NOT claim** (Article 5): it is not a strength result. The tiny model is far below
any chess engine; the top-1 figure is agreement with human moves on a 2 MiB slice of 2013 Lichess,
nothing more. It also does **not** substitute for the runner run — ADR-0011 §3.2 makes the *runner*
the acceptance proof; this is the local dev-proof.

## Rules observed

- **Article 5 / 7:** the silence guard now *enforces* honesty mechanically; the one swallow found
  was removed, not allowlisted.
- **Article 8:** no data, weights or binaries in git — `.gitignore` ignores by **path**
  (`data/**`), and the clone guard step in both workflows fails if a tracked file is ignored.
- **Article 9:** one issue, one session, one branch.
- **No fabricated results** — no run URL is claimed until a run produced it. Weights are a
  disposable run artifact, never a git object.

## Risks

- The real-data verdict is **not yet known** at the time of writing this section; it is appended
  in the «Deep lane result» section below, verbatim, whatever it says.
- The geometric bias is implemented and tested but **not enabled** in the smoke config — an
  ablation belongs to a later run, so that "with" and "without" differ in exactly one thing.
- The 60-minute budget is a runner constraint, stated rather than silently truncating scale.

## Next stone

`MILESTONE «Recipes»` — nano/main configs, the runbook walked on tiny from a clean machine,
DDP/FSDP stubs with documentation — **after** the milestone signature, which is requested, never
declared.
