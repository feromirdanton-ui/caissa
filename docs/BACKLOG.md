# BACKLOG — the first ten kickoff issues (GOAL.md §11)

> GitHub Issues is the task queue (GOAL.md §6.1). This file is the human-readable mirror of the
> first ten issues, written out at Phase 0 kickoff. Each issue carries: goal, context, DoD
> checklist, artifacts, failure criterion. Labels: `P0`–`P3`, `phase-N`, `idea`, `parked`.
> Board: Backlog → Ready → Doing → Done → Parked.

---

## Issue #1 — Repo skeleton + CI + Makefile + pre-commit

- **Labels:** `P0`, `phase-0`
- **Goal:** Stand up the repository exactly as fixed by ADR-0002, with a green CI on push.
- **Context:** GOAL.md §4.6, §5 Phase 0. Everything else depends on this.
- **DoD:**
  - [x] Repo tree per ADR-0002
  - [x] `Makefile` with `data / smoke / eval / lint / test`
  - [x] `pyproject.toml`, `.pre-commit-config.yaml`
  - [x] `.github/workflows/ci.yml` triggers on push
  - [x] `make lint test` green *(session-0002: green from the delivered zip, RED from a real clone — P0-1)*
  - [x] **green CI run from a fresh clone** *(session-0003: ci.yml run 38110618393, SUCCESS)*
- **Artifacts:** repo tree. *(session-0003: repository `feromirdanton-ui/caissa`, initial commit `defec25`)*
- **Failure criterion:** CI red, or a required target missing.

## Issue #2 — docs/MANIFEST.md (concept manifest)

- **Labels:** `P0`, `phase-0`
- **Goal:** State the concept of §1 in our own words — brief and precise.
- **Context:** GOAL.md §11 task 2.
- **DoD:**
  - [x] Four organs described
  - [x] Five acts described
  - [x] Value axis (understanding, not Elo) stated
  - [x] Data-manifest schema documented
- **Artifacts:** `docs/MANIFEST.md`.
- **Failure criterion:** the concept is paraphrased from memory rather than understood.

## Issue #3 — ADR-0001 (representation) + ADR-0002 (repo layout)

- **Labels:** `P0`, `phase-0`
- **Goal:** Fix the two foundational decisions, each with alternatives.
- **Context:** GOAL.md §3 Article 6, §4.1, §4.6.
- **DoD:**
  - [x] ADR-0001: context / options / decision / consequences
  - [x] ADR-0002: context / options / decision / consequences
  - [x] Open questions listed for later ADRs
- **Artifacts:** `DECISIONS/adr-0001-representation.md`, `DECISIONS/adr-0002-repo-layout.md`.
- **Failure criterion:** a decision without alternatives.

## Issue #4 — lichess_bulk loader: resumable + sha256 + manifest

- **Labels:** `P0`, `phase-1`
- **Goal:** A loader that survives interruption and never re-downloads a completed shard.
- **Context:** GOAL.md §7, §3 Article 4.
- **DoD:**
  - [x] `.part` + HTTP `Range` resume implemented *(session-0002: hardened — the range response is now validated, P0-2)*
  - [x] sha256 recorded in the manifest *(session-0002: and re-verified on later runs, P1-2)*
  - [ ] completed shard skipped on re-run
  - [x] unit test against a local HTTP fixture (no network in CI) *(session-0002: `tests/unit/test_lichess_bulk.py`)*
- **Artifacts:** `src/caissa/acquisition/lichess_bulk.py`, manifest entry, test.
- **Failure criterion:** a restart re-downloads completed work.

## Issue #5 — streaming PGN parser + filters + tests on fixtures

- **Labels:** `P0`, `phase-2`
- **Goal:** Parse PGN as a generator; filter rated standard; bucket by Elo.
- **Context:** GOAL.md §5 Phase 2.
- **DoD:**
  - [ ] generator-based parser (no RAM loading)
  - [ ] rated-standard filter + Elo buckets
  - [x] tests on a committed fixture *(session-0002: the fixture is now committable — P0-1)*
- **Artifacts:** `src/caissa/processing/pgn.py`, `filters.py`, tests.
- **Failure criterion:** the parser loads a whole file into memory.

## Issue #6 — sharding + QA report on 100k games

- **Labels:** `P0`, `phase-2`
- **Goal:** Write `jsonl.zst` shards atomically and emit a QA report.
- **Context:** GOAL.md §5 Phase 2, §11 task 6.
- **DoD:**
  - [ ] atomic shard writes (no `.part` left behind)
  - [ ] QA report: Elo / opening / time-control distributions
  - [ ] report written to `LOG/`
- **Artifacts:** `src/caissa/processing/sharding.py`, QA report.
- **Failure criterion:** a crash corrupts a previously written shard.

## Issue #7 — representation + legality mask + property tests

- **Labels:** `P0`, `phase-3`
- **Goal:** The mask is complete and sound over random positions.
- **Context:** GOAL.md §3 Article 1, §5 Phase 3.
- **DoD:**
  - [ ] every legal move allowed by the mask
  - [ ] every illegal move rejected
  - [ ] promotions / en passant / castling covered
  - [ ] 10k positions property-tested without errors
- **Artifacts:** `src/caissa/representation/`, `tests/property/`.
- **Failure criterion:** any legal move missing, or any illegal move allowed.

## Issue #8 — tiny model + `make smoke`

- **Labels:** `P0`, `phase-4`
- **Goal:** CPU smoke: ≤10 min, loss decreasing, 1000 sampled moves 100% legal.
- **Context:** GOAL.md §10.
- **DoD:**
  - [ ] tiny config (<10M)
  - [ ] `make smoke` ≤10 min on CPU
  - [ ] loss decreasing
  - [ ] 1000 sampled moves — 100% legal
- **Artifacts:** `src/caissa/model/`, `src/caissa/training/smoke.py`.
- **Failure criterion:** any sampled move illegal, or loss not decreasing.

## Issue #9 — queens_guard v0 + golden suite + Requiem-1

- **Labels:** `P1`, `phase-6`
- **Goal:** The guard catches 100% of golden-suite traps, including Requiem-1.
- **Context:** GOAL.md §3 Article 2, §8.
- **DoD:**
  - [ ] guard enumerates checks / captures / pins / forks
  - [ ] Requiem-1: Ne5 alarms; Be2, Qe2 pass
  - [ ] golden suite green
- **Artifacts:** `src/caissa/guards/queens_guard.py`, `tests/golden/`.
- **Failure criterion:** Requiem-1 fails, or the test is deleted.

## Issue #10 — arena stub vs Stockfish + report

- **Labels:** `P1`, `phase-6`
- **Goal:** A 20-game arena with a blunder-rate report; MOCK opponent until Stockfish is wired.
- **Context:** GOAL.md §4.5, §11 task 10.
- **DoD:**
  - [ ] 20 games played
  - [ ] report emitted, honestly labeled MOCK when no engine is present
  - [ ] blunder-rate metric defined
- **Artifacts:** `src/caissa/eval/arena.py`, report.
- **Failure criterion:** a mock result presented as a real strength result.

---

## Session-0002 audit registry (adopted, session-0003)

> The external acceptance audit of Phase 0 (`docs/ISSUES-0002.md`) is adopted into the backlog as
> the authoritative defect registry. Status as of session-0003.

| ID | Severity | Finding | Status |
|---|---|---|---|
| P0-1 | P0 | `tests/fixtures/mini.pgn` was gitignored → a fresh clone could not run the suite | **Fixed** (v2) + **generalized** (session-0003: `.gitignore` is now path-based, not extension-based) |
| P0-2 | P0 | the downloader recorded a manifest for a corrupt/partial file (silent `200` instead of `206`) | **Fixed** (v2) — range response validated, `416` handled loudly |
| P0-3 | P0 | no repository, no CI run — "green" was never proven on a runner | **Fixed** (session-0003: repo created, ci.yml + nightly.yml green) |
| P1-2 | P1 | sha256 not re-verified on later runs | **Fixed** (v2) |
| P1-7 | P1 | Requiem-1 FEN duplicated in the test instead of parsed from GOAL.md §8 | **Fixed** (v2) — parsed from the constitution, fails loudly if §8 changes shape |
| P2-1 | P2 | manifest anchor was the CWD, not the repo root | **Fixed** (session-0003) |
| P2-4 | P2 | actions referenced by mutable tags | **Fixed** (session-0003) — pinned by commit SHA (ADR-0005) |

**Open (not yet closed):** Issue #4 "completed shard skipped on re-run", Issue #5 generator parser,
Issue #6 sharding + QA, Issue #7 property tests, Issue #8 tiny model, Issue #9 guard v0, Issue #10
arena stub. These are Phase 1+ work and are **not** part of the Phase 0 acceptance request.
