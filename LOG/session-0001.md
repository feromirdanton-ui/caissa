# session-0001 — Phase 0 foundation

- **Date:** 2026-10-11
- **Phase:** 0 — Foundation
- **Agent:** Caïssa-1 construction agent
- **Branch:** `feat/issue-1` (repo skeleton + CI + Makefile + pre-commit)

## What (what was built this session)

Phase 0 foundation, per GOAL.md §5 and §11 tasks 1–3:

- **Repo skeleton** exactly as fixed by ADR-0002 (`src/caissa/` with one subpackage per organ,
  plus `configs/ docs/ DECISIONS/ LOG/ tests/ tools/ data/`).
- **Makefile** with the required targets `data / smoke / eval / lint / test`, plus `install`,
  `format`, `train` (the Definition of Awesome entry point), `ci`, `clean`.
- **pyproject.toml** (setuptools, `src/` layout, ruff + black + mypy + pytest config, console
  scripts `caissa-download`, `caissa-shard`, `caissa-eval`).
- **.pre-commit-config.yaml** — including a local hook that blocks committing data/weights
  (Article 8).
- **CI** — `.github/workflows/ci.yml` (lint + test matrix + smoke) and `nightly.yml`
  (property + golden + arena stub).
- **docs/MANIFEST.md** (concept manifest) and **docs/TRAINING.md** (runbook).
- **DECISIONS/ADR-0001** (representation) and **ADR-0002** (repo layout).
- **Real code:** representation tokens + legality mask; Queens Guard v0 (checks, captures via
  SEE, pins, forks); counterfactual harness with a labeled MOCK engine; tiny model (torch path +
  numpy CPU fallback); `make smoke` runner; acquisition (resumable Lichess loader + manifests);
  processing (streaming PGN, filters, sharding, QA); memory schema; arena stub; lichess-bot
  skeleton.
- **Tests:** unit + golden (Requiem-1) + property, with a PGN fixture.
- **docs/BACKLOG.md** — the first ten kickoff issues (§11) written out.

## Why (decisions made)

- **ADR-0001:** chose 64 square-tokens + meta-tokens with a `(from, to, promo)` move triple, so
  the legality mask is a trivial `64×64×promo` tensor applied inside the softmax (Article 1).
- **ADR-0002:** chose a single package with one subpackage per organ; `guards/` is a sibling of
  `model/`, never inside it (Article 2).
- **Honesty:** the arena opponent and the counterfactual engine are MOCKs, labeled as such in
  code and output. No training results are produced or claimed (Article 5).
- **No data in git:** `.gitignore` + a pre-commit hook enforce manifests-only (Article 8).

## Artifacts

- Repo tree under `caissa/` (see ADR-0002 for the canonical layout).
- `docs/BACKLOG.md` — 10 kickoff issues.
- This journal.

## Risks / open questions

- The torch model path is written but not exercised in CI (torch is an optional extra); the numpy
  fallback is what `make smoke` runs. A later session should add a torch-marked test.
- History depth K and the exact meta-token layout are deferred to a later ADR (ADR-0001 open
  questions).
- DDP/FSDP are stubs, explicitly not verified on a cluster.

## Next stones

1. Issue #4 — Lichess bulk loader end-to-end against a local HTTP fixture (resume + sha256).
2. Issue #5 — streaming PGN parser + filters on real fixtures.
3. Issue #7 — representation + legality mask property tests over 10k random positions.
4. Issue #9 — Queens Guard v0 golden suite, with Requiem-1 as the first and eternal case.
