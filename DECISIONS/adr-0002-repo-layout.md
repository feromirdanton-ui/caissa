# ADR-0002 — Repository layout

- **Status:** Accepted
- **Date:** 2026-10-11
- **Deciders:** Caïssa-1 construction agent
- **Supersedes:** —
- **Related:** GOAL.md §4.6, §3 Articles 2, 8, 9

## Context

The project is an endless build with an external task queue (GitHub Issues) and a strict honesty
regime: data outside git, decisions in ADRs, a session journal, and a hard separation between the
model and the Queens Guard (Article 2). The layout must make those boundaries *physical*, so that
violating them is awkward rather than merely discouraged.

## Options considered

1. **Flat `src/` with feature modules.** Simple, but the organ boundaries blur; the Guard can drift
   into the model; data and code mix.
2. **Monorepo with separate packages per organ** (`caissa-model`, `caissa-guards`, …).
   - *Pro:* hard boundaries; independent versioning.
   - *Con:* heavy for a build phase; slows the session cycle; over-engineered before the organs exist.
3. **Single package `src/caissa/` with one subpackage per organ, plus top-level `docs/`,
   `DECISIONS/`, `LOG/`, `configs/`, `tests/`, `tools/`.** *(chosen)*
   - *Pro:* boundaries are directories; light enough for one-command work; matches GOAL.md §4.6.
   - *Con:* boundaries are conventions, enforced by review and tests rather than by the build system.

## Decision

Adopt **option 3**, exactly as sketched in GOAL.md §4.6:

```
caissa/
  GOAL.md  Makefile  pyproject.toml  .pre-commit-config.yaml
  .github/workflows/     # ci.yml (lint+test+smoke), nightly.yml (integration)
  configs/               # tiny | nano | main + experiments
  docs/                  # MANIFEST.md (concept), TRAINING.md (runbook)
  DECISIONS/             # adr-NNN-*.md
  LOG/                   # session-NNNN.md, WEEK-NNN.md
  src/caissa/
    acquisition/  processing/  representation/  model/
    training/     guards/      memory/          eval/  serving/
  tests/                 # unit | integration | golden | property
  tools/                 # caissa-download, caissa-shard, caissa-eval
  data/                  # git-ignored; only data/manifests/ is tracked
```

Rules that follow from the layout:

- **`guards/` is a sibling of `model/`, never a submodule of it** (Article 2). The Guard imports
  `python-chess` and nothing from `model/`.
- **`data/` is git-ignored except `data/manifests/`** (Article 8). The `.gitignore` and a
  pre-commit hook both enforce this.
- **`DECISIONS/` and `LOG/` are first-class**, not an afterthought (Articles 6, 9).
- **`tests/` is split by kind** so CI can run the fast set on every push and the heavy set nightly.

## Consequences

- The session cycle (GOAL.md §6.2) maps onto the layout: a change touches `src/`, `tests/`, and
  `LOG/session-NNNN.md`.
- Adding a new organ means adding a directory and an ADR — not restructuring the repo.
- The layout is a convention; a future ADR may promote an organ to its own package if it grows.

## Open questions (deferred)

- Whether `serving/` (Act V) should live in this repo or a separate one.
- Whether `tools/` becomes a console-script-only package with no importable code.
