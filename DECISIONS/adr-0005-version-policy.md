# ADR-0005 — Version policy and environment parity

- **Status:** Accepted (session-0003)
- **Date:** 2026-10-11
- **Trigger:** the session-0002 audit found that the local loop and the CI runner could drift:
  the Makefile assumed a POSIX shell, CI pinned `3.11` while `pyproject.toml` allowed `>=3.10`,
  and actions were referenced by mutable tags.

## Context

Two environments must produce the same result:

1. **The local loop** — a developer on Windows, possibly without `make` and without `git`.
2. **The CI runner** — Ubuntu, `make` present, a fresh clone.

If the two run *different* commands, "green locally" and "green in CI" are different claims, and
the acceptance proof (ADR-0003) is meaningless. Separately, the project must run on a modern
Python for the torch path while keeping the numpy-only paths usable on the newest interpreters.

## Options

1. **Makefile only.** Simple, but not cross-platform: Windows without `make` cannot run the loop.
   Rejected.
2. **A shell script only.** Same problem in reverse, and harder to test. Rejected.
3. **A Python runner (`scripts/run.py`) as the single source of commands, with the Makefile as a
   thin delegating wrapper.** Cross-platform, testable, and identical in both environments.
   **Chosen.**

## Decision

- **One source of commands.** `scripts/run.py` implements every target
  (`lint test smoke data-sample eval train ci clean …`). The `Makefile` delegates to it, so
  `make X` and `python scripts/run.py X` are the same thing. The local loop calls the script; the
  runner calls `make`; both converge.
- **Python version policy.**
  - **CI pins Python 3.12** for the lint, smoke and integration jobs — the version the torch path
    is validated against.
  - **The test matrix covers 3.10 / 3.11 / 3.12** — the range `pyproject.toml` declares
    (`requires-python = ">=3.10"`).
  - **numpy-only paths stay compatible with 3.14** — no dependency may require a version that
    blocks the newest interpreter on the CPU-only path. The torch extra is opt-in
    (`pip install -e ".[torch]"`), so a 3.14 box can still run `make lint test smoke`.
- **Actions are pinned by commit SHA**, never by a mutable tag. A tag can be moved; a SHA cannot.
  Pins in use:
  - `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683` (v4.2.2)
  - `actions/setup-python@0b93645e9fea7318ecaed2b359559ac225c90a2b` (v5.3.0)
  - `actions/upload-artifact@b4b15b8c7c6ac21ea08fcf65892d2ee8f75cf882` (v4.4.3)
- **The clone guard runs first** in `ci.yml`: it fails if any tracked file is ignored by
  `.gitignore`, or if an editable install leaves the tree dirty. This is the machine-checkable
  form of Article 11 (the repository is the source of truth).

## Consequences

- **Positive:** "green locally" and "green in CI" are now the same claim, because they run the
  same commands. The acceptance proof is meaningful.
- **Positive:** a moved action tag cannot silently change what CI runs.
- **Negative / cost:** `scripts/run.py` must be kept in sync with the Makefile's target list. The
  Makefile is a thin wrapper, so the surface is small; a test asserts the target set matches.
- **Negative / cost:** pinning by SHA means upgrades are explicit (a new SHA, a new ADR note), not
  automatic. That is intended.
- **Follow-up:** when the torch path is exercised in CI (a `torch`-marked test), the 3.12 pin
  becomes load-bearing and should be revisited in a later ADR.
