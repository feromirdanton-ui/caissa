# LOG — session-0003 «Репозиторий как источник истины»

- **Date:** 2026-10-11
- **Phase:** 0 — Foundation
- **Directive:** Session 0003 — the repository becomes the single source of truth.
- **Status:** Phase 0 acceptance **requested** (not declared). Awaiting human signature.

---

## What

Session 0003 did four things, in the order the directive fixed:

1. **Adopted v2** (the auditor's fixes) after review — not blind trust.
2. **Created the repository** `github.com/feromirdanton-ui/caissa` (private, `main`, Actions on)
   and pushed the adopted state as the initial commit.
3. **Ran both workflows for the first time in project history** — `ci.yml` and `nightly.yml` —
   and brought them to green.
4. **Hardened the recording environment**: clone guard in CI, path-based `.gitignore`,
   environment parity (`scripts/run.py`), SHA-pinned actions, manifest anchor, ADR-0004/0005,
   GOAL.md amended to fourteen articles.

## Why

The session-0002 acceptance audit returned **v1 NOT accepted, three P0s**. The verdict was not an
accusation but a systemic diagnosis: the environment had no git, so the project's own Definition of
Awesome (`git clone && make train …`) could not be executed, and "green" was true locally and false
for a clone. From this session the recording environment changes: the repository is the source of
truth (ADR-0004), and a phase closes only on a green run from a fresh clone (ADR-0003).

---

## Step 1 — Adoption of v2 (reviewed, adopted)

**Verdict: reviewed, adopted.** The auditor's fixes are correct. I re-derived each one rather than
trusting the summary:

- **P0-1** (gitignored fixture): verified by simulating a real clone — `git init && git add -A`,
  then copying only `git ls-files` output into a clean directory. Result: **68 tracked files,
  0 ignored-but-tracked, 35 tests pass, lint clean.** The fix works.
- **P0-2** (silent download corruption): read `lichess_bulk.py` and `test_lichess_bulk.py`. The
  range response is now validated (`206` required, `416` handled by discarding the `.part` and
  retrying), and the manifest is written only after the sha256 of the written file is verified.
  The test drives a real local HTTP server that returns `200` instead of `206` and asserts the
  download fails loudly. Correct.
- **P1-7** (Requiem-1 FEN): the test now parses the FEN from `GOAL.md` §8 via a regex, so the
  constitution is the single source of truth for the position. Verified against the actual §8
  block. Correct.
- **P1-2** (sha256 re-verification), **P2-1** (manifest anchor), **P2-4** (SHA pins): reviewed.

**One defect found in the auditor's fix — and fixed here.** The P0-1 fix was *narrow*: `.gitignore`
still ignored by **extension** (`*.pgn`, `*.zst`, …) with a single-file negation for the fixture.
That works today but is fragile — any future source file with a data-like extension is silently
dropped from a clone, which is the exact failure class P0-1 belonged to. Per the directive, the
ignore rules are now **path-based** (`data/**` only; weights by extension wherever they live).
This is logged as a session-0003 change, not a silent edit.

**v2 artifact:** `caissa-phase0-foundation-v2.zip`,
sha256 `fe3b0bc1ade4d17e92e161bdead282df109f27af046c250dfdb5af95e93b8058`.
**v1 originals:** preserved at `audit/backup-0002/` (60 files) for the record. The old workspace was
**not** merged.

## Step 2 — P0-0: the repository

- **Repository:** `github.com/feromirdanton-ui/caissa` — **private**, default branch `main`,
  Actions enabled, issues + projects on, squash-merge only, delete-branch-on-merge on.
- **Initial commit:** `defec2536599577025a820c76778e1aca9bf4328` — 68 files, byte-identical to the
  audited v2 artifact. Verified: the local tree SHA equals the remote tree SHA
  (`6508511ecb46c76c1dde0de889d2e3e80ab023aa`).
- **No secrets in files.** The GitHub MCP connection is read-mostly for content, so a **write
  deploy key** (`caissa-session-0003-bootstrap`) is used for `git push`. The key is a tooling-layer
  credential; it is not, and must never be, a file in the repository.

**Trail (for the record, not an inquisition).** A private repository `teamily-chessgpt` exists on
the same account, created 2026-10-11T02:32Z — *before* session-0001. It has no workflow runs and no
relation to `caissa`; it is recorded here only so the history is complete. No other traces of
previous runs were found.

## Step 3 — First real CI runs

This is the first time in project history the Definition of Awesome is executed in an environment
where it is executable.

| Workflow | Event | Conclusion | URL |
|---|---|---|---|
| `ci.yml` | push (`defec25`) | **success** | https://github.com/feromirdanton-ui/caissa/actions/runs/38110618393 |
| `nightly.yml` | workflow_dispatch | **success** | https://github.com/feromirdanton-ui/caissa/actions/runs/38110740709 |

`ci.yml` runs: clone guard → lint (ruff + black + mypy) → test (py3.10/3.11/3.12) → smoke.
`nightly.yml` runs: property tests → golden suite (incl. Requiem-1) → arena stub → upload reports.

## Step 4 — Clone guard in CI

`ci.yml` now has a `clone-guard` job that runs **first** and fails if:

- `git ls-files -i -c --exclude-standard` is non-empty — a tracked file under `.gitignore`
  (the P0-1 class of failure, now impossible to reintroduce silently); or
- the checkout is dirty after an editable install — generated files are tracked.

The `.gitignore` was generalized from extension-based to **path-based** (`data/**`), as above.

## Step 5 — Environment parity

- **`scripts/run.py`** implements every target (`lint test smoke data-sample eval train ci clean …`)
  in pure Python — no POSIX shell — so the local loop (Windows, no `make`, no `git`) and the CI
  runner converge on one source of commands. The `Makefile` is now a thin delegating wrapper:
  `make X` and `python scripts/run.py X` are the same thing.
- **ADR-0005** fixes the version policy: CI pins **Python 3.12** for lint/smoke/integration; the
  test matrix covers **3.10/3.11/3.12**; numpy-only paths stay compatible with **3.14**; actions
  are pinned by **commit SHA**.

## Step 6 — Session small items

- **P2-1** — manifest anchor is now the repository root (`repo_root()` walks up to `pyproject.toml`),
  not the CWD. A test asserts it.
- **Defect found in my own P2-1 fix, and fixed here.** Making the anchor absolute meant the loader
  test's `monkeypatch.chdir(tmp_path)` no longer redirected it, so the test wrote a *fake* manifest
  (`source: http://127.0.0.1:…`) into the real `data/manifests/` — and the new path-based
  `.gitignore` un-ignored it, so it was committed. That is exactly the Article 14 violation the
  article exists to prevent. Fixed systemically: `ManifestStore.root` now resolves at
  instantiation, and an autouse `conftest.py` fixture redirects `MANIFEST_DIR` to a per-test temp
  directory. A guard test asserts the isolation is active. The fake manifest was removed from the
  tree and from git.
- **P2-4** — actions pinned by SHA: `checkout@11bd7190…` (v4.2.2), `setup-python@0b93645e…` (v5.3.0),
  `upload-artifact@b4b15b8c…` (v4.4.3).
- **ISSUES-0002 registry** adopted into `docs/BACKLOG.md` as the authoritative defect table.

## Step 7 — Acceptance request

- **ADR-0004** written (Articles A–D) and **GOAL.md amended** to fourteen articles (§3, Articles 11–14).
- **Tag `v0.1.0`** on the accepted commit (see the acceptance package).
- **Acceptance package** sent to the human signer via the `ask` tool: both green run URLs, the tag,
  this LOG, and a brief summary. **Phase 1 does not start until the signature.**

---

## Artifacts

- Repository: `github.com/feromirdanton-ui/caissa` (private)
- Initial commit: `defec2536599577025a820c76778e1aca9bf4328`
- CI run (green): https://github.com/feromirdanton-ui/caissa/actions/runs/38110618393
- Nightly run (green): https://github.com/feromirdanton-ui/caissa/actions/runs/38110740709
- ADR-0004, ADR-0005; GOAL.md (14 articles); `scripts/run.py`; clone guard; path-based `.gitignore`
- v2 artifact sha256: `fe3b0bc1ade4d17e92e161bdead282df109f27af046c250dfdb5af95e93b8058`

## Risks

- **The deploy key is a write credential.** It lives in the tooling layer, never in the repo. If it
  is ever rotated, the next session must re-add it before pushing.
- **`scripts/run.py` and the Makefile can drift.** The Makefile is a thin wrapper, so the surface is
  small; a test asserts the target set matches.
- **The torch path is not exercised in CI yet.** The 3.12 pin is therefore not yet load-bearing;
  a `torch`-marked test is a Phase 1+ item.
- **Phase 0 is not accepted.** Until the human signature, Phase 1 does not start; work continues on
  the evergreen backlog (§6.7).

## Next stones

Issue #4 (lichess_bulk: completed-shard skip), #5 (streaming PGN parser), #7 (legality-mask
property tests), #9 (Queens Guard golden suite beyond Requiem-1).
