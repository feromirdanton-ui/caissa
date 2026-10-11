# ADR-0004 — The repository is the single source of truth

- **Status:** Accepted (session-0003)
- **Date:** 2026-10-11
- **Supersedes:** nothing. **Amends:** GOAL.md §3 (adds Articles 11–14).
- **Trigger:** the session-0002 acceptance audit. Phase 0 was delivered as a zip with no `.git`.
  `make lint test` was green in the workspace and **red from a fresh clone** (P0-1), and no CI run
  had ever executed (P0-3). "Green" was true locally and false for a clone.

## Context

The project's own Definition of Awesome is `git clone && make train CONFIG=configs/nano.yaml`.
A delivery that cannot be cloned cannot satisfy it. The build environment had no repository, so
the recording artifact (a zip) was the only trace — and a zip is not a history: it has no commits,
no branches, no tags, no CI, and no way to prove that what was tested is what was shipped.

The audit's verdict was not an accusation but a systemic diagnosis: **an environment without git
could not fulfil the project's own definition of readiness.** The recording environment had to
change before any further code was written.

## Options

1. **Keep delivering zips.** Cheap, but every phase would repeat the same failure: no clone, no CI,
   no history, no acceptance proof. Rejected — it cannot satisfy §2.
2. **Local git only, push later.** Better, but a local repository is still a draft: it proves
   nothing to a reviewer and runs no CI. Rejected as a *recording* environment.
3. **GitHub repository as the single source of truth.** Commits, branches, tags, releases and
   history live in the repository; CI runs on the runner from a fresh clone; a zip becomes only a
   release attachment to a tag. The workspace is a draft. **Chosen.**

## Decision

Adopt four articles into the constitution (GOAL.md §3, Articles 11–14):

- **A — The repository is the single source of truth for code.** Commits, branches, tags,
  releases, history. A zip is only a release attachment to a tag. The workspace is a draft.
  *Actions without a repository is a guard without a fortress.*
- **B — Local run = dev-proof. Green CI on the runner from a fresh clone = acceptance proof.**
  A phase closes only on the second (see ADR-0003).
- **C — Phase closure: green run + tag + LOG + human signature.** The agent **requests**
  acceptance — it does not declare it.
- **D — The manifest is a certification of truth, not a journal.** A record is created only after
  verifying the sha256 of the written file; a `200`-instead-of-`206` and a `416` are loud failures,
  never silent records.

Operationally: the repository is `github.com/feromirdanton-ui/caissa` (private), branch `main`,
Actions enabled, no secrets in files (tokens live in the tooling layer).

## Consequences

- **Positive:** every phase now has a clone, a history and a CI run to point at. Acceptance is
  evidence-based. The zip stops being the artifact and becomes an attachment.
- **Positive:** the clone guard (ci.yml) makes the P0-1 class of failure impossible to reintroduce
  silently — a tracked-but-ignored file fails the build.
- **Negative / cost:** the agent must be able to write to the repository. In this environment the
  GitHub MCP connection is read-mostly for content, so a **write deploy key** is used for `git push`
  (see LOG/session-0003.md). The key is a tooling-layer credential, never a file in the repo.
- **Negative / cost:** a red CI run now blocks a phase. That is the point (Article 7).
- **Follow-up:** ADR-0003 (acceptance protocol) already fixes the "green run from a fresh clone"
  criterion; this ADR makes the repository the place that run happens.

## Alternatives considered and rejected

- **GitLab / self-hosted.** No connection available in this environment; GitHub is the connected
  provider. Revisit only if the connection changes.
- **Committing the zip as a release asset only, no repo.** Rejected: a release needs a repository.
