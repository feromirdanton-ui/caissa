# ADR-0009 — Publication through the API: publish → verify → CI as one ritual

- **Status:** Accepted (session-0006)
- **Date:** 2026-10-11
- **Phase:** 2 — Processing
- **Related:** GOAL.md §3 Articles 11–12 (repository is the source of truth; green CI is the proof), ADR-0004, ADR-0005; the session-0005 defect.

## Context

The build environment has **no git credentials** — no SSH key, no credential helper, no
`GITHUB_TOKEN` in the environment. `git push` and `git fetch` cannot authenticate; they block on
a username prompt. The repository is nevertheless the single source of truth (ADR-0004), so the
work must reach it. The available path is the GitHub API: create blob → create tree → create
commit → update ref, driven by MCP tools.

That path has a failure mode the git CLI does not have. In session-0005 a commit **declared a fix
that the tree did not carry**: the commit message described the change, but the `create_tree` call
omitted one of the changed files, so the old version stayed in place while the message claimed
otherwise. The git CLI makes this class of error nearly impossible — it stages exactly what is in
the working tree and refuses to commit a path that does not exist. The API does not: a tree is a
list you write by hand, and an omitted entry is a **silent no-op**, not an error.

## Options

1. **Use the API and trust the commit message.** Rejected: this is exactly the session-0005 defect.
   A commit that claims a change it does not contain is a false record — the same class as a
   manifest that certifies a file it never verified (Article 14).
2. **Abandon the API; require git credentials.** Rejected: the environment does not have them, and
   the project must be buildable where it is built. Refusing to record is worse than recording
   carefully.
3. **Use the API, but make every publish end with a verification against an intended-file
   manifest.** The transport is irrelevant — the commits are real commits with real shas — but the
   *ritual* is fixed: **publish → verify → CI**, and a mismatch is a loud failure. **Chosen.**

## Decision

- **The transport is not the decision.** Whether a commit is made by `git push` or by the API, it
  is a real commit on a real branch with a real sha. Nothing in the constitution privileges one
  transport over the other.
- **Every publish is followed by a verification.** After the ref update, the new commit's tree is
  listed recursively (`GITHUB_GET_A_TREE`) and compared against the **manifest of intended files**
  — the exact set of paths the commit was supposed to change. A path that is missing, or that
  carries an unexpected sha, is a **loud failure**: the publish is not accepted until the tree
  matches the manifest.
- **CI is the third step, not an afterthought.** A green run on the runner from a fresh clone is
  the acceptance proof (ADR-0003). The ritual is atomic: publish → verify → CI.
- **The skill records the asymmetry.** `publishing-files-to-github-via-api` now states plainly that
  the git CLI gives this protection for free and the API does not, so the verification step is
  mandatory, not optional.

## Consequences

- **Positive:** the session-0005 defect class is closed. A commit can no longer claim a change the
  tree does not carry without the verification failing loudly.
- **Positive:** the same discipline as Article 14 — a record is created only after the thing it
  records has been verified.
- **Cost:** one extra API call per publish (the recursive tree listing) and a manifest to maintain.
  Cheap relative to a false record.
- **Known limit:** the verification proves the tree matches the *intended* manifest; it cannot
  prove the manifest itself was complete. A file the author forgot to list is still omitted — the
  guard catches a *dropped* file, not a *forgotten* one. Recorded, not hidden.
- **Follow-up:** when the environment gains git credentials, the API path stays as the fallback and
  the verification stays as the ritual — the guard is about the record, not the transport.
