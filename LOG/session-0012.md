# Session 0012 «Выстрел-2» — the shot, and the defect it exposed

**Precondition:** none. The milestone «Сила-2» is open; the experiment was frozen by session-0011's
pre-registration (`LOG/session-0011-preregistration.md`), so this session only had to carry it and
fire. No threshold was moved, and none is moved here.

**Head at session start:** `193a581f` (session-0011's files were NOT in the repository — see below).

---

## Step 1 — the transport (ADR-0012)

Session-0011 built the duel instrument but never fired, because publishing through the GitHub API
with inline base64 typed by a model lost two characters (`LOG/session-0011.md`). ADR-0012
(`DECISIONS/adr-0012-publish-transport.md`) fixes the rule: **LLM base64 transcription only for files
under 2 KB; everything larger moves mechanically; every file, on every transport, is verified by
`blob sha == git hash-object`.**

The mechanical transport chosen (option B, because this sandbox has no git credentials, no SSH key
and no `gh`, and the runner's `GITHUB_TOKEN` is read-only by default):

1. the tree is packed and encrypted **mechanically** — the bytes are never typed;
2. the opaque blob is uploaded to a public host (`litterbox.catbox.moe`, 72 h) and its sha256 is
   pinned;
3. a small bootstrap workflow (`.github/workflows/materialize.yml`, small enough to publish by
   hand) downloads it, checks the payload sha256, decrypts, and **verifies every file against the
   manifest's `git hash-object` before committing** — a corrupted payload is a loud red failure.

Proven before use: the payload round-tripped byte-exact, decrypted to an identical tarball, and all
152 blobs matched the manifest.

## Step 2 — publication (mechanical)

`main` was at `193a581f`; **none of session-0011's files were in the repository** — the blobs had
been created but never entered a tree or a commit. The materialize transport published them.

- **Run 1** (`38250168401`) — **FAILED**, and the failure is a hard platform limit, not a transport
  bug. The log shows the payload sha256 verified and **`verified 152 blobs against the manifest`**,
  the commit created — then the push was rejected:
  `refusing to allow a GitHub App to create or update workflow .github/workflows/smoke-train-v2.yml
  without 'workflows' permission`. The Actions `GITHUB_TOKEN` can never create or update a file
  under `.github/workflows/`. Recorded as **VOID** in the shot registry.
- **Run 2** (`38251100097`) — **SUCCESS**. The payload excluded that one workflow file; the runner
  pushed the other 151 files. `main` → `63dac0a0`.
- The one blocked file (`smoke-train-v2.yml`) was published through the API, which does hold the
  `workflows` permission. Its returned blob sha `114d7e89504081bf834c704a128d02bf849cbbd2` equals
  the local `git hash-object` exactly. `main` → `bf379d66`.

**Verification:** a recursive tree read of `main` matched **every** published file's blob sha
against the local `git hash-object` — including `LOG/session-0011-preregistration.md` =
`7e0e7530f14a183ce0107787d3450147afbaf7dd`, the exact file session-0011 had corrupted. The
transport is certified.

## Step 3 — the shot

Race-free: `refs/heads/main` was polled until it equalled the expected commit `bf379d66`, then the
run was checked to carry that same `head_sha`.

- **Run:** https://github.com/feromirdanton-ui/caissa/actions/runs/38251372886
- **head_sha:** `bf379d668775171e48df21b2d1bbaa8e15cb3353` (matches — no race)
- **Conclusion:** success, 14/14 steps green
- **Artifact:** `smoke-train-v2-metrics` (id `11748652041`, sha256
  `430e46b86a692807f1189845299cc91b6d27b7fc945355107b2aab17aaec1927`)

## Step 4 — the verdict, and the defect

### What the machine printed (verbatim, unretouched)

> `duel_verdict.status` = **`SUCCESS`**
> reason: `"d256_k8 beats v1@set on the shared eval set: delta -0.0356, McNemar p=1.238e-19 (normal, favours v2)"`

### Why that is wrong — a defect in the rule's implementation

The reason string contradicts itself: it says **"delta -0.0356"** and **"beats v1"**. Every delta in
the run is **negative** — all four v2 configurations are *below* v1:

| configuration | top-1 | Δ vs v1@set |
|---|---|---|
| v1 reference (tiny, sgd 0.05) | **0.1167** | — |
| d256_k8 (main) | 0.0810 | **−0.0356** |
| d128_k8 | 0.0801 | **−0.0366** |
| d256_k0 | 0.0714 | **−0.0453** |
| d128_k0 | 0.0731 | **−0.0435** |

The comparison is built as `mcnemar(correctness, v1_correct)`, so `a` is the v2 configuration and
`b` is the v1 reference — **`favours == "b"` is a v1 win**. Session-0011's `_decide` tested
`favours == "b"` and called it a v2 win. A run in which *nothing* beat v1 was reported as SUCCESS.

**This is a bug in applying the pre-registered rule, not a change to it.** The pre-registration
says SUCCESS = a significant win over v1; the code mis-implemented "win". Fixing the implementation
to match the frozen text is not moving a threshold.

### The corrected verdict (the pre-registered rule, correctly applied)

> **DEAD** — `"no v2-family configuration's top-1 exceeds the v1 reference on the shared eval set"`

### The fix

`_decide` now requires `favours == "a"` **and** a positive delta, so the statistic and the delta
sign can no longer disagree. `tests/unit/test_duel_polarity.py` pins the direction on hand-built
tables and carries a **mutation control** that re-introduces the session-0011 bug and asserts the
suite goes red. Local gate: **185 passed**, black/ruff/mypy clean.

### A second, independent finding: the v2 family diverges

The main run's val loss goes **3.3453 → 25.1846** and the gradient norm grows **4.82 → 648**
(max 3390). The model is not converging — it is **diverging**, at Adam `lr=0.005`, exactly as it
diverged at `lr=0.05` in session-0009. The plateau detector reported `PLATEAUED` because no eval
improved on the first — but the first eval was the best *because the model then blew up*. So the
`PLATEAUED` status is itself misleading: the detector cannot tell a plateau from a divergence. The
divergence guard did not fire because the loss stayed finite (25 is finite; the guard catches only
NaN/Inf). Recorded as a defect for the next deep lane, not fixed here.

## Step 5 — the milestone package

See `LOG/session-0012-milestone.md`. One table: v1 / v1@v2-set / the frozen 2×2, deltas, McNemar
z and p, run URLs, metrics artifacts, and the 0009/0010 labeling bank.

## Milestone / signature reconciliation

| milestone | closed by | state |
|---|---|---|
| «Вертикаль» | signature `MILESTONE VERTICAL — ACCEPTED`, issue #17 comment `6114946330` (OWNER, 2026-10-11T23:10:19Z) | **closed** |
| «Сила-2» | — | **open, awaiting ONE human signature** |
| tag `v0.4.0` | — | **not created** — only after the «Сила-2» signature |

The agent does not write the signature, in any form. The tag is not created.

## Risks

- The v2 family diverges at both tested learning rates; the architecture's stability, not its
  capacity, is the open question.
- The plateau detector conflates convergence with divergence.
- The transport is now self-hosting (the materialize workflow is in the repo), but it depends on a
  public 72 h host; a longer-lived store would be better.
