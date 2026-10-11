# Session 0006 — Phase 2 «Processing»

**Date:** 2026-10-11 · **Phase:** 2 — Processing · **Branch:** `main` (direct, per ADR-0004/0009)

## Precondition — the human signature (verified, quoted verbatim)

Phase 2 may not start until Phase 1 is signed. The signature is in issue #17:

> **PHASE 1 — ACCEPTED**
> Feromir Danton · 22:50 11:10:2026

- Comment: https://github.com/feromirdanton-ui/caissa/issues/17#issuecomment-6113045248
- Author: `feromirdanton-ui` (repository OWNER)
- Verdict: **signed** — Phase 2 is open.

The signature is quoted, never written by the agent. It is the human's act.

## Step 0 — coverage of the layer-2 tail

`git diff --stat df4ac577 0f87a7ba` touched `LOG/session-0005.md` and two test files —
tests are code, so the rule "code changed → re-run the acceptance workflow" applies.

- `data-sample.yml` run #5 on `0f87a7ba` → **SUCCESS**
  https://github.com/feromirdanton-ui/caissa/actions/runs/38169910736
- Verdict: coverage closed.

## Step 1 — ADR-0009 «Publication through the API»

`DECISIONS/adr-0009-publication-via-api.md`. The transport is not the decision — the
commits are real commits. The **ritual** is: **publish → verify → CI**, and a mismatch is
a loud failure. Motivation: the session-0005 defect (a commit declared a fix the tree did
not carry). The skill `publishing-files-to-github-via-api` now states plainly that the git
CLI gives this protection for free and the API does not.

**The ritual was executed on every publish this session** — each commit's tree was listed
recursively and compared against the intended-file manifest before the ref was updated.

## Step 2 — Phase 2 core (GOAL §5)

| Module | What it does |
|---|---|
| `processing/pgn.py` | Streaming PGN parser — generators, O(1) memory; loud `PgnParseError`; `TruncatedStreamError` for a truncated zstd stream |
| `processing/filters.py` | Rated-standard predicate (Event + Variant), Elo buckets, time-control classification (ADR-0010) |
| `processing/dedup.py` | Game identity (Site URL, else sha256) + deduplicator |
| `processing/sharding.py` | Atomic `jsonl.zst` shards + QA report with per-shard sha256 |
| `processing/labeling.py` | Blunder labeling from a measured engine delta; MOCK labeled as MOCK |
| `processing/cli.py` | `caissa-shard` / `caissa-label` |

`make processing` and `make label` added; `scripts/run.py` gained the `processing` and
`label` targets (ADR-0005: one source of commands).

## Step 3 — the runner as the acceptance machine for processing

`.github/workflows/processing-sample.yml` (workflow_dispatch): clean runner → clone guard
→ pull the honest minimum → stream → filter → dedup → shard → QA → verify every shard
sha256 → upload the QA artifact. Data never enters git (`data/**` ignored by path).

### Run history — the runner found what the local suite could not

**Run #1 — RED.** https://github.com/feromirdanton-ui/caissa/actions/runs/38171149118
The pipeline died on the real Lichess day-prefix. Root cause: **a Lichess archive is a
concatenation of zstd frames**, and `decompressobj` stops at the end of the first frame
(and raises if fed more). The local fixtures were single-frame, so 73 tests were green
while the runner was red. Fixed in `_ZstdRaw._feed` by rolling over to a fresh
decompressobj via `unused_data`; `_at_frame_start` is the honest truncation signal.

**Run #2 — GREEN.** https://github.com/feromirdanton-ui/caissa/actions/runs/38172049856
on `82acedfe`. All 12 steps green. **But the log exposed a silent false success**: the
blunder-labeling step failed with `unrecognized arguments: --label` and `|| true` swallowed
it — the run reported success while the labeler never ran. Article 5 violation.

**Run #3 — GREEN (final).** Dispatched on `25296149` after the fix below.

### Measured QA (from run #2's log — real numbers, not estimates)

- games seen: **56 962** · kept: **56 843** · duplicates: 0 · parse-skipped: 0
- truncated input: **True** (the day-prefix is a byte prefix — named, not hidden)
- rejections: `not_rated_standard: 119`
- Elo buckets: 0-1200: 432 · 1200-1600: 28 919 · 1600-2000: 26 578 · 2000-2400: 914
- time controls: blitz 20 777 · bullet 16 303 · rapid 18 367 · classical 1 294 · correspondence 102
- shards: **6** (5 × 10 000 + 1 × 6 843), every sha256 verified against the file on disk
- artifact `processing-sample-qa` id 11714790277

## Step 4 — the label-routing fix

`scripts/run.py` passed `--label` as a flag; the dispatcher expects the `label`
subcommand. Fixed, and the `|| true` mask removed — a labeling failure now fails the run.
Two regression tests pin the routing. 75 passed, lint clean.

## DoD deviation — stated openly

The Phase 2 DoD is **1M games raw→shards reproducibly**. The environment cannot hold 1M
games honestly, so the run processed the **day-prefix: 56 962 games** — the honest scale.
The full procedure for an engineer with the disk is in `docs/TRAINING.md`. A small honest
dataset beats a silent truncation, and both beat a fabricated volume (Article 5).

## Artifacts

- ADR-0009, ADR-0010 · `processing/*` · `tests/unit/test_processing_{parser,filters,shards}.py`
- `processing-sample.yml` · `scripts/run.py` · `Makefile` · `pyproject.toml`
- Commits: `8af383d5` (core) → `82acedfe` (zstd frames) → `25296149` (label routing)

## Risks

- The `Event`-prefix rule (ADR-0010) depends on the source's header convention; a source
  that omits "Rated" would be filtered out entirely. The reason counter makes it visible.
- The labeler ran with the MOCK engine on the runner (no Stockfish binary). The report is
  tagged MOCK — it is a wiring test, not a strength result.
- The API publish path cannot catch a *forgotten* file, only a *dropped* one (ADR-0009).

## Next stone

Phase 2 acceptance package to the human signer. Until the signature, Phase 3 does not
start — work continues on the evergreen backlog §6.7 (golden suite, guard traps).
