# ADR-0006 — Data-acquisition guarantees: an honest minimum, measured

- **Status:** Accepted (session-0005)
- **Date:** 2026-10-11
- **Phase:** 1 — Acquisition
- **Related:** GOAL.md §7 (data map), §3 Articles 4, 5, 8, 14; §6.6 (resource hygiene), ADR-0004.

## Context

Phase 1 must deliver loaders for the ocean of sources in GOAL.md §7 that are resumable, carry a
sha256, and write a manifest — and `make data-sample` must pull "a month of Lichess + 1000 puzzles
+ a TWIC month" (Phase 1 DoD, §5). Two facts about the real world break a naive reading of that DoD:

1. **A month of Lichess is units of GB.** The build environment's disk (`/workspace`, overlay) is
   4 GB. Pulling the reference month is possible only on an environment that has the gigabytes;
   on this one it is not, and pretending otherwise would be a silent truncation or a fabricated
   volume — both forbidden by Article 5.
2. **A "day of Lichess" is not separately addressable.** Lichess publishes one `.zst` per month;
   there is no per-day URL. The archive is *chronological*, so its **head is the start of the
   month** — a day of play, obtainable as a byte prefix.
3. **TWIC issue numbers are not derivable from a date.** The site serves `twic{NNNN}g.zip` with no
   index we can rely on. Guessing the number for a month and asserting the result would be a claim,
   not a measurement.

## Options

1. **Pull the full month regardless and let the environment fail.** Rejected: it burns the budget,
   stops the pipeline, and tempts a "shorten it quietly" fix — the failure class Article 5 exists
   to prevent.
2. **Fabricate a small synthetic dataset and label it as the sample.** Rejected: a simulation
   passed off as a result (Article 5). It also proves nothing about the real sources' behaviour
   (Range, `Content-Length`, zstd framing).
3. **Domain-delegate: pull the honest minimum that the environment can actually afford, and
   measure everything.** A byte *prefix* of a real Lichess month is a real day of real data; a
   discovered TWIC issue is a real issue; a 1000-row sample of the real puzzles archive is real
   puzzles. **Chosen.**

## Decision

The loaders keep the `lichess_bulk` guarantees and add three source-specific ones:

- **`download_prefix`** (Lichess day-sample): a resumable, atomic, sha256-certified download of the
  first N bytes. Range is *bounded* (`bytes=<n>-<prefix-1>`) and validated; a `200`-instead-of-`206`
  restarts from zero (P0-2); a short body raises rather than publishing. The exact byte count is
  recorded in the QA report — a truncated sample is **named**, not hidden.
- **Puzzles**: one streaming decompression pass both counts every row (`rows` in the manifest) and
  writes the first `limit` rows to a sample CSV with a rating histogram. The manifest certifies the
  sha256 of the **written** archive before it is recorded (Article 14).
- **TWIC**: the month is *discovered* by probing a window around a **measured** anchor
  (issue 1500 → 2023-08-08, measured by `HEAD` on 2026-10-11) and keeping only issues whose
  `Last-Modified` falls inside the requested month. No issue number is ever assumed; if the window
  finds nothing, the loader refuses and records nothing. Public-site requests are spaced by a
  politeness pause (§6.6).

Operationally:

- `make data-sample` (via `scripts/run.py data-sample`) pulls the three sources into `data/`,
  checks a **disk quota** before starting and prints free space at start and end, cleans/reports
  `.part` files, and writes `data/qa/qa-lite.json`.
- The runner proves it a second way: `.github/workflows/data-sample.yml` runs the same command on a
  clean runner, verifies the manifests, and includes an in-run **resume proof** (truncate the day
  file to 50%, restart, assert fewer bytes fetched and a matching sha256).
- Data is **never** tracked: `data/**` is ignored by path and the clone guard fails the build if a
  tracked file is ignored.

## Consequences

- **Positive:** every number in the QA report is measured from bytes on disk; the manifest is a
  certification (Article 14), not a journal.
- **Positive:** the pipeline is fully exercised on real sources even on a 4 GB box — the day-sample
  is genuinely a day, just obtained as a prefix.
- **Cost / honest gap:** the prefix is a *truncated* stream. Its tail is not decoded (a zstd stream
  ends mid-block); the QA scan reads up to the last complete block and drops the remainder. This is
  recorded as a known limitation, not presented as a full-month parse. Phase 2's streaming parser is
  what handles whole archives.
- **Cost:** the TWIC anchor ages. The probe window (±12 weeks, widened once) makes a drifted
  estimate self-correcting, but a very old month may need an explicit anchor. Recorded as an open
  question below.
- **Follow-up:** the full-month path (no prefix) stays available by raising `lichess_prefix_bytes`
  to the archive size on an environment with the disk; the DoD text should be read as "a day of
  Lichess bulk" until such an environment exists.

## Open questions

- Should the anchor be a small committed index (issue → date) generated once, instead of a
  constant plus a probe?
- Should the day-sample be defined by *decompressed* volume (N games) rather than compressed bytes,
  so the QA count is stable across compression versions?
- When a full-month environment exists, should `data-sample` switch automatically on free-space
  detection, or stay explicit via config?
