# Session 0009 — FAST LANE: labeling economy on REAL Stockfish (MOCK debt closed)

**This closes the Phase-2 MOCK debt.** Session 0006 shipped the blunder labeler with a MOCK engine
fallback; session 0007 fixed the Stockfish *fetch*; this session runs the labeler on a **real
Stockfish 19** over **10 000 positions drawn from the actual shards** the vertical trains on.

## What was run

```
python -m caissa.processing.cli label data/qa/sample.pgn \
    --limit 10000 --engine stockfish --out data/qa/labels.json \
    --timeout-s 1500 --checkpoint-every 500
```

`data/qa/sample.pgn` is not a fixture: it is **400 games reconstructed from `data/shards/*.jsonl.zst`**
by `scripts/shards_to_pgn.py`, so the labeler measures the same games the model sees.

## Engine provenance (a tool, verified, not declared)

| field | value |
|---|---|
| version | `sf_19` (official-stockfish/Stockfish) |
| tarball sha256 | `9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f` (verified) |
| binary sha256 | `0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19` (measured) |
| manifest | `data/manifests/stockfish.json` |

The binary sha256 matches the one measured in session-0007 — the same engine, re-verified.

## Measured result (real engine, not MOCK)

| metric | value |
|---|---|
| positions labeled | **10 000** |
| elapsed | 168.6 s |
| throughput | **59.32 positions/s** |
| engine calls | 9 464 |
| cache hits | **10 536** |
| timed out | **False** (completed within the 1500 s budget) |
| engine | `stockfish` (not MOCK) |

**Label distribution:** BEST 3 936 · GOOD 2 445 · INACCURACY 1 351 · BLUNDER 1 219 · MISTAKE 1 049.

## The economy, stated as a number

The `EvalCache` is the difference between a run that fits its budget and one that does not. The
before/after of adjacent plies share a position, so a game of *n* plies needs *n+1* distinct
evaluations rather than *2n*. Here that is **10 536 cache hits against 9 464 engine calls** — the
cache served **52.7%** of all lookups for free. Without it the run would have made ~20 000 engine
calls, roughly doubling the wall-clock.

## Honest ETA

At the measured **59.32 positions/s**, the ETA for a larger run is arithmetic, not a guess:

| target | ETA at 59.32 pos/s |
|---|---|
| 10 000 (this run) | 169 s (measured) |
| 100 000 | ~28 min |
| 1 000 000 | ~4.7 h |

The timeout policy is a **stated stop, not a silent truncation**: had the 1500 s budget been hit,
the report would carry `timed_out=True` and a reason naming how many of the 10 000 positions were
reached. It was not hit — the run completed all 10 000.

## Acceptance

This is a **separate acceptance** from the milestone: it closes the MOCK debt and does not block the
vertical's verdict. The report is tagged with the real engine name, so it cannot be mistaken for a
MOCK wiring test.
