# ADR-0010 — Filter policy: what a training game is

- **Status:** Accepted (session-0006)
- **Date:** 2026-10-11
- **Phase:** 2 — Processing
- **Related:** GOAL.md §5 (Phase 2), §7 (data map), §3 Articles 1, 5; ADR-0006.

## Context

Phase 2 turns the raw ocean into shards. The first question is not *how* to filter but *what a
training game is* — and the answer must be a decision with alternatives, not an accident of code.
The session-0002 audit found the original `is_rated_standard` accepted **any** game carrying two
numeric Elos, which is not the same thing: a rated *variant* game (Chess960, Antichess) also
carries Elos, and a casual game can carry them too. Feeding variant games to a standard-chess
model is a silent corruption of the corpus.

## Options

1. **Keep every game with two numeric Elos.** Rejected: it admits variants and casual games — the
   session-0002 defect. The corpus would be quietly wrong.
2. **Filter on Elo only, ignore the variant.** Rejected: Elo is a strength signal, not a rules
   signal. A 2000-rated Chess960 game is not a standard-chess example.
3. **Require the Event to say "Rated" AND the Variant to be standard/absent, then bucket by Elo and
   classify by time control.** **Chosen.**

## Decision

- **Rated standard only.** A game is admitted only if its `Event` starts with "Rated" **and** its
  `Variant` is empty or one of the standard spellings (`standard`, `chess`, `from position`).
  Everything else is rejected with a named reason (`not_rated_standard`).
- **Elo buckets.** Ratings are bucketed as `0-1200`, `1200-1600`, `1600-2000`, `2000-2400`,
  `2400-4000`. The bucket is a *label on the game*, not a filter by default — Act IV's
  Elo-conditioning needs the whole range, so the default keeps every bucket and the QA report
  shows the distribution.
- **Time-control policy.** A `TimeControl` header is parsed to `(base, increment)` and classified
  by estimated game length `base + 40 * increment`: bullet `< 180 s`, blitz `< 480 s`,
  rapid `< 1500 s`, else classical; `-` is correspondence. The default keeps **every** category
  (the corpus wants all of them); a caller may restrict via `time_controls`.
- **Every rejection is counted, never silent.** `GameFilter.accepts` returns `(accepted, reason)`,
  and the QA report carries a counter per reason. A drop the report cannot explain is a bug.
- **Dedup by identity.** A game's id is its `Site` URL when present, else a sha256 over the
  identity headers plus the move text. The second copy of a game is dropped and counted.

## Consequences

- **Positive:** the corpus is standard chess by construction; a variant game cannot enter it
  silently.
- **Positive:** the QA report explains every drop, so a filter bug shows up as an unexplained
  counter, not as a smaller-than-expected shard.
- **Cost:** the `Event`-prefix rule depends on the source's header convention (Lichess writes
  "Rated Blitz game"). A source that omits "Rated" would be filtered out entirely — recorded as an
  open question, and the reason counter makes it visible immediately.
- **Follow-up:** when a second source (TWIC) is processed, its header convention must be checked
  against this rule before its games are admitted.

## Open questions

- Should the Elo bucket be a *filter* for Act I (a narrower, cleaner corpus) and a *label* only for
  Act IV? The current default keeps all buckets; the decision is deferred to the Act I recipe.
- Should correspondence games be excluded by default? They are kept now; their very long games may
  distort the ply distribution.
