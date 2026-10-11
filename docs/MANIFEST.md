# MANIFEST — the concept of Caïssa-1

> Written in the construction agent's own words, brief and precise (GOAL.md §11, task 2).
> This is the *concept* manifest. The *data* manifest (source / sha256 / volume / schema / rows)
> lives in `data/manifests/` and is described in §4 below.

## 1. The problem

A chess engine today is one of two things:

- **Strong and mute** — Stockfish, AlphaZero. It plays better than any human and can explain
  nothing. Its strength is a number, not an understanding.
- **Eloquent and blind** — an LLM that predicts moves as text. It talks fluently about chess and
  hallucinates the rules: illegal moves, pieces that teleport, a board that drifts.

Neither is a *being* you can learn from. Caïssa-1 is the attempt to be both at once: strong **and**
able to say why, with a memory of patterns and the patience to teach.

## 2. The four organs

| Organ | System | What it is | Where it lives |
|---|---|---|---|
| **Intuition** | 1 | One forward pass → a move + an evaluation. Illegal moves are *architecturally impossible*: the legality mask sits inside the softmax. | `src/caissa/model/`, `src/caissa/representation/` |
| **Word** | 2 | Verbal reasoning before a hard move — candidates, motives, risks. Verified by deed, not by plausibility. | `src/caissa/model/` (comment head), `src/caissa/guards/counterfactual.py` |
| **Memory** | — | A retrieval index of *motives*: "isolated pawn d5", "weak f7", "Carlsbad structure". | `src/caissa/memory/` |
| **Queens Guard** | — | A hygiene module. Before a move is published: every check, capture, pin and fork — mine and the opponent's. The move does not go out until hygiene passes. | `src/caissa/guards/` |

The Guard is not a feature. It was born from a real queen blunder (see Requiem-1, GOAL.md §8) and
it is the meaning of the project: strength that does not understand itself is not worth shipping.

## 3. The five acts (prepared here, launched elsewhere)

1. **Act I — Language.** SFT on move prediction + position reconstruction (anti-board-drift).
2. **Act II — Voice.** SFT on explanations, counterfactually verified by an engine. An explanation
   is *true* only if its prediction survives a perturbation test.
3. **Act III — Memory.** Retrieval of motives, integrated into generation.
4. **Act IV — Strength.** RL self-play (MCTS priors), KL distillation from Stockfish with annealing,
   Elo-conditioning. Honest weakening: "play like 1900" means choosing moves statistically typical
   of 1900 humans — not blundering at random.
5. **Act V — Teacher.** A mentor bot for live students; its logs return to training.

## 4. The value axis

Not Elo — that summit is taken. The axis is **understanding**:

- blunder-rate **below** human at equal Elo,
- explanations that are **verifiable** (counterfactual harness), not merely plausible,
- measurable **student learnability**.

## 5. Data manifest (the only data that enters git)

Per Article 8, no data or weights are committed. What is committed is a manifest per source:

```json
# data/manifests/<source>.json
source: https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
sha256: <hex>
bytes: <int>
rows: <int|null>
schema: pgn
fetched_at: <iso8601>
license: <spdx-or-url>
notes: <free text>
```

A manifest is written **at download time**, never from memory (GOAL.md §7). The loader is
resumable (`.part` + HTTP Range) and idempotent: a completed, hash-verified shard is never re-downloaded.

## 6. What this repository is not

- It is **not** a trained model. There are no training results here, and none are fabricated.
- It is **not** a data dump. It contains loaders and manifests, not terabytes.
- It is **not** finished. The work is endless — there is always a next stone.
