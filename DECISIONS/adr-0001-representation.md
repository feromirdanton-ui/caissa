# ADR-0001 — Board and move representation

- **Status:** Accepted
- **Date:** 2026-10-11
- **Deciders:** Caïssa-1 construction agent
- **Supersedes:** —
- **Related:** GOAL.md §4.1, §3 Article 1

## Context

The Intuition organ must produce a move in a single forward pass, and an illegal move must be
*architecturally impossible* (Article 1). The Word organ must share one dictionary with the board
so that language and position are the same alphabet. The representation therefore decides both the
model's input/output shape and the shape of the legality mask.

## Options considered

1. **FEN / text tokens.** Feed the position as a string; predict the move as a string.
   - *Pro:* trivially shares a vocabulary with language; LLM-friendly.
   - *Con:* board drift (the model hallucinates pieces); the legality mask is not expressible as a
     simple tensor operation; illegal moves are only *discouraged*, never impossible.
2. **AlphaZero-style planes (8×8×C).** Convolutional input, policy over 4672 move indices.
   - *Pro:* proven; strong.
   - *Con:* the move index space is a hand-built enumeration; the mask is a fixed 4672-vector that
     must be kept in sync with the rules; poor fit for a decoder-only language model.
3. **64 square-tokens + meta-tokens; move = (from, to, promo) triple.** *(chosen)*
   - *Pro:* the legality mask is a `64×64×promo` boolean tensor — trivial and complete; the same
     token vocabulary serves board and language; a decoder-only transformer with RoPE fits naturally.
   - *Con:* more tokens per position than a plane encoding; needs careful meta-token design.

## Decision

Adopt **option 3**.

- **Board** = 64 square-tokens (one per square, encoding piece type + colour, or empty) plus
  **meta-tokens**: side to move, castling rights, en-passant field, halfmove clock, ply, and an
  **Elo bucket** (for Act IV conditioning).
- **Move** = a triple of tokens `from | to | promo`. The policy head emits three distributions; the
  legality mask is applied **inside the softmax** over the `from` and `to` axes.
- **History** = the last K positions (K and serialization format to be fixed by a later ADR).
- **Single source of truth for rules** = `python-chess`. No custom rule code (Article 1).

## Consequences

- The legality mask is a pure function of the position, computed with `python-chess`, and is the
  heart of the project (`src/caissa/guards/legality_mask.py`).
- Property tests can assert, over random positions, that *every* legal move is allowed by the mask
  and *every* illegal move is not (Phase 3 DoD).
- The Word organ inherits the same vocabulary — board and language share one dictionary.
- A later ADR must fix K (history depth) and the exact meta-token layout before Act I training.

## Open questions (deferred to later ADRs)

- History depth K and its serialization.
- Whether the promo axis is a separate head or folded into the `to` axis.
- Elo-bucket granularity for conditioning.
