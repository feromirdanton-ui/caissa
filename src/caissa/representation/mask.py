"""The legality mask — heart of the project (Article 1).

A pure function of the position, computed with python-chess. The mask is applied
*inside the softmax* so that an illegal move is architecturally impossible.

Shapes:
  legal_move_mask(board) -> bool[64, 64]        mask[from, to] = any legal move from->to
  promo_mask(board)      -> bool[64, 64, 5]     mask[from, to, promo] (promo 0 = none)
"""

from __future__ import annotations

import chess
import numpy as np

PROMO_PIECES = (0, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean [64, 64] mask of legal from->to moves for the side to move."""
    mask = np.zeros((64, 64), dtype=bool)
    for move in board.legal_moves:
        mask[move.from_square, move.to_square] = True
    return mask


def promo_mask(board: chess.Board) -> np.ndarray:
    """Boolean [64, 64, 5] mask; index 0 is 'no promotion', 1..4 are N,B,R,Q."""
    mask = np.zeros((64, 64, 5), dtype=bool)
    for move in board.legal_moves:
        idx = PROMO_PIECES.index(move.promotion) if move.promotion else 0
        mask[move.from_square, move.to_square, idx] = True
    return mask


def apply_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set masked logits to -inf so a softmax over them yields exactly 0.

    This is the operation that makes illegal moves impossible, not merely unlikely.
    """
    out = np.array(logits, dtype=float, copy=True)
    out[~mask] = -np.inf
    return out
