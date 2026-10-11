"""Unit tests for the representation organ (ADR-0001)."""

from __future__ import annotations

import chess
import numpy as np

from caissa.representation.mask import apply_mask, legal_move_mask, promo_mask
from caissa.representation.tokens import board_to_tokens, move_to_tokens


def test_board_tokens_length(start_board: chess.Board) -> None:
    tokens = board_to_tokens(start_board)
    assert len(tokens) == 64 + 8  # 64 squares + side, 4 castling, ep, hm, ply
    assert tokens[0] == "R"  # a1 rook
    assert tokens[4] == "K"  # e1 king
    assert tokens[63] == "r"  # h8 rook
    assert "<w>" in tokens


def test_board_tokens_elo_bucket(start_board: chess.Board) -> None:
    tokens = board_to_tokens(start_board, elo_bucket=1900)
    assert tokens[-2] == "<elo>"
    assert tokens[-1] == "1900"


def test_move_tokens() -> None:
    assert move_to_tokens(chess.Move.from_uci("e2e4")) == (12, 28, 0)
    assert move_to_tokens(chess.Move.from_uci("e7e8q")) == (52, 60, chess.QUEEN)


def test_legal_mask_matches_python_chess(start_board: chess.Board) -> None:
    mask = legal_move_mask(start_board)
    assert mask.shape == (64, 64)
    assert mask.sum() == 20  # 20 legal first moves
    for move in start_board.legal_moves:
        assert mask[move.from_square, move.to_square]


def test_promo_mask_shape(start_board: chess.Board) -> None:
    mask = promo_mask(start_board)
    assert mask.shape == (64, 64, 5)


def test_apply_mask_forbids_illegal() -> None:
    logits = np.zeros((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[12, 28] = True  # e2e4 only
    masked = apply_mask(logits, mask)
    assert np.isneginf(masked[0, 0])
    assert masked[12, 28] == 0.0
