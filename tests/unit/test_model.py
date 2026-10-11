"""Unit tests for the tiny model and the masked softmax (Article 1)."""

from __future__ import annotations

import chess
import numpy as np

from caissa.model.numpy_tiny import NumpyTinyModel, encode, teacher_move
from caissa.model.tiny import masked_softmax


def test_encode_dim(start_board: chess.Board) -> None:
    assert encode(start_board).shape == (64 * 12 + 1,)


def test_masked_softmax_zeroes_illegal() -> None:
    logits = np.zeros((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[12, 28] = True
    probs = masked_softmax(logits, mask)
    assert probs[12, 28] == 1.0
    assert probs[0, 0] == 0.0


def test_sampled_moves_are_legal(start_board: chess.Board) -> None:
    model = NumpyTinyModel(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        move = model.sample_move(start_board, rng)
        assert move in start_board.legal_moves


def test_training_reduces_loss(start_board: chess.Board) -> None:
    model = NumpyTinyModel(seed=2)
    target = teacher_move(start_board)
    first = model.train_step(start_board, target)
    for _ in range(50):
        last = model.train_step(start_board, target)
    assert last < first
