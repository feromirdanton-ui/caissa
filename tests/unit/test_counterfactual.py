"""Unit tests for the counterfactual harness (Article 3)."""

from __future__ import annotations

import chess

from caissa.guards.counterfactual import MockEngine, Verdict, counterfactual_check


def test_mock_engine_is_labeled() -> None:
    assert MockEngine().name == "mock"


def test_illegal_move_is_unverified(start_board: chess.Board) -> None:
    result = counterfactual_check(start_board, chess.Move.from_uci("e2e5"), MockEngine())
    assert result.verdict is Verdict.UNVERIFIED
    assert "illegal" in result.note


def test_capture_is_confirmed(start_board: chess.Board) -> None:
    # Set up a free capture: black queen on d5, white pawn can take on e4? Use a simple one.
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    result = counterfactual_check(board, chess.Move.from_uci("e4d5"), MockEngine())
    assert result.verdict is Verdict.CONFIRMED
    assert result.engine == "mock"
    assert "MOCK" in result.note
