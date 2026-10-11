"""Property tests: the legality mask is complete and sound (GOAL.md §5 Phase 3).

Over random positions: every legal move is allowed by the mask, and every illegal move is not.
"""

from __future__ import annotations

import random

import chess
import pytest

from caissa.representation.mask import legal_move_mask


def _random_positions(n: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(n):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        yield board


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mask_allows_every_legal_move(seed: int) -> None:
    for board in _random_positions(200, seed=seed):
        mask = legal_move_mask(board)
        for move in board.legal_moves:
            assert mask[
                move.from_square, move.to_square
            ], f"legal move {move.uci()} missing from mask in {board.fen()}"


def test_mask_rejects_illegal_moves() -> None:
    for board in _random_positions(200, seed=7):
        mask = legal_move_mask(board)
        legal_pairs = {(m.from_square, m.to_square) for m in board.legal_moves}
        for frm in range(64):
            for to in range(64):
                if mask[frm, to]:
                    assert (
                        frm,
                        to,
                    ) in legal_pairs, (
                        f"mask allows illegal {chess.Move(frm, to).uci()} in {board.fen()}"
                    )


def test_mask_covers_special_moves() -> None:
    # Castling
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    mask = legal_move_mask(board)
    assert mask[chess.E1, chess.G1]  # O-O
    assert mask[chess.E1, chess.C1]  # O-O-O
    # En passant
    board = chess.Board("8/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    mask = legal_move_mask(board)
    assert mask[chess.E5, chess.D6]
    # Promotion
    board = chess.Board("8/P7/8/8/8/8/8/4K3 w - - 0 1")
    mask = legal_move_mask(board)
    assert mask[chess.A7, chess.A8]
