"""Legality helpers — thin wrappers over python-chess (Article 1).

No custom rule code lives here. python-chess is the single source of truth.
"""

from __future__ import annotations

import chess


def legal_moves(board: chess.Board) -> list[chess.Move]:
    return list(board.legal_moves)


def is_legal(board: chess.Board, move: chess.Move) -> bool:
    return move in board.legal_moves
