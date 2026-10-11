"""Board and move tokenization (ADR-0001).

Board = 64 square-tokens + meta-tokens. Move = (from, to, promo) triple.
The same vocabulary is intended to serve board and language (the Word organ).

This module is pure and depends only on python-chess (Article 1: single source of truth).
"""

from __future__ import annotations

import chess

# --- Square tokens: one per square, encoding piece type + colour, or empty. ---
# 12 piece tokens + 1 empty token = 13 square tokens.
PIECE_TOKENS: tuple[str, ...] = (
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",  # white
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",  # black
    ".",  # empty
)
SQUARE_TOKENS: tuple[str, ...] = PIECE_TOKENS  # alias, kept explicit for clarity

# --- Meta tokens: side to move, castling, ep, clocks, ply, Elo bucket. ---
META_TOKENS: tuple[str, ...] = (
    "<w>",
    "<b>",  # side to move
    "<K>",
    "<Q>",
    "<k>",
    "<q>",  # castling rights
    "<ep>",
    "<noep>",  # en-passant field present?
    "<hm>",
    "<ply>",  # halfmove clock, ply (structural markers)
    "<elo>",  # Elo bucket marker (value follows)
)

_PIECE_TO_CHAR = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def _square_token(board: chess.Board, square: int) -> str:
    piece = board.piece_at(square)
    if piece is None:
        return "."
    ch = _PIECE_TO_CHAR[piece.piece_type]
    return ch if piece.color == chess.WHITE else ch.lower()


def board_to_tokens(board: chess.Board, elo_bucket: int | None = None) -> list[str]:
    """Return the token sequence for a position: 64 square-tokens then meta-tokens.

    Square order is a1..h8 (python-chess square indices 0..63).
    """
    tokens = [_square_token(board, sq) for sq in range(64)]
    tokens.append("<w>" if board.turn == chess.WHITE else "<b>")
    for right, tok in (
        (chess.BB_H1, "<K>"),
        (chess.BB_A1, "<Q>"),
        (chess.BB_H8, "<k>"),
        (chess.BB_A8, "<q>"),
    ):
        if board.castling_rights & right:
            tokens.append(tok)
    tokens.append("<ep>" if board.ep_square is not None else "<noep>")
    tokens.append("<hm>")
    tokens.append("<ply>")
    if elo_bucket is not None:
        tokens.append("<elo>")
        tokens.append(str(int(elo_bucket)))
    return tokens


def move_to_tokens(move: chess.Move) -> tuple[int, int, int]:
    """Return the (from, to, promo) triple for a move.

    promo is 0 for no promotion, else the python-chess piece type (2..5).
    """
    return (move.from_square, move.to_square, move.promotion or 0)
