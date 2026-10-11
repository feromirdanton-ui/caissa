"""Numpy tiny policy model — the CPU fallback used by `make smoke`.

This is a REAL (if tiny) linear model, not a mock: it is trained by SGD, its loss decreases,
and its policy head applies the legality mask inside the softmax, so every sampled move is
legal by construction (Article 1). It exists so the smoke test runs with no torch/GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from caissa.representation.mask import legal_move_mask

_PIECE_INDEX = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}
FEATURE_DIM = 64 * 12 + 1  # 12 piece planes + side-to-move


def encode(board: chess.Board) -> np.ndarray:
    x = np.zeros(FEATURE_DIM, dtype=float)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        x[square * 12 + _PIECE_INDEX[(piece.piece_type, piece.color)]] = 1.0
    x[-1] = 1.0 if board.turn == chess.WHITE else 0.0
    return x


@dataclass
class NumpyTinyModel:
    """Linear from/to policy head with a masked softmax."""

    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w_from = rng.normal(0, 0.01, size=(FEATURE_DIM, 64))
        self.w_to = rng.normal(0, 0.01, size=(FEATURE_DIM, 64))

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.outer(x @ self.w_from, np.ones(64)) + np.outer(np.ones(64), x @ self.w_to)

    def masked_probs(self, board: chess.Board) -> np.ndarray:
        logits = self.logits(encode(board))
        mask = legal_move_mask(board)
        z = np.where(mask, logits, -np.inf)
        z = z - np.max(z)
        e = np.where(mask, np.exp(z), 0.0)
        total = e.sum()
        return e / total if total > 0 else e

    def sample_move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        probs = self.masked_probs(board)
        flat = probs.ravel()
        idx = int(rng.choice(flat.size, p=flat / flat.sum()))
        frm, to = divmod(idx, 64)
        move = chess.Move(frm, to)
        if move not in board.legal_moves:  # promotion: pick the queen promotion
            move = chess.Move(frm, to, promotion=chess.QUEEN)
        return move

    def train_step(self, board: chess.Board, target: chess.Move, lr: float = 0.5) -> float:
        """One SGD step on cross-entropy over the masked (from, to) distribution."""
        x = encode(board)
        mask = legal_move_mask(board)
        logits = self.logits(x)
        z = np.where(mask, logits, -np.inf)
        z = z - np.max(z)
        e = np.where(mask, np.exp(z), 0.0)
        probs = e / e.sum()

        loss = -np.log(max(probs[target.from_square, target.to_square], 1e-12))

        grad = probs.copy()
        grad[target.from_square, target.to_square] -= 1.0
        grad[~mask] = 0.0
        grad_from = grad.sum(axis=1)  # [64]
        grad_to = grad.sum(axis=0)  # [64]
        self.w_from -= lr * np.outer(x, grad_from)
        self.w_to -= lr * np.outer(x, grad_to)
        return float(loss)


def teacher_move(board: chess.Board) -> chess.Move:
    """A deterministic heuristic teacher: prefer captures, then central moves.

    This is a teaching signal for the smoke test only — not a chess engine.
    """
    best: tuple[float, chess.Move] | None = None
    for move in board.legal_moves:
        score = 0.0
        victim = board.piece_at(move.to_square)
        if victim is not None:
            score += 10.0 * victim.piece_type
        center = 3.5 - max(
            abs(chess.square_file(move.to_square) - 3.5),
            abs(chess.square_rank(move.to_square) - 3.5),
        )
        score += center
        if best is None or score > best[0]:
            best = (score, move)
    assert best is not None
    return best[1]
