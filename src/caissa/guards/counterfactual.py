"""Counterfactual harness (Article 3: the Word is verified by deed).

Pipeline: (claim) -> (position perturbation) -> (engine delta) -> (verdict).
An explanation is true only if its prediction is confirmed by a perturbation test.
An unverified explanation is marked UNVERIFIED and never enters the Act II corpus.

Engines:
  - MockEngine      — a deterministic material+heuristic evaluator, LABELED MOCK.
  - StockfishEngine — wraps a real Stockfish binary when one is available.
The mock is never passed off as a result: its verdicts carry engine="mock".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import chess


class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    UNVERIFIED = "UNVERIFIED"


class Engine(Protocol):
    name: str

    def evaluate(self, board: chess.Board) -> int:
        """Return an evaluation in centipawns from the side-to-move's perspective."""
        ...


_PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class MockEngine:
    """MOCK engine — deterministic material + tiny mobility term. NOT a real engine.

    It exists so the harness and its tests can run with no external binary. Any output
    produced with it is labeled engine="mock" and must never be reported as a result.
    """

    name = "mock"

    def evaluate(self, board: chess.Board) -> int:
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = _PIECE_CP[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
        # mobility, small weight
        mobility = board.legal_moves.count()
        score += mobility if board.turn == chess.WHITE else -mobility
        return score if board.turn == chess.WHITE else -score


class StockfishEngine:
    """Real Stockfish wrapper. Raises if no binary is found — never silently mocks."""

    name = "stockfish"

    def __init__(self, path: str | None = None, depth: int = 12) -> None:
        import chess.engine  # local import: only needed when a real engine is used

        resolved = path or shutil.which("stockfish")
        if resolved is None:
            raise FileNotFoundError(
                "Stockfish binary not found. Set STOCKFISH=/path/to/stockfish or install it."
            )
        self._engine = chess.engine.SimpleEngine.popen_uci(resolved)
        self._depth = depth

    def evaluate(self, board: chess.Board) -> int:
        import chess.engine

        info = self._engine.analyse(board, chess.engine.Limit(depth=self._depth))
        score = info["score"].pov(board.turn)
        return int(score.score(mate_score=100000) or 0)

    def close(self) -> None:
        self._engine.quit()


@dataclass
class CounterfactualResult:
    fen: str
    move: str
    engine: str
    delta_cp: int
    verdict: Verdict
    note: str = ""


def counterfactual_check(
    board: chess.Board,
    move: chess.Move,
    engine: Engine,
    threshold_cp: int = 50,
) -> CounterfactualResult:
    """Test the claim "this move improves the mover's position".

    Perturbation: play `move`. Delta = eval(after, mover's view) - eval(before, mover's view).
    CONFIRMED if delta >= threshold, REFUTED if delta <= -threshold, else UNVERIFIED.
    """
    if move not in board.legal_moves:
        return CounterfactualResult(
            fen=board.fen(),
            move=move.uci(),
            engine=engine.name,
            delta_cp=0,
            verdict=Verdict.UNVERIFIED,
            note="illegal move — cannot be verified",
        )

    before = engine.evaluate(board)
    after_board = board.copy()
    after_board.push(move)
    after = engine.evaluate(after_board)
    delta = after - before

    if delta >= threshold_cp:
        verdict = Verdict.CONFIRMED
    elif delta <= -threshold_cp:
        verdict = Verdict.REFUTED
    else:
        verdict = Verdict.UNVERIFIED

    return CounterfactualResult(
        fen=board.fen(),
        move=move.uci(),
        engine=engine.name,
        delta_cp=delta,
        verdict=verdict,
        note="MOCK engine — not a result" if engine.name == "mock" else "",
    )
