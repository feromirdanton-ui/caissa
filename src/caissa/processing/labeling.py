"""Blunder labeling interface (GOAL.md §5 Phase 2).

A label is derived from a *measured* engine delta, never from a heuristic guess: position +
played move + engine evaluation before and after -> centipawn loss -> label. This is the
interface the Queens Guard golden suite and the Act II explanation corpus will be built on.

Engines:
  - :class:`~caissa.guards.counterfactual.StockfishEngine` — a real Stockfish binary.
  - :class:`~caissa.guards.counterfactual.MockEngine` — a deterministic material evaluator,
    **LABELED MOCK**. A report produced with it is tagged ``MOCK`` and must never be
    presented as a result (Article 5).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import chess
import chess.pgn

from caissa.guards.counterfactual import Engine, MockEngine

# Centipawn-loss thresholds (Lichess-style). A loss is measured, not assumed.
INACCURACY_CP = 50
MISTAKE_CP = 100
BLUNDER_CP = 200


class BlunderLabel(str, Enum):
    BEST = "BEST"
    GOOD = "GOOD"
    INACCURACY = "INACCURACY"
    MISTAKE = "MISTAKE"
    BLUNDER = "BLUNDER"


def classify_loss(cp_loss: int) -> BlunderLabel:
    """Map a centipawn loss to a label. ``cp_loss`` is >= 0 (loss for the mover)."""
    if cp_loss >= BLUNDER_CP:
        return BlunderLabel.BLUNDER
    if cp_loss >= MISTAKE_CP:
        return BlunderLabel.MISTAKE
    if cp_loss >= INACCURACY_CP:
        return BlunderLabel.INACCURACY
    if cp_loss <= 10:
        return BlunderLabel.BEST
    return BlunderLabel.GOOD


@dataclass
class LabeledMove:
    fen: str
    move: str
    cp_loss: int
    label: BlunderLabel
    engine: str

    @property
    def is_mock(self) -> bool:
        return self.engine == "mock"


def label_move(board: chess.Board, move: chess.Move, engine: Engine) -> LabeledMove:
    """Label one move by the centipawn loss it incurred, from the mover's perspective."""
    before = engine.evaluate(board)
    after_board = board.copy()
    after_board.push(move)
    after = engine.evaluate(after_board)
    # `evaluate` is from the side-to-move's view; after the move it is the opponent's view,
    # so negate it to keep both numbers in the mover's frame.
    cp_loss = max(0, before - (-after))
    return LabeledMove(
        fen=board.fen(),
        move=move.uci(),
        cp_loss=cp_loss,
        label=classify_loss(cp_loss),
        engine=engine.name,
    )


@dataclass
class LabelReport:
    engine: str
    positions: int = 0
    labels: Counter = field(default_factory=Counter)
    samples: list[LabeledMove] = field(default_factory=list)

    @property
    def is_mock(self) -> bool:
        return self.engine == "mock"

    def to_markdown(self) -> str:
        tag = "MOCK — not a result" if self.is_mock else f"engine={self.engine}"
        lines = [
            f"# Blunder-label report ({tag})",
            "",
            f"- positions labeled: {self.positions}",
            "",
            "## Label distribution",
            "",
        ]
        lines += [f"- {label}: {count}" for label, count in sorted(self.labels.items())]
        return "\n".join(lines) + "\n"


def label_games(
    games: Iterable[chess.pgn.Game],
    engine: Engine,
    *,
    limit: int = 10_000,
    sample_size: int = 20,
) -> LabelReport:
    """Label up to ``limit`` positions drawn from ``games``; keep ``sample_size`` examples."""
    report = LabelReport(engine=engine.name)
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            if report.positions >= limit:
                return report
            if board.is_game_over():
                break
            labeled = label_move(board, move, engine)
            report.positions += 1
            report.labels[labeled.label.value] += 1
            if len(report.samples) < sample_size:
                report.samples.append(labeled)
            board.push(move)
    return report


def default_engine(prefer_stockfish: bool = True) -> Engine:
    """A real Stockfish when one is available, else the MOCK engine (honestly labeled).

    The fallback is **reported, never silent**: if a Stockfish was asked for and not found, the
    caller is told which binary was missing and that the result will be a MOCK. A bare
    ``except ...: pass`` here would hide exactly the condition that makes the report unusable as
    a result (Article 5), so the reason is named instead.
    """
    if prefer_stockfish:
        from caissa.guards.counterfactual import StockfishEngine

        try:
            return StockfishEngine()
        except FileNotFoundError as exc:
            print(
                f"[label] Stockfish unavailable ({exc}) — falling back to the MOCK engine; "
                "the report is tagged MOCK and is not a strength result."
            )
    return MockEngine()
