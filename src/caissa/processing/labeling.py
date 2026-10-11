"""Blunder labeling interface (GOAL.md §5 Phase 2).

A label is derived from a *measured* engine delta, never from a heuristic guess: position +
played move + engine evaluation before and after -> centipawn loss -> label. This is the
interface the Queens Guard golden suite and the Act II explanation corpus will be built on.

Engines:
  - :class:`~caissa.guards.counterfactual.StockfishEngine` — a real Stockfish binary.
  - :class:`~caissa.guards.counterfactual.MockEngine` — a deterministic material evaluator,
    **LABELED MOCK**. A report produced with it is tagged ``MOCK`` and must never be
    presented as a result (Article 5).

The labeling *economy* (session-0008) lives here too: :class:`EvalCache` ensures one engine call
per distinct position, and :func:`label_games` states its timeout and checkpoints rather than
silently truncating a long run.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
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
    # --- the labeling economy (session-0008): measured, not guessed -----------------
    engine_calls: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    timeout_reason: str = ""

    @property
    def is_mock(self) -> bool:
        return self.engine == "mock"

    @property
    def positions_per_second(self) -> float:
        """Measured throughput. The only honest basis for an ETA."""
        return self.positions / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def eta_seconds(self, remaining_positions: int) -> float | None:
        """Seconds to finish ``remaining_positions`` at the MEASURED rate, or None if unknown.

        Returning None (rather than a guess) when no rate has been observed is the honest
        answer: an ETA invented before the first measurement is a number, not a forecast.
        """
        rate = self.positions_per_second
        return remaining_positions / rate if rate > 0 else None

    def to_dict(self) -> dict:
        """The report as the JSON the run uploads (labels.json)."""
        return {
            "engine": self.engine,
            "mock": self.is_mock,
            "positions": self.positions,
            "labels": dict(sorted(self.labels.items())),
            "engine_calls": self.engine_calls,
            "cache_hits": self.cache_hits,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "positions_per_second": round(self.positions_per_second, 4),
            "timed_out": self.timed_out,
            "timeout_reason": self.timeout_reason,
            "samples": [
                {"fen": s.fen, "move": s.move, "cp_loss": s.cp_loss, "label": s.label.value}
                for s in self.samples
            ],
        }

    def to_markdown(self) -> str:
        tag = "MOCK — not a result" if self.is_mock else f"engine={self.engine}"
        lines = [
            f"# Blunder-label report ({tag})",
            "",
            f"- positions labeled: {self.positions}",
            f"- engine calls: {self.engine_calls} (cache hits: {self.cache_hits} — one call per "
            "distinct position, not per ply)",
            f"- measured throughput: {self.positions_per_second:.2f} positions/s over "
            f"{self.elapsed_seconds:.1f}s",
        ]
        if self.timed_out:
            lines.append(f"- **STOPPED BY TIMEOUT** — {self.timeout_reason}")
        lines += ["", "## Label distribution", ""]
        lines += [f"- {label}: {count}" for label, count in sorted(self.labels.items())]
        return "\n".join(lines) + "\n"


class EvalCache:
    """Engine evaluations keyed by FEN, so one position is measured **once**.

    The saving is not micro-optimisation, it is structural. Labeling a game evaluates each board
    before a move and the board after it; the "after" of ply *i* is byte-identical to the "before"
    of ply *i+1*, so a game of *n* plies needs *n+1* distinct evaluations rather than *2n* — and a
    transposition or a repeated position across games is free. The cache is the difference between
    a run that finishes in its budget and one that does not.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._values: dict[str, int] = {}
        self.hits = 0
        self.misses = 0
        self.calls = 0

    def get(self, board: chess.Board) -> int:
        key = board.fen()
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        self.calls += 1
        value = self.engine.evaluate(board)
        self._values[key] = value
        return value


def label_move_cached(board: chess.Board, move: chess.Move, cache: EvalCache) -> LabeledMove:
    """Label one move using the shared cache (one engine call per distinct position)."""
    before = cache.get(board)
    after_board = board.copy()
    after_board.push(move)
    after = cache.get(after_board)
    cp_loss = max(0, before - (-after))
    return LabeledMove(
        fen=board.fen(),
        move=move.uci(),
        cp_loss=cp_loss,
        label=classify_loss(cp_loss),
        engine=cache.engine.name,
    )


def label_games(
    games: Iterable[chess.pgn.Game],
    engine: Engine,
    *,
    limit: int = 10_000,
    sample_size: int = 20,
    timeout_s: float | None = None,
    on_checkpoint: Callable[[LabelReport], None] | None = None,
    checkpoint_every: int = 0,
) -> LabelReport:
    """Label up to ``limit`` positions, reusing every evaluation (see :class:`EvalCache`).

    Three honest guarantees, each of which the report *states* rather than hides:

    - **One engine call per distinct position.** The before/after of adjacent plies share a
      position, so the cache roughly halves the calls; ``engine_calls`` and ``cache_hits`` are
      recorded so the saving is visible and a cache regression cannot pass unnoticed.
    - **A timeout is a stated stop, not a silent truncation.** If ``timeout_s`` is exceeded the
      loop stops and the report carries ``timed_out=True`` together with the reason — a partial
      report that says it is partial, which is the only kind worth uploading.
    - **Checkpoints are written from the measured state.** ``on_checkpoint`` is called every
      ``checkpoint_every`` positions with the report as it stands, so a 40-minute run is not lost
      to a 41st-minute failure.
    """
    started = time.monotonic()
    cache = EvalCache(engine)
    report = LabelReport(engine=engine.name)
    checkpoints = 0
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            if report.positions >= limit:
                report.elapsed_seconds = time.monotonic() - started
                _sync(report, cache)
                return report
            if timeout_s is not None and time.monotonic() - started >= timeout_s:
                report.timed_out = True
                report.timeout_reason = (
                    f"stopped after {report.positions} of {limit} positions at the "
                    f"{timeout_s:.0f}s budget — the report is PARTIAL"
                )
                report.elapsed_seconds = time.monotonic() - started
                _sync(report, cache)
                if on_checkpoint is not None:
                    on_checkpoint(report)
                return report
            if board.is_game_over():
                break
            labeled = label_move_cached(board, move, cache)
            report.positions += 1
            report.labels[labeled.label.value] += 1
            if len(report.samples) < sample_size:
                report.samples.append(labeled)
            board.push(move)
            if (
                on_checkpoint is not None
                and checkpoint_every > 0
                and report.positions % checkpoint_every == 0
            ):
                report.elapsed_seconds = time.monotonic() - started
                _sync(report, cache)
                on_checkpoint(report)
                checkpoints += 1
    report.elapsed_seconds = time.monotonic() - started
    _sync(report, cache)
    print(
        f"[label] {report.positions} positions in {report.elapsed_seconds:.1f}s "
        f"({report.positions_per_second:.2f}/s), engine calls {report.engine_calls} "
        f"(cache hits {report.cache_hits}), checkpoints {checkpoints}"
    )
    return report


def _sync(report: LabelReport, cache: EvalCache) -> None:
    """Copy the cache's measured counters onto the report (one source of truth for the numbers)."""
    report.engine_calls = cache.calls
    report.cache_hits = cache.hits


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
