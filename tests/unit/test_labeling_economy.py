"""The labeling economy: cache, timeout, checkpoints, honest ETA (session-0008, fast lane).

Session-0007 left the real-Stockfish labeling job running far longer than a CI budget. This is the
fast-lane answer: prove, offline and deterministically, that

  1. one engine call is made per **distinct position** (the before/after of adjacent plies is the
     same position, so the cache nearly halves the work);
  2. a timeout **stops and says so** in the report rather than truncating silently;
  3. a long run writes **checkpoints**, so a failure at minute 41 does not lose minutes 1-40;
  4. an ETA is derived from a **measured** rate, and is ``None`` (not a guess) before any
     measurement exists.

None of this needs Stockfish: the engine is a counting test double, so the numbers the assertions
check are exact and the test runs in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn
import pytest

from caissa.guards.counterfactual import MockEngine
from caissa.processing.cli import label_main
from caissa.processing.labeling import EvalCache, LabelReport, label_games


class CountingEngine:
    """A deterministic engine that counts how often it is asked to evaluate."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, board: chess.Board) -> int:
        self.calls += 1
        # A cheap, deterministic, position-dependent value (material).
        return sum(
            (1 if board.piece_at(sq) and board.piece_at(sq).color else -1)
            * (board.piece_at(sq).piece_type if board.piece_at(sq) else 0)
            for sq in chess.SQUARES
        )


def _game(moves: list[str]) -> chess.pgn.Game:
    game = chess.pgn.Game()
    node = game
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    return game


LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]


def test_one_engine_call_per_distinct_position() -> None:
    """The cache must evaluate a position once, not twice ('after' of ply i == 'before' of i+1)."""
    engine = CountingEngine()
    report = label_games([_game(LINE)], engine, limit=1000)

    # 10 plies -> 11 distinct positions evaluated, 10 moves labeled.
    assert report.positions == 10
    assert report.engine_calls == 11, f"expected one call per position, got {report.engine_calls}"
    assert report.cache_hits == 9, "each labeled ply after the first should reuse a cached position"
    assert report.engine_calls < 2 * report.positions, "the cache did not halve the engine work"


def test_cache_counts_hits_and_misses() -> None:
    """A direct check on the cache itself: repeats hit, new positions miss."""
    engine = CountingEngine()
    cache = EvalCache(engine)
    board = chess.Board()
    cache.get(board)
    cache.get(board)
    board.push(chess.Move.from_uci("e2e4"))
    cache.get(board)
    assert (cache.calls, cache.hits, cache.misses) == (2, 1, 2)


def test_timeout_stops_and_says_so() -> None:
    """A timeout must produce a PARTIAL report that admits it, never a silent truncation."""
    engine = CountingEngine()
    report = label_games([_game(LINE)], engine, limit=1000, timeout_s=0.0)

    assert report.timed_out is True
    assert report.positions == 0
    assert "PARTIAL" in report.timeout_reason
    assert "PARTIAL" in report.to_markdown()


def test_checkpoints_are_written_during_a_long_run() -> None:
    """Progress must be checkpointed so a late failure does not discard the whole run."""
    engine = CountingEngine()
    seen: list[int] = []
    report = label_games(
        [_game(LINE)] * 5,
        engine,
        limit=1000,
        on_checkpoint=lambda r: seen.append(r.positions),
        checkpoint_every=10,
    )
    assert seen, "no checkpoint was written for a multi-game run"
    assert report.positions == 50
    assert seen == sorted(seen), "checkpoints are not monotonic"


def test_eta_uses_a_measured_rate_and_is_none_before_measurement() -> None:
    """An ETA is a forecast only when a rate has been measured; otherwise it is honestly unknown."""
    fresh = LabelReport(engine="counting")
    assert fresh.eta_seconds(1000) is None, "an ETA was invented before the first measurement"

    measured = LabelReport(engine="counting", positions=100, elapsed_seconds=10.0)
    assert measured.positions_per_second == pytest.approx(10.0)
    assert measured.eta_seconds(500) == pytest.approx(50.0)


def test_mock_engine_is_labeled_a_mock() -> None:
    """Article 5: a mock report says it is a mock, in its own payload and its own prose."""
    report = label_games([_game(LINE)], MockEngine(), limit=1000)
    assert report.is_mock is True
    payload = report.to_dict()
    assert payload["mock"] is True
    assert "MOCK" in report.to_markdown()


def test_cli_label_writes_the_enriched_report(tmp_path: Path) -> None:
    """`caissa-label` must persist the economy numbers and a checkpoint file, tagged MOCK."""
    pgn = tmp_path / "g.pgn"
    with pgn.open("w") as fh:
        for _ in range(3):
            print(_game(LINE), file=fh)
    out = tmp_path / "labels.json"

    code = label_main(
        [
            str(pgn),
            "--engine",
            "mock",
            "--limit",
            "30",
            "--out",
            str(out),
            "--checkpoint-every",
            "10",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["mock"] is True
    assert payload["positions"] == 30
    assert payload["engine_calls"] >= 1
    assert "positions_per_second" in payload
    assert payload["timed_out"] is False
    assert out.with_suffix(".checkpoint.json").exists(), "the checkpoint file was not written"
