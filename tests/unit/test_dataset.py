"""Position dataset from shards (session-0007, milestone «Vertical»).

A shard row carries the game's moves as UCI. This suite pins the replay rules that make the data
trustworthy: replay from the correct start position, **stop at** a break rather than patch it,
and never leak a game across the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
import zstandard as zstd

from caissa.training.dataset import (
    ShardStats,
    flatten,
    game_positions,
    iter_shard_records,
    load_games,
    move_index,
    split_games,
)


def _write_shard(path: Path, records: list[dict]) -> None:
    payload = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    path.write_bytes(zstd.ZstdCompressor().compress(payload))


def _game_record(uci: list[str], **extra) -> dict:
    record = {"white": "A", "black": "B", "uci": uci, "moves": len(uci)}
    record.update(extra)
    return record


OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


def test_replays_a_game_into_positions() -> None:
    positions = game_positions(_game_record(OPENING))
    assert len(positions) == len(OPENING)
    board, move = positions[0]
    assert board.fen() == chess.Board().fen(), "the first position is the initial one"
    assert move.uci() == "e2e4"


def test_replay_stops_at_the_first_illegal_move() -> None:
    """A corrupt tail must truncate the game, never be guessed around."""
    positions = game_positions(_game_record(["e2e4", "e7e5", "e4e5"]))  # 3rd move is illegal
    assert len(positions) == 2


def test_replay_honours_a_non_standard_start() -> None:
    fen = "r3kbnr/ppp2ppp/2n5/3q4/4P1b1/2B2N2/PPP2PPP/R2QKB1R w KQkq - 0 10"
    positions = game_positions(_game_record(["f1e2"], start_fen=fen))
    assert len(positions) == 1
    assert positions[0][0].fen() == fen, "the replay must begin from the recorded start"


def test_empty_or_missing_moves_yield_nothing() -> None:
    assert game_positions({"white": "A", "uci": []}) == []
    assert game_positions({"white": "A"}) == []


def test_iter_shard_records_reads_every_shard(tmp_path: Path) -> None:
    _write_shard(tmp_path / "shard-00000.jsonl.zst", [_game_record(OPENING)])
    _write_shard(tmp_path / "shard-00001.jsonl.zst", [_game_record(OPENING), _game_record(OPENING)])
    records = list(iter_shard_records(tmp_path))
    assert len(records) == 3


def test_load_games_stops_on_a_game_boundary(tmp_path: Path) -> None:
    """max_positions caps the load, but never in the middle of a game."""
    records = [_game_record(OPENING) for _ in range(10)]
    _write_shard(tmp_path / "shard-00000.jsonl.zst", records)
    stats = ShardStats(shards=["shard-00000.jsonl.zst"])
    games = load_games(tmp_path, max_positions=8, min_moves=2, stats=stats)
    total = sum(len(g) for g in games)
    assert total >= 8, "the cap triggers after the boundary, so total may exceed it"
    assert all(len(g) == len(OPENING) for g in games), "no half game is ever loaded"
    assert stats.games == len(games)


def test_load_games_counts_short_and_broken_replays(tmp_path: Path) -> None:
    records = [
        _game_record(["e2e4"]),  # too short for min_moves=2
        _game_record(["e2e4", "e7e5", "e4e5"]),  # broken tail
        _game_record(OPENING),
    ]
    _write_shard(tmp_path / "shard-00000.jsonl.zst", records)
    stats = ShardStats(shards=["shard-00000.jsonl.zst"])
    games = load_games(tmp_path, max_positions=100, min_moves=2, stats=stats)
    assert stats.skipped_games == 1
    assert stats.broken_replays == 1
    assert len(games) == 2


def test_split_rejects_a_bad_fraction() -> None:
    with pytest.raises(ValueError):
        split_games([[]], val_fraction=1.0)


def test_flatten_respects_the_limit() -> None:
    games = [[(chess.Board(), chess.Move.from_uci("e2e4"))] * 3] * 4
    assert len(flatten(games, limit=5)) == 5
    assert len(flatten(games)) == 12


def test_move_index_matches_from_to_encoding() -> None:
    move = chess.Move.from_uci("e2e4")
    assert move_index(move) == move.from_square * 64 + move.to_square
    assert move_index(move) == 12 * 64 + 28
