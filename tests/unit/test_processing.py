"""Unit tests for the streaming PGN parser, filters and sharding."""

from __future__ import annotations

from pathlib import Path

from caissa.processing.filters import elo_bucket, game_elo_bucket, is_rated_standard
from caissa.processing.pgn import iter_games
from caissa.processing.sharding import shard_games

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mini.pgn"


def test_iter_games_streams() -> None:
    games = list(iter_games(FIXTURE))
    assert len(games) == 2
    assert games[0].headers["White"] == "Alice"


def test_elo_bucket() -> None:
    assert elo_bucket(1500) == "1200-1600"
    assert elo_bucket(2500) == "2400-4000"


def test_is_rated_standard() -> None:
    games = list(iter_games(FIXTURE))
    assert all(is_rated_standard(g) for g in games)
    assert game_elo_bucket(games[0]) in {"1600-2000", "2000-2400"}


def test_shard_games_writes_zst(tmp_path: Path) -> None:
    report = shard_games(iter_games(FIXTURE), tmp_path, shard_size=1)
    assert report.games == 2
    assert report.kept == 2
    shards = list(tmp_path.glob("*.jsonl.zst"))
    assert len(shards) == 2
    assert not list(tmp_path.glob("*.part"))  # atomic writes leave no .part
