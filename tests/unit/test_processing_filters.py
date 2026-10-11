"""Unit tests for the game filters (ADR-0010)."""

from __future__ import annotations

from pathlib import Path

from caissa.processing.filters import (
    GameFilter,
    elo_bucket,
    game_elo_bucket,
    is_rated_standard,
    parse_time_control,
    time_control_category,
)
from caissa.processing.pgn import iter_games

FIXTURES = Path(__file__).parent.parent / "fixtures"
MINI = FIXTURES / "mini.pgn"
MESSY = FIXTURES / "messy.pgn"


def test_elo_bucket() -> None:
    assert elo_bucket(1500) == "1200-1600"
    assert elo_bucket(2500) == "2400-4000"
    assert elo_bucket(5000) == "unknown"


def test_parse_time_control() -> None:
    assert parse_time_control("300+0") == (300, 0)
    assert parse_time_control("180+2") == (180, 2)
    assert parse_time_control("-") is None
    assert parse_time_control("garbage") is None


def test_time_control_category() -> None:
    assert time_control_category("60+0") == "bullet"
    assert time_control_category("300+0") == "blitz"
    assert time_control_category("600+0") == "rapid"
    assert time_control_category("1800+0") == "classical"
    assert time_control_category("-") == "correspondence"


def test_is_rated_standard_rejects_variants() -> None:
    """The session-0002 defect: a rated *variant* game is not a rated standard game."""
    games = list(iter_games(MESSY))
    by_site = {g.headers["Site"]: g for g in games}
    assert is_rated_standard(by_site["https://lichess.org/good0001"])
    assert not is_rated_standard(
        by_site["https://lichess.org/variant04"]
    ), "Chess960 is not standard"
    assert not is_rated_standard(by_site["https://lichess.org/casual05"]), "casual is not rated"


def test_game_elo_bucket() -> None:
    games = list(iter_games(MINI))
    assert game_elo_bucket(games[0]) in {"1600-2000", "2000-2400"}


def test_game_filter_reports_a_reason() -> None:
    games = list(iter_games(MESSY))
    by_site = {g.headers["Site"]: g for g in games}
    f = GameFilter(min_elo=1600)
    accepted, reason = f.accepts(by_site["https://lichess.org/good0001"])
    assert not accepted and reason == "elo_below_min"
    accepted, reason = f.accepts(by_site["https://lichess.org/variant04"])
    assert not accepted and reason == "not_rated_standard"


def test_game_filter_time_control_policy() -> None:
    games = list(iter_games(MESSY))
    by_site = {g.headers["Site"]: g for g in games}
    f = GameFilter(time_controls=frozenset({"blitz"}))
    accepted, _ = f.accepts(by_site["https://lichess.org/good0001"])
    assert accepted, "180+2 is blitz"
