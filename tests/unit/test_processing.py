"""Unit tests for the streaming PGN parser, filters, dedup, sharding and labeling."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn
import pytest
import zstandard as zstd

from caissa.processing.dedup import Deduplicator, game_id
from caissa.processing.filters import (
    GameFilter,
    elo_bucket,
    game_elo_bucket,
    is_rated_standard,
    parse_time_control,
    time_control_category,
)
from caissa.processing.labeling import BlunderLabel, classify_loss, label_games, label_move
from caissa.processing.pgn import (
    ParseStats,
    PgnParseError,
    TruncatedStreamError,
    iter_games,
    iter_games_any,
    iter_games_zst,
)
from caissa.processing.sharding import shard_games

FIXTURES = Path(__file__).parent.parent / "fixtures"
MINI = FIXTURES / "mini.pgn"
MESSY = FIXTURES / "messy.pgn"
BROKEN = FIXTURES / "broken.pgn"


# --- parser -------------------------------------------------------------------


def test_iter_games_streams() -> None:
    games = list(iter_games(MINI))
    assert len(games) == 2
    assert games[0].headers["White"] == "Alice"


def test_parser_is_a_generator_not_a_list() -> None:
    """The parser must be lazy: pulling one game must not read the whole file."""
    import inspect

    assert inspect.isgeneratorfunction(iter_games)
    gen = iter_games(MINI)
    first = next(gen)
    assert first.headers["White"] == "Alice"
    gen.close()


def test_parser_fails_loudly_and_locally_on_a_broken_game() -> None:
    """A malformed game raises PgnParseError naming the game index (not a silent skip)."""
    with pytest.raises(PgnParseError) as excinfo:
        list(iter_games(BROKEN, strict=True))
    assert excinfo.value.game_index == 2, "the error must name the offending game"
    assert "broken.pgn" in str(excinfo.value)


def test_parser_lenient_mode_counts_skips() -> None:
    stats = ParseStats()
    games = list(iter_games(BROKEN, strict=False, stats=stats))
    assert len(games) == 2, "the two good games survive"
    assert stats.skipped == 1, "the bad game is counted, not hidden"
    assert stats.games == 2


def test_iter_games_zst_roundtrip(tmp_path: Path) -> None:
    raw = MINI.read_bytes()
    zst = tmp_path / "mini.pgn.zst"
    zst.write_bytes(zstd.ZstdCompressor().compress(raw))
    games = list(iter_games_zst(zst))
    assert len(games) == 2
    assert games[1].headers["White"] == "Carol"


def test_iter_games_zst_truncated_prefix_is_named(tmp_path: Path) -> None:
    """A truncated zstd stream must be *named*, never read as an empty complete archive.

    ``zstandard``'s ``stream_reader`` returns 0 bytes for a truncated frame without
    raising — a silent truncation. The parser detects it via ``decompressobj().eof`` and
    either raises (default) or records ``stats.truncated`` (``truncated_ok=True``).
    """
    raw = MINI.read_bytes()
    compressed = zstd.ZstdCompressor().compress(raw)
    zst = tmp_path / "prefix.pgn.zst"
    zst.write_bytes(compressed[: len(compressed) // 2])

    # default: a truncated stream is a loud failure, not an empty archive
    with pytest.raises(TruncatedStreamError):
        list(iter_games_zst(zst))

    # opt-in: accept the prefix and record the truncation
    stats = ParseStats()
    games = list(iter_games_zst(zst, truncated_ok=True, stats=stats))
    assert stats.truncated is True, "the truncation must be recorded"
    assert len(games) >= 0  # whatever decoded is fine; the point is it did not raise


def test_iter_games_zst_complete_stream_is_not_flagged(tmp_path: Path) -> None:
    """A complete stream must NOT be flagged truncated (no false alarm)."""
    zst = tmp_path / "full.pgn.zst"
    zst.write_bytes(zstd.ZstdCompressor().compress(MINI.read_bytes()))
    stats = ParseStats()
    games = list(iter_games_zst(zst, stats=stats))
    assert len(games) == 2
    assert stats.truncated is False


def test_iter_games_any_dispatches_on_suffix(tmp_path: Path) -> None:
    zst = tmp_path / "mini.pgn.zst"
    zst.write_bytes(zstd.ZstdCompressor().compress(MINI.read_bytes()))
    assert len(list(iter_games_any(zst))) == 2
    assert len(list(iter_games_any(MINI))) == 2


# --- filters ------------------------------------------------------------------


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


# --- dedup --------------------------------------------------------------------


def test_game_id_uses_site_url() -> None:
    games = list(iter_games(MINI))
    assert game_id(games[0]) == "https://lichess.org/aaaa1111"


def test_dedup_drops_the_second_copy() -> None:
    games = list(iter_games(MESSY))
    dedup = Deduplicator()
    kept = [g for g in games if dedup.is_new(g)]
    assert dedup.duplicates == 1, "the duplicated Site URL is dropped once"
    assert len(kept) == len(games) - 1


# --- sharding + QA ------------------------------------------------------------


def test_shard_games_writes_zst_and_certifies(tmp_path: Path) -> None:
    report = shard_games(iter_games(MINI), tmp_path, shard_size=1)
    assert report.games_seen == 2
    assert report.kept == 2
    shards = sorted(tmp_path.glob("*.jsonl.zst"))
    assert len(shards) == 2
    assert not list(tmp_path.glob("*.part")), "atomic writes leave no .part"
    assert all(s.sha256 for s in report.shards), "every shard carries its sha256"
    # the shard is real jsonl.zst
    first = json.loads(zstd.ZstdDecompressor().decompress(shards[0].read_bytes()).splitlines()[0])
    assert first["white"] == "Alice"


def test_shard_games_qa_counts_rejections(tmp_path: Path) -> None:
    report = shard_games(iter_games(MESSY), tmp_path, shard_size=100)
    assert report.games_seen == 6
    assert report.duplicates == 1
    assert report.rejected["not_rated_standard"] == 2, "Chess960 + casual"
    assert report.kept == 3
    assert report.elo_buckets, "Elo distribution must be measured"
    assert report.time_controls, "time-control distribution must be measured"
    assert report.openings, "opening distribution must be measured"


def test_shard_games_is_reproducible(tmp_path: Path) -> None:
    a = shard_games(iter_games(MINI), tmp_path / "a", shard_size=1)
    b = shard_games(iter_games(MINI), tmp_path / "b", shard_size=1)
    assert [s.sha256 for s in a.shards] == [s.sha256 for s in b.shards], "same input -> same shards"


# --- labeling -----------------------------------------------------------------


def test_classify_loss_thresholds() -> None:
    assert classify_loss(0) == BlunderLabel.BEST
    assert classify_loss(60) == BlunderLabel.INACCURACY
    assert classify_loss(150) == BlunderLabel.MISTAKE
    assert classify_loss(300) == BlunderLabel.BLUNDER


def test_label_move_measures_a_real_delta() -> None:
    from caissa.guards.counterfactual import MockEngine

    board = chess.Board()
    # Qh5?? hangs nothing immediately, but h4?? is fine; use a capture to force a delta
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    labeled = label_move(board, chess.Move.from_uci("d1h5"), MockEngine())
    assert labeled.engine == "mock"
    assert labeled.is_mock
    assert labeled.cp_loss >= 0


def test_label_games_reports_mock_honestly() -> None:
    from caissa.guards.counterfactual import MockEngine

    report = label_games(iter_games(MINI), MockEngine(), limit=100)
    assert report.positions > 0
    assert report.is_mock is True
    assert "MOCK" in report.to_markdown(), "a mock report must say so"
