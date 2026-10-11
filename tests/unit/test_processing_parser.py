"""Unit tests for the streaming PGN parser."""

from __future__ import annotations

from pathlib import Path

import pytest
import zstandard as zstd

from caissa.processing.pgn import (
    ParseStats,
    PgnParseError,
    TruncatedStreamError,
    iter_games,
    iter_games_any,
    iter_games_zst,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
MINI = FIXTURES / "mini.pgn"
BROKEN = FIXTURES / "broken.pgn"


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


def test_iter_games_zst_multi_frame_archive(tmp_path: Path) -> None:
    """A Lichess archive is a CONCATENATION of zstd frames — all of them must be read.

    ``decompressobj`` stops at the end of the first frame and raises if fed more; the
    parser rolls over to a fresh object using ``unused_data``. This is the defect the
    first processing-sample run found on the runner (a real Lichess day-prefix).
    """
    raw = MINI.read_bytes()
    half = len(raw) // 2
    compressor = zstd.ZstdCompressor()
    # two independent frames, concatenated — exactly how Lichess ships its archives
    zst = tmp_path / "multi.pgn.zst"
    zst.write_bytes(compressor.compress(raw[:half]) + compressor.compress(raw[half:]))
    stats = ParseStats()
    games = list(iter_games_zst(zst, stats=stats))
    assert len(games) == 2, "both frames must be decompressed, not just the first"
    assert stats.truncated is False


def test_iter_games_any_dispatches_on_suffix(tmp_path: Path) -> None:
    zst = tmp_path / "mini.pgn.zst"
    zst.write_bytes(zstd.ZstdCompressor().compress(MINI.read_bytes()))
    assert len(list(iter_games_any(zst))) == 2
    assert len(list(iter_games_any(MINI))) == 2
