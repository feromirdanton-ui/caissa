"""Realism fixtures — the session-0005/0006 lessons, pinned as tests (ADR-0011 fast lane).

Each test here encodes a defect that escaped the local suite because the fixtures were too
clean. If any of these fails, a regression that would only have shown up on the runner was
caught locally instead — which is the entire purpose of the fast lane.
"""

from __future__ import annotations

from pathlib import Path

import chess.pgn
import pytest
import zstandard as zstd

from caissa.processing.pgn import (
    ParseStats,
    PgnParseError,
    TruncatedStreamError,
    iter_games,
    iter_games_zst,
)
from caissa.testing.realism import (
    RealismPaths,
    build_broken_pgn,
    build_valid_pgn,
    to_multiframe_zst,
    truncate_at_frame_boundary,
    truncate_mid_frame,
    write_realism_fixtures,
)


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> RealismPaths:
    return write_realism_fixtures(tmp_path_factory.mktemp("realism"), games=6, seed=3)


# --- lesson 1: a multi-frame stream (session-0006) -------------------------------------------


def _decompress_all_frames(data: bytes) -> bytes:
    """Decompress EVERY concatenated frame, the way the streaming parser must."""
    out = bytearray()
    pos = 0
    while pos < len(data):
        obj = zstd.ZstdDecompressor().decompressobj()
        out += obj.decompress(data[pos:])
        consumed = len(data) - pos - len(obj.unused_data)
        if consumed <= 0:
            break
        pos += consumed
    return bytes(out)


def test_multiframe_roundtrip_is_lossless() -> None:
    """Walking every frame must reproduce the payload exactly."""
    payload = build_valid_pgn(6, seed=1)
    multi = to_multiframe_zst(payload, frames=3)
    assert _decompress_all_frames(multi) == payload


def test_naive_single_frame_decompress_is_the_trap() -> None:
    """The trap itself: a plain ``decompress()`` reads ONLY the first frame.

    This is precisely what made 73 tests green while the runner was red (session-0006). Pinning
    it here means the trap is documented and cannot silently return.
    """
    payload = build_valid_pgn(6, seed=1)
    multi = to_multiframe_zst(payload, frames=3)
    naive = zstd.ZstdDecompressor().decompress(multi, max_output_size=1 << 24)
    assert naive != payload, "a naive decompress must NOT return the whole payload"
    assert payload.startswith(naive), "it returns exactly the first frame's bytes"
    assert len(naive) < len(payload)


def test_multiframe_is_actually_multiple_frames() -> None:
    """The fixture must really have >1 frame, or it would not test the defect."""
    payload = build_valid_pgn(6, seed=1)
    multi = to_multiframe_zst(payload, frames=3)
    single = zstd.ZstdCompressor().compress(payload)
    frames = 0
    pos = 0
    while pos < len(multi):
        obj = zstd.ZstdDecompressor().decompressobj()
        obj.decompress(multi[pos:])
        consumed = len(multi) - pos - len(obj.unused_data)
        if consumed <= 0:
            break
        frames += 1
        pos += consumed
    assert frames == 3, f"expected 3 frames, built {frames}"
    # A naive single-frame reader would stop after frame 1 and return only one third.
    assert len(single) != len(multi) or multi != single


def test_parser_reads_every_frame(fixtures: RealismPaths) -> None:
    """The parser must yield every game across all frames — the session-0006 defect."""
    stats = ParseStats()
    games = list(iter_games_zst(fixtures.valid_multi_frame, stats=stats))
    assert stats.truncated is False
    assert len(games) == 6, f"6 games were written, parser saw {len(games)}"
    # The control: a single-frame archive of the same payload must yield the same count.
    single_stats = ParseStats()
    single_games = list(iter_games_zst(fixtures.valid_single_frame, stats=single_stats))
    assert len(single_games) == len(games)


# --- lesson 2: truncation must be named, not hidden (session-0005) ----------------------------


def test_truncated_mid_frame_raises_by_default(fixtures: RealismPaths) -> None:
    """A byte prefix must raise rather than look like a short-but-complete archive."""
    with pytest.raises(TruncatedStreamError):
        list(iter_games_zst(fixtures.truncated_mid_frame))


def test_truncated_mid_frame_is_tolerated_and_named(fixtures: RealismPaths) -> None:
    stats = ParseStats()
    games = list(iter_games_zst(fixtures.truncated_mid_frame, stats=stats, truncated_ok=True))
    assert stats.truncated is True, "the truncation must be recorded, not silently ignored"
    assert len(games) < 6, "a prefix cannot contain all the games"


def test_truncation_at_a_frame_boundary_is_not_flagged(fixtures: RealismPaths) -> None:
    """The control: a stream that ends exactly on a frame boundary is complete, not truncated.

    Without this distinction the parser would flag every short archive as truncated — a false
    alarm that would make the honest signal useless.
    """
    stats = ParseStats()
    games = list(iter_games_zst(fixtures.truncated_at_boundary, stats=stats))
    assert stats.truncated is False, "a frame-boundary cut is a complete stream"
    assert len(games) > 0, "the first frame carries games"


def test_truncate_helpers_are_accurate() -> None:
    payload = build_valid_pgn(6, seed=2)
    multi = to_multiframe_zst(payload, frames=3)
    mid = truncate_mid_frame(multi, 0.5)
    assert 0 < len(mid) < len(multi)
    assert mid == multi[: len(multi) // 2]
    boundary = truncate_at_frame_boundary(multi, keep_frames=1)
    assert 0 < len(boundary) < len(multi)
    assert zstd.ZstdDecompressor().decompress(
        boundary, max_output_size=1 << 24
    ), "a boundary cut is still a valid archive"


# --- lesson 3: one corrupt game among valid ones ---------------------------------------------


def test_broken_fixture_has_exactly_one_bad_game(tmp_path: Path) -> None:
    """The generator's contract is 'exactly one bad game' so the count is assertable."""
    path = tmp_path / "broken.pgn"
    path.write_bytes(build_broken_pgn(6, seed=3, broken_index=1))

    strict_stats = ParseStats()
    with pytest.raises(PgnParseError):
        list(iter_games(path, strict=True, stats=strict_stats))

    lenient_stats = ParseStats()
    games = list(iter_games(path, strict=False, stats=lenient_stats))
    assert (
        lenient_stats.skipped == 1
    ), f"expected exactly 1 skipped game, got {lenient_stats.skipped}"
    assert len(games) == 5
    assert lenient_stats.errors, "the reason for the skip must be recorded"


def test_broken_fixture_written_to_disk_matches_builder(fixtures: RealismPaths) -> None:
    assert fixtures.broken_pgn.read_bytes() == build_broken_pgn(6, seed=3)
    # and it is real PGN: the valid games parse one by one
    with open(fixtures.broken_pgn, encoding="utf-8") as fh:
        parsed, errors = 0, 0
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            if game.errors:
                errors += 1
            else:
                parsed += 1
    assert parsed == 5 and errors == 1


# --- determinism: a fixture built in CI must equal one built locally --------------------------


def test_fixtures_are_deterministic(tmp_path: Path) -> None:
    a = write_realism_fixtures(tmp_path / "a", games=4, seed=9)
    b = write_realism_fixtures(tmp_path / "b", games=4, seed=9)
    for name in (
        "valid_single_frame",
        "valid_multi_frame",
        "truncated_mid_frame",
        "truncated_at_boundary",
        "broken_pgn",
    ):
        assert (
            getattr(a, name).read_bytes() == getattr(b, name).read_bytes()
        ), f"{name} is not deterministic"


def test_different_seeds_differ(tmp_path: Path) -> None:
    a = write_realism_fixtures(tmp_path / "a", games=4, seed=1)
    b = write_realism_fixtures(tmp_path / "b", games=4, seed=2)
    assert a.valid_single_frame.read_bytes() != b.valid_single_frame.read_bytes()
