"""Unit tests for dedup, sharding + QA, and blunder labeling."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import zstandard as zstd

from caissa.processing.dedup import Deduplicator, game_id
from caissa.processing.labeling import BlunderLabel, classify_loss, label_games, label_move
from caissa.processing.pgn import iter_games
from caissa.processing.sharding import shard_games

FIXTURES = Path(__file__).parent.parent / "fixtures"
MINI = FIXTURES / "mini.pgn"
MESSY = FIXTURES / "messy.pgn"


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


def test_cli_dispatch_routes_label_subcommand(tmp_path: Path) -> None:
    """`caissa-label` must route to the labeler, not the sharder.

    The first processing-sample run on the runner exposed this: run.py passed `--label`
    as a flag, the dispatcher expected the `label` subcommand, and the labeler never ran
    while `|| true` hid the failure. This test pins the routing.
    """
    from caissa.processing.cli import main

    out = tmp_path / "labels.json"
    rc = main(["label", str(MINI), "--limit", "20", "--engine", "mock", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["mock"] is True
    assert payload["positions"] > 0


def test_cli_dispatch_defaults_to_shard(tmp_path: Path) -> None:
    from caissa.processing.cli import main

    out = tmp_path / "shards"
    qa = tmp_path / "qa.json"
    rc = main([str(MINI), "--out", str(out), "--qa", str(qa), "--shard-size", "10"])
    assert rc == 0
    assert json.loads(qa.read_text())["kept"] == 2
