"""`smoke-train` — the deep-lane experiment, exercised **locally** on real fixtures.

This is the local proof that the vertical holds together (ADR-0011: the fast lane runs every
session). The runner repeats the same experiment on REAL Lichess games; here the input is the
realism fixture set, so the test is deterministic and needs no network.

What it proves, and what it does not:

- It proves the composition: shards → positions → split by game → train → metrics → verdict, and
  that a real tiny model learns the fixture's move distribution and samples 100% legally.
- It does NOT claim chess strength and does NOT claim the real-data verdict. The real verdict
  belongs to the runner run, on real games.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
import yaml

from caissa.processing.pgn import iter_games
from caissa.processing.sharding import shard_games
from caissa.testing.realism import build_valid_pgn
from caissa.training import smoke_train


@pytest.fixture()
def shards_from_pgn(tmp_path: Path) -> Path:
    """Build real jsonl.zst shards from a realistic multi-game PGN payload."""
    pgn = tmp_path / "sample.pgn"
    pgn.write_bytes(build_valid_pgn(40, seed=11))
    out = tmp_path / "shards"
    report = shard_games(iter_games(pgn), out, shard_size=10)
    assert report.kept == 40, "the fixture must survive the filter/dedup path"
    assert not list(out.glob("*.part"))
    return out


@pytest.fixture()
def config(tmp_path: Path, shards_from_pgn: Path) -> Path:
    cfg = {
        "shards": str(shards_from_pgn),
        "metrics": str(tmp_path / "qa" / "metrics.json"),
        "val_fraction": 0.05,
        "seed": 0,
        "max_train_positions": 600,
        "max_val_positions": 200,
        "epochs": 3,
        "lr": 1.0,
        "sample_moves": 100,
        "budget_minutes": 60,
    }
    path = tmp_path / "smoke-train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_vertical_learns_and_stays_legal(config: Path) -> None:
    """The whole composition runs, the model beats the straw man, and sampling is legal."""
    metrics = smoke_train.run(str(config))

    assert metrics["schema"] == "caissa-smoke-train-metrics/1"
    assert metrics["legality"]["sampled"] == 100
    assert (
        metrics["legality"]["rate"] == 1.0
    ), "Article 1: the mask must make illegal sampling impossible"

    # The pre-registered criterion, evaluated locally on fixtures.
    assert metrics["model"]["val_loss_last"] < metrics["model"]["val_loss_first"]
    assert metrics["model"]["top1_accuracy"] > metrics["baseline"]["val_accuracy"]
    assert metrics["verdict"]["status"] == "ALIVE"
    assert metrics["verdict"]["within_budget"] is True


def test_metrics_artifact_is_written_and_self_describing(config: Path) -> None:
    metrics = smoke_train.run(str(config))
    written = json.loads(Path(metrics["metrics"]).read_text()) if "metrics" in metrics else None
    on_disk = json.loads((config.parent / "qa" / "metrics.json").read_text())
    assert written is None or written == on_disk
    assert on_disk["data"]["train_games"] > 0
    assert on_disk["data"]["val_games"] > 0
    # Loss curves, both of them, so a reader can see the shape and not just the endpoints.
    assert len(on_disk["model"]["val_loss_curve"]) >= 2
    assert len(on_disk["model"]["train_loss_curve"]) >= 1
    assert on_disk["baseline"]["move"].count(":") == 1


def test_split_is_by_game_not_by_position(shards_from_pgn: Path) -> None:
    """No game may appear on both sides — a leaked predecessor would inflate val accuracy."""
    from caissa.training.dataset import load_games, split_games

    games = load_games(shards_from_pgn, max_positions=10_000, min_moves=2)
    train, val = split_games(games, val_fraction=0.2, seed=0)
    train_ids = {id(g) for g in train}
    val_ids = {id(g) for g in val}
    assert not (train_ids & val_ids)
    assert len(train) + len(val) == len(games)


def test_missing_shards_are_a_loud_failure(tmp_path: Path) -> None:
    """An empty shard directory must stop the run, never produce a metrics file of zeros."""
    cfg = tmp_path / "empty.yaml"
    cfg.write_text(
        yaml.safe_dump({"shards": str(tmp_path / "nope"), "metrics": str(tmp_path / "m.json")})
    )
    with pytest.raises(SystemExit):
        smoke_train.run(str(cfg))


def test_invalid_legality_fails_the_run(monkeypatch: pytest.MonkeyPatch, config: Path) -> None:
    """A legality leak makes the run INVALID and the CLI exits non-zero — a defect, not a finding."""
    from caissa.model.numpy_tiny import NumpyTinyModel

    def leaky(self, board, rng):  # noqa: ANN001, ANN202 - test double
        # A *guaranteed* illegal move: the first (from, to) pair the board rejects. Using a
        # pseudo-legal move would not do — at most positions a pseudo-legal move IS legal, so
        # the double would silently keep the rate at 100%.
        for frm in chess.SQUARES:
            for to in chess.SQUARES:
                candidate = chess.Move(frm, to)
                if candidate not in board.legal_moves:
                    return candidate
        raise AssertionError("no illegal move exists for this board")

    monkeypatch.setattr(NumpyTinyModel, "sample_move", leaky)
    # `main` returns non-zero (it does not raise) so CI fails the run on an INVALID verdict.
    assert smoke_train.main(["--config", str(config)]) == 1
