"""`smoke-train-v2` -- the scaled vertical, exercised **locally** on real fixtures.

The runner repeats this on REAL Lichess games; here the input is the realism fixture set, so the
test is deterministic and needs no network. It proves the *machinery*: shards -> history windows ->
split by game -> train -> metrics -> pre-registered verdict, plus the two ablations running from the
same split and reporting a delta each.

It does NOT claim chess strength, and the fixture's accuracy is meaningless as a result -- a small
fixture of two opening lines is separable, so top-1 saturates. The assertions are therefore about
properties that must hold (legality, artifacts, deltas, split integrity), never about a number that
only the real run can produce.

The divergence guard -- the new INVALID branch of session-0010 -- is tested in
`test_smoke_train_v2_guard.py`, which carries its own mutation control.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from caissa.testing.v2_support import make_config, make_shards
from caissa.training import smoke_train_v2
from caissa.training.dataset import build_history_samples, count_short_games, load_games


@pytest.fixture()
def shards(tmp_path: Path) -> Path:
    return make_shards(tmp_path)


@pytest.fixture()
def config(tmp_path: Path, shards: Path) -> Path:
    return make_config(tmp_path, shards)


def test_vertical_v2_runs_and_stays_legal(config: Path) -> None:
    """The whole v2 composition runs; sampling is 100% legal and the verdict is one of the four."""
    metrics = smoke_train_v2.run(str(config))

    assert metrics["schema"] == "caissa-smoke-train-v2-metrics/1"
    # The gate counts DISTINCT positions (the fixture repeats positions), and every one must be legal.
    assert metrics["legality"]["sampled"] >= 10
    assert (
        metrics["legality"]["rate"] == 1.0
    ), "Article 1: the mask must make illegal moves impossible"
    assert metrics["verdict"]["status"] in {"SUCCESS", "DEAD", "INCONCLUSIVE", "INVALID"}
    assert metrics["data"]["train_samples"] > 0
    assert metrics["data"]["val_samples"] > 0
    assert metrics["model"]["loss_decreasing"] is True


def test_both_artifacts_are_written_and_self_describing(config: Path) -> None:
    """The metrics and the ablations are two files on disk, both complete enough to publish."""
    smoke_train_v2.run(str(config))

    metrics = json.loads((config.parent / "qa" / "metrics-v2.json").read_text())
    ablations = json.loads((config.parent / "qa" / "ablations-v2.json").read_text())

    assert len(metrics["model"]["val_loss_curve"]) >= 2
    assert len(metrics["model"]["train_loss_curve"]) >= 1
    assert metrics["baseline"]["move"].count(":") == 1

    assert ablations["schema"] == "caissa-smoke-train-v2-ablations/1"
    labels = {a["label"] for a in ablations["ablations"]}
    assert labels == {"no_geometry_bias", "no_history"}
    for entry in ablations["ablations"]:
        # Every ablation must carry a real delta versus the main run -- a missing delta would make
        # the ablation unreadable, which is the whole point of running it.
        assert "delta_top1_vs_main" in entry
        assert entry["legality"]["rate"] == 1.0
        assert entry["config"]["history"] >= 0


def test_ablation_overrides_actually_change_the_configuration(config: Path) -> None:
    """The two ablations must differ from the main run in exactly the intended knob."""
    main = smoke_train_v2.run(str(config))
    ablations = json.loads((config.parent / "qa" / "ablations-v2.json").read_text())
    by_label = {a["label"]: a for a in ablations["ablations"]}

    assert by_label["no_geometry_bias"]["config"]["geometric_bias"] is False
    assert main["config"]["geometric_bias"] is True

    assert by_label["no_history"]["config"]["history"] == 0
    assert main["config"]["history"] == 4


def test_history_windows_do_not_cross_a_game_boundary(shards: Path) -> None:
    """Every window must come from ONE game: a leaked predecessor would be a different game's move."""
    games = load_games(shards, max_positions=10_000, min_moves=2)
    samples = build_history_samples(games, history=4)
    assert samples, "the fixture produced no history windows"
    # Every board in a window must be reachable by replaying the same game's moves; the cheapest
    # proxy is that the window's move stack grows by exactly one per slot.
    for sample in samples:
        depths = [b.ply() for b in sample.history]
        assert depths == sorted(depths), "history slots are not in chronological order"
        assert depths[-1] - depths[0] == len(depths) - 1, "the window skipped or repeated a ply"


def test_games_too_short_for_a_window_are_counted_not_padded() -> None:
    """A game shorter than K+1 contributes nothing and is counted -- never padded with a fake board."""
    games = [
        [(chess.Board(), chess.Move.from_uci("e2e4"))],  # 1 position, cannot fill a K=4 window
        [(chess.Board(), chess.Move.from_uci("d2d4"))] * 10,  # long enough
    ]
    assert count_short_games(games, history=4) == 1
    samples = build_history_samples(games, history=4)
    assert all(len(s.history) == 5 for s in samples)


def test_missing_shards_are_a_loud_failure(tmp_path: Path) -> None:
    import yaml

    cfg = tmp_path / "empty.yaml"
    cfg.write_text(
        yaml.safe_dump({"shards": str(tmp_path / "nope"), "metrics": str(tmp_path / "m.json")})
    )
    with pytest.raises(SystemExit):
        smoke_train_v2.run(str(cfg))


def test_invalid_legality_fails_the_run(monkeypatch: pytest.MonkeyPatch, config: Path) -> None:
    """A legality leak makes the run INVALID and the CLI exits non-zero -- a defect, not a finding."""
    from caissa.model.numpy_history import NumpyHistoryModel

    def leaky(self, board, x_hist):  # noqa: ANN001, ANN202 - test double
        for frm in chess.SQUARES:
            for to in chess.SQUARES:
                candidate = chess.Move(frm, to)
                if candidate not in board.legal_moves:
                    return candidate
        raise AssertionError("no illegal move exists for this board")

    # The Article 1 gate plays the policy's top-1 move, so that is the seam to break.
    monkeypatch.setattr(NumpyHistoryModel, "argmax_move", leaky)
    assert smoke_train_v2.main(["--config", str(config)]) == 1


def test_eval_set_is_identical_across_main_and_ablations(config: Path) -> None:
    """Session-0009 preflight: the main run and BOTH ablations are scored on the SAME positions.

    The confounder this pins: `no_history` sets K=0, and if the window were rebuilt at K=0 the
    ablation would be evaluated on a LARGER set (ply >= 0) than the main run (ply >= 4) -- so a
    top-1 delta would mix the component's effect with a change of eval set. `window_history`
    keeps the main run's K for every configuration, so the counts must match exactly.
    """
    main = smoke_train_v2.run(str(config))
    ablations = json.loads((config.parent / "qa" / "ablations-v2.json").read_text())
    by_label = {a["label"]: a for a in ablations["ablations"]}

    for label in ("no_geometry_bias", "no_history"):
        assert (
            by_label[label]["data"]["val_samples"] == main["data"]["val_samples"]
        ), f"{label} was scored on a different eval set than the main run"
        assert by_label[label]["data"]["val_games"] == main["data"]["val_games"]

    # The K=0 ablation must still READ a single board (its input shrank) while the eval set did not.
    assert by_label["no_history"]["config"]["history"] == 0
    assert main["config"]["history"] == 4


def test_v1_reference_is_retrained_and_scored_on_both_sets(config: Path) -> None:
    """Session-0010 item 5: v1 is retrained in the same run and scored on BOTH eval sets.

    The v1->v2 comparison is only honest if both models are measured with the same ruler, so the
    artifact must carry `v1_at_v2_set` alongside `v1_at_v1_set`.
    """
    metrics = smoke_train_v2.run(str(config))
    v1 = metrics["v1_reference"]

    assert v1["label"] == "v1_reference"
    assert v1["model"] == "numpy_tiny"
    assert v1["v1_at_v1_set"]["val_positions"] > 0
    assert v1["v1_at_v2_set"]["val_positions"] == metrics["data"]["val_samples"]
