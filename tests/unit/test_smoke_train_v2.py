"""`smoke-train-v2` — the convergence duel, exercised **locally** on real fixtures.

The runner repeats this on REAL Lichess games; here the input is the realism fixture set, so the
test is deterministic and needs no network. It proves the *machinery*: shards -> history windows ->
split by game -> plateau-bounded train -> metrics -> the factorial of configurations -> a McNemar
comparison against the v1 reference for every delta.

It does NOT claim chess strength, and the fixture's accuracy is meaningless as a result -- a small
fixture of a few opening lines is separable, so top-1 saturates. The assertions are therefore about
properties that must hold (legality, artifacts, deltas, pairing, split integrity), never about a
number that only the real run can produce.

Session-0011 changed the shape: the two ablations became a **factorial** (`configurations`), the
budget became a **plateau** rather than an epoch count, and every delta is now a **McNemar**
comparison. The plateau and McNemar logic are unit-tested deterministically in `test_plateau.py` and
`test_mcnemar.py`; this file asserts the pipeline wires them together.

The divergence guard -- the INVALID branch of session-0010 -- is tested in
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


def _ablations(config: Path) -> dict:
    return json.loads((config.parent / "qa" / "ablations-v2.json").read_text())


def test_vertical_v2_runs_and_stays_legal(config: Path) -> None:
    """The whole v2 composition runs; sampling is 100% legal and the verdict is a known word."""
    metrics = smoke_train_v2.run(str(config))

    assert metrics["schema"] == "caissa-smoke-train-v2-metrics/1"
    # The gate counts DISTINCT positions (the fixture repeats positions), and every one must be legal.
    assert metrics["legality"]["sampled"] >= 10
    assert (
        metrics["legality"]["rate"] == 1.0
    ), "Article 1: the mask must make illegal moves impossible"
    assert metrics["verdict"]["status"] in {
        "PLATEAUED",
        "SUCCESS",
        "DEAD",
        "INCONCLUSIVE",
        "INVALID",
    }
    assert metrics["data"]["train_samples"] > 0
    assert metrics["data"]["val_samples"] > 0
    assert metrics["convergence"]["steps"] > 0
    assert metrics["model"]["loss_decreasing"] is True


def test_both_artifacts_are_written_and_self_describing(config: Path) -> None:
    """The metrics and the configurations are two files on disk, both complete enough to publish."""
    smoke_train_v2.run(str(config))

    metrics = json.loads((config.parent / "qa" / "metrics-v2.json").read_text())
    ablations = _ablations(config)

    assert len(metrics["model"]["val_loss_curve"]) >= 2
    assert len(metrics["model"]["train_loss_curve"]) >= 1
    assert metrics["baseline"]["move"].count(":") == 1
    assert metrics["baseline"]["ruler"] == "set_local"

    assert ablations["schema"] == "caissa-smoke-train-v2-ablations/1"
    labels = {a["label"] for a in ablations["ablations"]}
    assert labels == {"lower_capacity", "no_history", "lower_capacity_no_history"}
    for entry in ablations["ablations"]:
        # Every configuration must carry a real delta AND its significance test versus v1 -- a
        # missing delta would make the comparison unreadable, which is the point of running it.
        assert "top1_delta_vs_v1" in entry
        assert {
            "z",
            "p_value",
            "method",
            "favours",
            "significant",
            "n",
            "only_a_right",
            "only_b_right",
        } <= set(entry["mcnemar_vs_v1"])
        assert entry["legality"]["rate"] == 1.0
        assert entry["convergence"]["steps"] > 0
        assert entry["config"]["history"] >= 0
    assert ablations["duel_verdict"]["reason"], "a verdict without a reason is not a verdict"


def test_configuration_overrides_change_exactly_the_intended_knob(config: Path) -> None:
    """Each configuration must differ from the main run in exactly the intended knob (the 2x2)."""
    main = smoke_train_v2.run(str(config))
    by_label = {a["label"]: a for a in _ablations(config)["ablations"]}

    assert main["config"]["d_model"] == 8
    assert main["config"]["history"] == 2

    # capacity axis at fixed history
    assert by_label["lower_capacity"]["config"]["d_model"] == 4
    assert by_label["lower_capacity"]["config"]["history"] == 2
    # history axis at fixed capacity
    assert by_label["no_history"]["config"]["history"] == 0
    assert by_label["no_history"]["config"]["d_model"] == 8
    # the cell that closes the factorial
    assert by_label["lower_capacity_no_history"]["config"]["d_model"] == 4
    assert by_label["lower_capacity_no_history"]["config"]["history"] == 0


def test_eval_set_is_identical_across_every_configuration(config: Path) -> None:
    """Session-0009 preflight: main and EVERY configuration are scored on the SAME positions.

    The confounder this pins: a configuration with K=0 would, if the window were rebuilt at K=0, be
    evaluated on a LARGER set than the main run -- so a top-1 delta would mix the component's effect
    with a change of eval set. `window_history` keeps the main run's K for every configuration, so
    the counts must match exactly, and a McNemar pairing stays valid.
    """
    main = smoke_train_v2.run(str(config))
    for entry in _ablations(config)["ablations"]:
        assert (
            entry["data"]["val_samples"] == main["data"]["val_samples"]
        ), f"{entry['label']} was scored on a different eval set than the main run"
        assert entry["data"]["val_games"] == main["data"]["val_games"]
        assert entry["mcnemar_vs_v1"]["n"] == main["data"]["val_samples"]


def test_v1_reference_is_retrained_and_scored_on_both_sets(config: Path) -> None:
    """The v1->v2 comparison is only honest if both models are measured with the same ruler."""
    metrics = smoke_train_v2.run(str(config))
    v1 = metrics["v1_reference"]

    assert v1["label"] == "v1_reference"
    assert v1["model"] == "numpy_tiny"
    assert (
        v1["optimizer"] == "sgd"
    ), "v1 keeps its own optimizer: lr is part of an architecture's price"
    assert v1["v1_at_v1_set"]["val_positions"] > 0
    assert v1["v1_at_v2_set"]["val_positions"] == metrics["data"]["val_samples"]


def test_plateau_budget_is_recorded_not_an_epoch_count(config: Path) -> None:
    """Session-0011: the budget is a plateau (patience/eps/ceiling), and the steps consumed are recorded."""
    metrics = smoke_train_v2.run(str(config))
    conv = metrics["convergence"]

    assert conv["patience"] == 1  # the fixture's tiny patience (see v2_support)
    assert conv["eps"] == pytest.approx(1e-4)
    assert conv["evaluations"] >= 1
    assert conv["epochs_run"] <= metrics["config"]["max_epochs"]
    # The triangle: plateau reached, or the ceiling fired first -- never neither silently.
    assert conv["plateau_reached"] or conv["epochs_run"] == metrics["config"]["max_epochs"]


def test_gradient_norm_is_logged_for_every_step(config: Path) -> None:
    """The fast-lane addition: a diverging run shows the norm before the loss ever reaches NaN."""
    metrics = smoke_train_v2.run(str(config))
    conv = metrics["convergence"]

    assert conv["grad_norm_samples"] == conv["steps"], "the Adam path must log a norm per step"
    assert conv["grad_norm_first"] is not None and conv["grad_norm_last"] is not None
    assert conv["grad_norm_max"] >= conv["grad_norm_last"]
    assert metrics["config"]["optimizer"] == "adam"


def test_history_windows_do_not_cross_a_game_boundary(shards: Path) -> None:
    """Every window must come from ONE game: a leaked predecessor would be a different game's move."""
    games = load_games(shards, max_positions=10_000, min_moves=2)
    samples = build_history_samples(games, history=2)
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
        [(chess.Board(), chess.Move.from_uci("e2e4"))],  # 1 position, cannot fill a K=2 window
        [(chess.Board(), chess.Move.from_uci("d2d4"))] * 10,  # long enough
    ]
    assert count_short_games(games, history=2) == 1
    samples = build_history_samples(games, history=2)
    assert all(len(s.history) == 3 for s in samples)


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


def test_the_shipped_config_matches_the_pre_registration() -> None:
    """The SHIPPED config must be the one the pre-registration promised, not merely a valid one.

    `LOG/session-0011-preregistration.md` section 5 fixes the experiment: add `d128@K8` on the
    capacity axis, and close `{128,256} x {K 0,8}` inside the v2 family, with geometry ON in all
    four cells and the v1 reference retained. A pre-registration that the config silently diverges
    from is worse than no pre-registration -- it is a promise the code did not keep. This test reads
    the real `configs/smoke-train-v2.yaml` and pins exactly that.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "configs" / "smoke-train-v2.yaml").read_text())

    # The main cell of the 2x2.
    assert cfg["history"] == 8
    assert cfg["window_history"] == 8, "the eval set must be built with the main run's K"
    assert cfg["model"]["d_model"] == 256
    assert cfg["model"]["n_heads"] == 8
    assert cfg["model"]["geometric_bias"] is True

    # The budget is a plateau, not a step count.
    assert cfg["plateau_patience"] == 5
    assert cfg["plateau_eps"] == pytest.approx(1e-4)
    assert cfg["max_epochs"] == 20
    assert cfg["budget_minutes"] <= 90

    # One recorded optimizer choice.
    assert cfg["optimizer"] == "adam"
    assert cfg["lr"] == pytest.approx(0.005)
    assert cfg["v1_reference"] is True

    # The eval set is the session-0011 10k.
    assert cfg["max_val_positions"] == 10_000
    assert cfg["progress_marker"] == pytest.approx(0.30)

    # The frozen factorial: {128,256} x {K 0,8}, geometry ON in every cell.
    by_name = {c["name"]: c for c in cfg["configurations"]}
    assert set(by_name) == {"d128_k8", "d256_k0", "d128_k0"}
    assert by_name["d128_k8"]["overrides"]["model"]["d_model"] == 128  # the capacity axis at K=8
    assert by_name["d256_k0"]["overrides"]["history"] == 0  # the history axis at d=256
    assert by_name["d256_k0"]["overrides"]["window_history"] == 8  # the eval set does NOT shrink
    assert by_name["d128_k0"]["overrides"]["window_history"] == 8
    for spec in cfg["configurations"]:
        assert spec["overrides"].get("model", {}).get("geometric_bias", True) is True
