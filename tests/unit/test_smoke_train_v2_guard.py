"""The divergence guard -- the new INVALID branch of session-0010, with its mutation control.

Session-0009 produced a GREEN run whose loss curves were `[NaN, NaN]` for the main configuration
*and* both ablations, all three reporting an identical top-1, while the run finished green with an
INCONCLUSIVE verdict. The number was not a measurement -- the model was destroyed and the pipeline
could not tell. The guard exists so that state is a loud RED failure, and this file proves it by
*deliberately creating* the state and asserting the run refuses it.

Split from `test_smoke_train_v2.py` so each published file stays small enough to move over the
GitHub API without a transcription slip (ADR-0009).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from caissa.testing.v2_support import make_config, make_shards, poison
from caissa.training import smoke_train_v2
from caissa.training.smoke_metrics import DivergenceError


@pytest.fixture()
def shards(tmp_path: Path) -> Path:
    return make_shards(tmp_path)


@pytest.fixture()
def config(tmp_path: Path, shards: Path) -> Path:
    return make_config(tmp_path, shards)


def test_parameters_finite_sees_a_poisoned_model() -> None:
    """MUTATION CONTROL, unit half: poisoning the weights makes `parameters_finite` return False."""
    from caissa.model.numpy_history import NumpyHistoryModel
    from caissa.training.smoke_metrics import parameters_finite

    model = NumpyHistoryModel(history=2, d_model=8, n_heads=2, geometry_bias=True, seed=0)

    # The healthy model is finite -- so the failure below is caused by the mutation, not the fixture.
    assert parameters_finite(model) is True

    poison(model)
    assert parameters_finite(model) is False


def test_diverged_run_is_red_and_writes_a_failure_not_metrics(
    monkeypatch: pytest.MonkeyPatch, config: Path
) -> None:
    """MUTATION CONTROL, end-to-end: a poisoned model makes the RUN fail and writes FAILURE, no metrics.

    Asserts (a) the run aborts with a non-zero exit, (b) a `FAILURE-*.json` artifact carries the
    cause and the INVALID verdict, and (c) the metrics artifact is NOT written -- because a green
    artifact with NaN numbers is the exact thing the guard exists to make impossible.
    """
    real_build = smoke_train_v2._build_model

    def poisoned_build(cfg: dict):  # noqa: ANN202 - test double
        model = real_build(cfg)
        poison(model)
        return model

    monkeypatch.setattr(smoke_train_v2, "_build_model", poisoned_build)

    metrics_path = Path(config.parent / "qa" / "metrics-v2.json")
    with pytest.raises(SystemExit) as excinfo:
        smoke_train_v2.run(str(config))

    assert excinfo.value.code == 3, "a divergence must exit non-zero (the run is RED)"
    assert not metrics_path.exists(), "metrics must NOT be written for a diverged run"
    failures = list((config.parent / "qa").glob("FAILURE-*.json"))
    assert failures, "a FAILURE artifact documenting the cause must be written"
    payload = json.loads(failures[0].read_text())
    assert payload["schema"] == "caissa-smoke-train-v2-failure/1"
    assert payload["verdict"]["status"] == "INVALID"


def test_metrics_writer_refuses_a_non_finite_value() -> None:
    """The writer must refuse to persist a non-finite number -- no honest artifact can hold a NaN."""
    with pytest.raises(DivergenceError):
        smoke_train_v2._assert_finite({"model": {"top1_accuracy": float("nan")}})
    with pytest.raises(DivergenceError):
        smoke_train_v2._assert_finite({"curve": [1.0, float("inf")]})
    # A finite structure passes untouched.
    smoke_train_v2._assert_finite({"model": {"top1_accuracy": 0.31}, "curve": [1.0, 0.5]})


def test_the_guard_catches_a_single_non_finite_step() -> None:
    """A non-finite LOSS on its own must raise -- the weights check is a second net, not the only one."""
    from caissa.training.smoke_metrics import assert_finite

    with pytest.raises(DivergenceError):
        assert_finite({"val_loss_curve": [3.4, float("nan")]})
