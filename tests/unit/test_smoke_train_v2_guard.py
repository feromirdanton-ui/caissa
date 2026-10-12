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


def test_a_finite_blow_up_is_red_and_writes_a_failure(
    monkeypatch: pytest.MonkeyPatch, config: Path
) -> None:
    """MUTATION CONTROL, session-0013: a FINITE divergence must fail the run exactly as a NaN does.

    Session-0012's run went 3.35 -> 25.2 with every number finite and was reported `PLATEAUED`.
    This test forces that shape -- a validation curve that blows up while staying finite -- and
    asserts the run is RED and writes a FAILURE artifact, so a finite blow-up can never again be
    read as a result. The curve is injected at the loop's own `mean_loss_v2`, the exact seam the
    real run uses, so the guard is exercised on the real code path.
    """
    from caissa.training import smoke_train_v2_loop

    real_mean = smoke_train_v2_loop.mean_loss_v2
    calls = {"n": 0}

    def exploding_mean(model, samples, history):  # noqa: ANN001, ANN202 - test double
        calls["n"] += 1
        # First reading is the best; every later reading is 8x it -- a finite blow-up.
        return 3.3 if calls["n"] == 1 else 26.4

    monkeypatch.setattr(smoke_train_v2_loop, "mean_loss_v2", exploding_mean)
    assert real_mean is not exploding_mean

    metrics_path = Path(config.parent / "qa" / "metrics-v2.json")
    with pytest.raises(SystemExit) as excinfo:
        smoke_train_v2.run(str(config))

    assert excinfo.value.code == 3, "a finite divergence must exit non-zero (the run is RED)"
    assert not metrics_path.exists(), "metrics must NOT be written for a diverged run"
    failures = list((config.parent / "qa").glob("FAILURE-*.json"))
    assert failures, "a FAILURE artifact documenting the finite blow-up must be written"
    payload = json.loads(failures[0].read_text())
    assert payload["verdict"]["status"] == "INVALID"


def test_tolerant_mode_records_a_divergence_and_still_ends_red(
    monkeypatch: pytest.MonkeyPatch, config: Path
) -> None:
    """MUTATION CONTROL, session-0013 factorial: a divergence is RECORDED as DIVERGES, run still RED.

    The factorial's whole point is to learn which fixes converge, so one diverging configuration
    must not abort the others -- but the run must still end non-zero, so a factorial containing a
    blow-up can never be read as a clean result. Both halves are asserted here.
    """
    import yaml

    from caissa.training import smoke_train_v2_loop

    cfg = yaml.safe_load(config.read_text())
    cfg["tolerate_divergence"] = True
    cfg["v1_reference"] = False
    cfg["configurations"] = []
    tolerant = config.parent / "tolerant.yaml"
    tolerant.write_text(yaml.safe_dump(cfg))

    calls = {"n": 0}

    def exploding_mean(model, samples, history):  # noqa: ANN001, ANN202 - test double
        calls["n"] += 1
        return 3.3 if calls["n"] == 1 else 26.4

    monkeypatch.setattr(smoke_train_v2_loop, "mean_loss_v2", exploding_mean)
    with pytest.raises(SystemExit) as excinfo:
        smoke_train_v2.run(str(tolerant))
    assert excinfo.value.code == 3, "a factorial containing a divergence must still end RED"

    metrics = json.loads((config.parent / "qa" / "metrics-v2.json").read_text())
    assert metrics["convergence_summary"][0]["status"] == "DIVERGES"
    assert metrics["duel_verdict"]["status"] == "CONVERGENCE_ONLY"


def test_an_UNEXPECTED_divergence_still_reddens_the_factorial(
    monkeypatch: pytest.MonkeyPatch, config: Path
) -> None:
    """MUTATION CONTROL: the pre-registered control may diverge; any OTHER cell must not.

    The F0 control is expected to diverge, so it is exempt -- but the exemption must not leak. This
    test makes the SAME curve diverge while declaring nothing expected, and asserts the run still
    ends RED. Without it, "tolerate a divergence" could quietly become "accept any divergence".
    """
    import yaml

    from caissa.training import smoke_train_v2_loop

    cfg = yaml.safe_load(config.read_text())
    cfg["tolerate_divergence"] = True
    cfg["v1_reference"] = False
    cfg["configurations"] = []
    cfg["expected_divergence"] = []
    path = config.parent / "unexpected.yaml"
    path.write_text(yaml.safe_dump(cfg))

    calls = {"n": 0}

    def exploding_mean(model, samples, history):  # noqa: ANN001, ANN202 - test double
        calls["n"] += 1
        return 3.3 if calls["n"] == 1 else 26.4

    monkeypatch.setattr(smoke_train_v2_loop, "mean_loss_v2", exploding_mean)
    with pytest.raises(SystemExit) as excinfo:
        smoke_train_v2.run(str(path))
    assert excinfo.value.code == 3
