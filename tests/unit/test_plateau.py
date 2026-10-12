"""The plateau budget — equal steps judge the stopwatch, not the players (session-0011).

The instrument change this session makes is small to state and easy to get subtly wrong: a
configuration trains until its validation loss stops improving. If the criterion is too eager it
stops before convergence and understates the model; if it is too patient, or if float jitter counts
as improvement, it runs to the ceiling and the "plateau" is really just a stopwatch again.

Every branch is pinned below on synthetic curves whose expected outcome is obvious, so a run's
`plateau_reached` flag means what the pre-registration says it means.
"""

from __future__ import annotations

import pytest

from caissa.training.smoke_metrics import DivergenceError, PlateauTracker


def test_a_monotonically_falling_curve_never_plateaus() -> None:
    """While val loss keeps improving, training must continue -- the budget is a limit, not a target."""
    tracker = PlateauTracker(patience=3, eps=1e-4, min_evals=2)
    for value in (3.0, 2.5, 2.0, 1.5, 1.0, 0.5):
        assert tracker.update(value) is False
    assert tracker.reached is False
    assert tracker.best == pytest.approx(0.5)
    assert tracker.best_eval == 6


def test_a_flat_curve_plateaus_after_exactly_patience_evaluations() -> None:
    """Five evaluations without improvement, with patience 5, fires on the fifth -- not the fourth."""
    tracker = PlateauTracker(patience=5, eps=1e-4, min_evals=1)
    assert tracker.update(3.0) is False  # the best
    for i in range(4):
        assert tracker.update(3.0) is False, f"fired early at non-improvement {i + 1}"
    assert tracker.update(3.0) is True, "patience 5 must fire on the 5th non-improving evaluation"
    assert tracker.since_improvement == 5


def test_eps_defends_against_float_jitter() -> None:
    """A 'improvement' smaller than eps must NOT reset patience, or a flat curve would never stop."""
    tracker = PlateauTracker(patience=2, eps=1e-3, min_evals=1)
    tracker.update(3.0)
    # Two evaluations that each fall by less than eps: neither is an improvement.
    assert tracker.update(3.0 - 1e-6) is False
    assert tracker.update(3.0 - 1e-6) is True
    assert tracker.best == pytest.approx(3.0), "a sub-eps change must not become the new best"

    # ... while a change LARGER than eps does reset it.
    other = PlateauTracker(patience=2, eps=1e-3, min_evals=1)
    other.update(3.0)
    assert other.update(2.0) is False
    assert other.since_improvement == 0
    assert other.best == pytest.approx(2.0)


def test_min_evals_prevents_a_lucky_first_epoch_from_being_a_limit() -> None:
    """Patience cannot fire before min_evals, so epoch 1 is never mistaken for convergence.

    `min_evals` counts EVERY evaluation, including the loop's initial reading of the untrained
    model -- which is why the loop performs one update before its first epoch. With min_evals=5 and
    patience=1 the tracker must survive four evaluations and fire on the fifth.
    """
    tracker = PlateauTracker(patience=1, eps=1e-4, min_evals=5)
    assert tracker.update(3.0) is False  # evaluation 1 -- the best
    assert tracker.update(3.0) is False  # 2
    assert tracker.update(3.0) is False  # 3
    assert tracker.update(3.0) is False, "min_evals=5 must delay the first possible firing"
    assert tracker.update(3.0) is True  # 5
    assert tracker.evals == 5


def test_a_late_improvement_resets_patience() -> None:
    """A curve that plateaus and then drops was not converged -- patience must reset, not accumulate."""
    tracker = PlateauTracker(patience=2, eps=1e-4, min_evals=1)
    tracker.update(3.0)
    tracker.update(3.0)  # 1 without improvement
    assert tracker.update(3.0) is True  # would fire here at patience 2
    fresh = PlateauTracker(patience=2, eps=1e-4, min_evals=1)
    fresh.update(3.0)
    fresh.update(3.0)
    assert fresh.update(1.0) is False, "a real improvement must reset patience"
    assert fresh.since_improvement == 0
    assert fresh.best_eval == 3


def test_a_non_finite_evaluation_is_a_loud_error_not_a_plateau() -> None:
    """NaN must raise: a destroyed model is INVALID, never 'converged at infinity'."""
    tracker = PlateauTracker(patience=1, eps=1e-4, min_evals=1)
    with pytest.raises(DivergenceError, match="non-finite"):
        tracker.update(float("nan"))
    with pytest.raises(DivergenceError):
        tracker.update(float("inf"))


def test_bad_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        PlateauTracker(patience=0)
    with pytest.raises(ValueError):
        PlateauTracker(min_evals=0)


def test_as_dict_carries_the_whole_budget() -> None:
    """The artifact must let a reader recompute why training stopped, without the code."""
    tracker = PlateauTracker(patience=2, eps=1e-3, min_evals=1)
    tracker.update(2.0)
    tracker.update(2.0)
    tracker.update(2.0)
    payload = tracker.as_dict()

    assert payload["patience"] == 2
    assert payload["eps"] == pytest.approx(1e-3)
    assert payload["evaluations"] == 3
    assert payload["best_val_loss"] == pytest.approx(2.0)
    assert payload["evals_without_improvement"] == 2
    assert payload["plateau_reached"] is True
