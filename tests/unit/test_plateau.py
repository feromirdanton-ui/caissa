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

from caissa.training.smoke_metrics import (
    DIVERGENCE_FACTOR,
    DivergenceError,
    PlateauTracker,
    classify_curve,
)

# The two REAL validation curves of session-0012, copied verbatim from the metrics artifact
# (`smoke-train-v2-metrics`, id 11748652041). They are the mutation control for the DIVERGED status:
# the detector must call the v2 curve a divergence and the v1 curve a plateau, on the actual numbers
# a run produced -- not on a synthetic curve chosen to make the test pass.
V2_CURVE_0012 = [
    3.3453018355435247,
    25.069838315988566,
    25.0644726504099,
    24.91592039166883,
    24.939187044644306,
    25.184626799891568,
]
V1_CURVE_0012 = [
    3.1662784919140865,
    3.2053693916509127,
    3.295548506390203,
    3.4131413501273316,
    3.4544932724192745,
    3.5407863825670587,
]


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


# --- session-0013: a FINITE blow-up is a divergence, not a plateau ------------------------------


def test_the_real_v2_curve_of_0012_is_a_divergence() -> None:
    """MUTATION CONTROL (v2 half): the actual 0012 v2 curve must classify as DIVERGED.

    This is the curve that session-0012 reported as `PLATEAUED` while it went 3.35 -> 25.2. If the
    detector ever stops calling it a divergence, this test goes red -- which is the point.
    """
    assert classify_curve(V2_CURVE_0012) == "DIVERGED"
    tracker = PlateauTracker(patience=5, eps=1e-4, min_evals=6)
    for value in V2_CURVE_0012:
        tracker.update(value)
    assert tracker.diverged is True
    assert tracker.status == "DIVERGED"


def test_the_real_v1_curve_of_0012_is_NOT_a_divergence() -> None:
    """MUTATION CONTROL (v1 half): v1's own curve rises 3.17 -> 3.54 (1.12x) -- overfitting, not a blow-up.

    The comparison is against the BEST, not the first: v1's first reading is its best, and a rule
    that compared final-to-first would still pass here -- but a rule that compared final-to-best
    with too small a factor would wrongly flag v1. This pins the factor above 1.12x.
    """
    assert classify_curve(V1_CURVE_0012) == "PLATEAUED"
    tracker = PlateauTracker(patience=5, eps=1e-4, min_evals=6)
    for value in V1_CURVE_0012:
        tracker.update(value)
    assert tracker.diverged is False
    assert tracker.status == "PLATEAUED"


def test_the_divergence_factor_sits_between_the_two_real_curves() -> None:
    """The recorded factor must separate the two real curves: above v1's 1.12x, below v2's 7.53x."""
    v1_ratio = V1_CURVE_0012[-1] / min(V1_CURVE_0012)
    v2_ratio = V2_CURVE_0012[-1] / min(V2_CURVE_0012)
    assert v1_ratio < DIVERGENCE_FACTOR < v2_ratio
    assert v1_ratio == pytest.approx(1.118, abs=0.01)
    assert v2_ratio == pytest.approx(7.53, abs=0.05)


def test_a_curve_that_improves_then_blows_up_is_a_divergence() -> None:
    """A curve that reaches a good best and then explodes is DIVERGED -- the best is what it achieved."""
    assert classify_curve([3.0, 2.0, 1.5, 1.6, 4.0]) == "DIVERGED"
    # ... while a curve that merely wobbles around its best is not.
    assert classify_curve([3.0, 2.0, 1.5, 1.6, 1.7]) == "PLATEAUED"


def test_an_empty_curve_is_not_a_divergence() -> None:
    """No evaluations is not a blow-up -- the guard must not fire on a curve that never ran."""
    assert classify_curve([]) == "PLATEAUED"


# --- session-0013: the pre-registered CONVERGES / DIVERGES / BETWEEN rule -----------------------


def test_convergence_verdict_converges_on_a_healthy_curve() -> None:
    """CONVERGES = final < first AND grad norm bounded AND legality 100% -- all three, together."""
    from caissa.training.smoke_metrics import convergence_verdict

    out = convergence_verdict([3.4, 3.2, 3.1], [5.0, 4.0, 3.0], 1.0)
    assert out["status"] == "CONVERGES"


def test_convergence_verdict_diverges_on_a_blow_up() -> None:
    """DIVERGES = final > 2x best -- the session-0012 v2 shape."""
    from caissa.training.smoke_metrics import convergence_verdict

    out = convergence_verdict(V2_CURVE_0012, [5.0, 600.0], 1.0)
    assert out["status"] == "DIVERGES"


def test_convergence_verdict_is_between_when_grad_norm_explodes_but_loss_does_not() -> None:
    """A bounded loss with an exploding gradient is neither -- recorded as facts, not a verdict."""
    from caissa.training.smoke_metrics import convergence_verdict

    out = convergence_verdict([3.4, 3.3, 3.2], [1.0, 50.0], 1.0)
    assert out["status"] == "BETWEEN"


def test_convergence_verdict_requires_full_legality() -> None:
    """A single illegal move makes the run INVALID, not CONVERGES -- Article 1 is a gate."""
    from caissa.training.smoke_metrics import convergence_verdict

    out = convergence_verdict([3.4, 3.2], [5.0, 4.0], 0.999)
    assert out["status"] == "BETWEEN"
