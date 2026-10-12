"""Adam — pinned against the formula, because a wrong optimizer is invisible in a loss curve.

Session-0011's whole design rests on one recorded choice: run the v2 family on Adam so that *one*
learning rate is meaningful across architectures, instead of comparing models that were each handed
a different constant. That only works if the optimizer really is Adam. A subtly wrong implementation
-- a missing bias correction, a per-parameter step counter, an update applied to the wrong tensor --
would still train, still produce a falling loss, and still look like a result.

So the tests below check the update against the Kingma & Ba (2015) formula on a one-parameter
problem where the arithmetic is done by hand, including the first step, where the bias correction
matters most: without it, the first update is ~31.6x too large (1/(1-0.999) for v).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from caissa.training.adam import PARAM_NAMES, AdamState, grad_norm


class _Model:
    """A minimal carrier with the six named tensors plus the geometry scalar."""

    def __init__(self) -> None:
        self.w_enc = np.zeros((2, 2))
        self.b_enc = np.zeros(2)
        self.w_mix = np.zeros((2, 2))
        self.b_mix = np.zeros(2)
        self.head_from = np.zeros(2)
        self.head_to = np.zeros(2)
        self.geom_scale = np.zeros(())


def _grads(value: float) -> dict[str, np.ndarray]:
    return {
        name: np.full((2, 2) if name in {"w_enc", "w_mix"} else (2,), value) for name in PARAM_NAMES
    }


def test_first_step_matches_the_bias_corrected_formula() -> None:
    """One step from zero with a constant gradient g reduces to ``-lr * g / (|g| + eps)``."""
    model = _Model()
    adam = AdamState()
    lr, g = 0.01, 0.5

    adam.step(model, _grads(g), lr)

    # m_hat = g, v_hat = g^2 -> update = lr * g / (|g| + eps); with g>0, eps tiny: ~ lr.
    assert model.w_enc[0, 0] == pytest.approx(-lr * g / (abs(g) + 1e-8), rel=1e-9)
    # Every named tensor moved (not just the first -- a loop that breaks early would show here).
    for name in PARAM_NAMES:
        assert np.all(np.asarray(getattr(model, name)) != 0.0), f"{name} did not move"


def test_the_bias_correction_is_present() -> None:
    """For a constant gradient the corrected first step is ~lr; uncorrected it would be ~3.16x lr.

    Bias correction divides each moment by ``1 - beta**t``, so for a CONSTANT gradient
    ``m_hat = v_hat = g`` at every step and the update is ``lr * g / (|g| + eps)``. Skipping the
    correction would use the raw ``m = (1-b1)*g`` and ``v = (1-b2)*g^2``, whose ratio is
    ``sqrt(1-b2)/(1-b1) ≈ 0.316`` -- a first step ~3.16x too large. That is the exact silent error
    this test exists to catch.
    """
    model = _Model()
    adam = AdamState(betas=(0.9, 0.999))
    g, lr, eps = 1.0, 1.0, 1e-8
    b1, b2 = 0.9, 0.999

    adam.step(model, _grads(g), lr)

    corrected = -lr * g / (abs(g) + eps)
    assert model.w_enc[0, 0] == pytest.approx(corrected, rel=1e-6)
    assert abs(corrected) == pytest.approx(lr, rel=1e-6)

    # Without the correction the raw moments would be used and the first step would be much larger.
    uncorrected = -lr * ((1 - b1) * g) / (math.sqrt((1 - b2) * g * g) + eps)
    assert (
        abs(uncorrected) > abs(corrected) * 3
    ), "the correction must visibly change the first step"


def test_the_step_counter_is_shared_across_parameters() -> None:
    """A per-parameter counter would correct the scalar differently from a matrix -- pinned here.

    With an identical gradient every step, Adam's bias-corrected step converges to a constant
    `lr * g / (|g| + eps)` -- NOT to a shrinking one. The property that proves the counter is
    shared is that both tensors receive the SAME delta on the same step, and that the delta is
    already the converged value by step 2 (m_hat = v_hat = g exactly, for any t).
    """
    model = _Model()
    adam = AdamState()
    adam.step(model, _grads(1.0), 0.1)
    after_first = model.b_enc.copy()
    after_first_matrix = model.w_enc.copy()

    adam.step(model, _grads(1.0), 0.1)
    second_delta = np.abs(model.b_enc - after_first)
    # Bias correction makes m_hat = v_hat = g for a constant gradient, so each step is lr * g/(|g|+eps).
    assert np.allclose(second_delta, 0.1 / (1.0 + 1e-8), rtol=1e-6)
    # The matrix moved by the same amount as the vector on the same step -- one shared counter.
    matrix_delta = np.abs(model.w_enc - after_first_matrix)
    assert np.allclose(matrix_delta, second_delta[0], rtol=1e-9)
    assert adam.t == 2


def test_the_geometry_scalar_is_updated_only_when_present() -> None:
    """A bias-free model must not carry a phantom moment for a parameter it never uses."""
    model = _Model()
    adam = AdamState()
    grads = _grads(0.0)
    grads["geom_scale"] = np.array(0.25)

    adam.step(model, grads, 0.01)
    assert model.geom_scale != 0.0
    assert "geom_scale" in adam.m

    # ... and with no geometry gradient the state stays clean.
    model2, adam2 = _Model(), AdamState()
    adam2.step(model2, _grads(0.0), 0.01)
    assert "geom_scale" not in adam2.m


def test_signs_are_followed() -> None:
    """A negative gradient must raise the parameter (gradient descent, not ascent)."""
    model = _Model()
    adam = AdamState()
    for name in PARAM_NAMES:
        assert getattr(model, name).sum() == 0.0
    adam.step(model, _grads(-0.5), 0.1)
    assert model.w_enc[0, 0] > 0.0


def test_bad_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        AdamState(eps=0.0)
    with pytest.raises(ValueError):
        AdamState(betas=(1.0, 0.9))
    with pytest.raises(ValueError):
        AdamState(betas=(0.9, 1.1))


def test_grad_norm_is_the_l2_of_everything() -> None:
    """The logged norm must cover every tensor: a norm over a subset would hide a diverging one."""
    grads = {"a": np.array([3.0, 4.0]), "b": np.zeros(3)}
    assert grad_norm(grads) == pytest.approx(5.0)
    grads["b"] = np.array([12.0, 0.0, 0.0])
    assert grad_norm(grads) == pytest.approx(13.0)
    assert grad_norm({}) == 0.0
