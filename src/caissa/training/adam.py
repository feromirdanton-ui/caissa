"""Adam, hand-written over the model's named gradients (session-0011).

Session-0009 found that v1 (a linear model) trained stably at ``lr=0.05`` while the v2 family
diverged above ``lr=0.01``. That chasm is the price of a badly conditioned objective, not evidence
of lower capacity — and comparing two architectures while each is forced onto a *different* step
size judges the schedules, not the models. Adam normalizes per-parameter step length, so one ``lr``
is meaningful across both, and the comparison stops depending on who was given the luckier constant.

Hand-written on purpose, like the rest of this pipeline: the optimizer is part of the training loop
whose metrics this project reports, so it is not taken on faith. ``tests/unit/test_adam.py`` pins
the exact update against the formula in Kingma & Ba (2015), including the bias correction — the
detail most likely to be silently wrong, and the one that makes early steps wildly too large if
omitted.

Nothing here reads the model's dtype policy or guesses: the parameter list is fixed and ordered, so
a run is reproducible step for step.
"""

from __future__ import annotations

import math

import numpy as np

# The learned tensors, in a FIXED order: an optimizer that iterates a dict would depend on
# insertion order, and a run that is not step-for-step reproducible is not a measurement.
PARAM_NAMES: tuple[str, ...] = (
    "w_enc",
    "b_enc",
    "w_mix",
    "b_mix",
    "head_from",
    "head_to",
)

# The geometric scale is a single learned scalar, present only when the bias is on. Kept separate
# so a bias-free model does not carry a phantom moment for a parameter it never uses.
GEOM_PARAM = "geom_scale"


def grad_norm(grads: dict[str, np.ndarray]) -> float:
    """L2 norm of the whole gradient — logged per step so a divergence is visible before NaN."""
    total = 0.0
    for value in grads.values():
        arr = np.asarray(value, dtype=float)
        total += float(np.sum(arr * arr))
    return math.sqrt(total)


class AdamState:
    """Adam moments for one model. Create one per configuration; never share across models."""

    def __init__(self, *, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1): {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0: {eps}")
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, model: object, grads: dict[str, np.ndarray], lr: float) -> None:
        """Apply one Adam update in place, bias-corrected.

        ``t`` counts steps for the whole model (not per parameter), which is what the bias
        correction assumes — a per-parameter counter would apply a different correction to the
        scalar than to a matrix and is a classic silent error.
        """
        self.t += 1
        names = list(PARAM_NAMES)
        if GEOM_PARAM in grads:
            names.append(GEOM_PARAM)

        for name in names:
            g = np.asarray(grads[name], dtype=float)
            if name not in self.m:
                self.m[name] = np.zeros_like(g)
                self.v[name] = np.zeros_like(g)
            # Moments are updated BEFORE bias correction so a caller can inspect them.
            self.m[name] = self.b1 * self.m[name] + (1.0 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[name] / (1.0 - self.b1**self.t)
            v_hat = self.v[name] / (1.0 - self.b2**self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(model, name, getattr(model, name) - update)


__all__ = ["GEOM_PARAM", "PARAM_NAMES", "AdamState", "grad_norm"]
