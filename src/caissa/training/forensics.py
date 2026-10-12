"""Divergence forensics — which parameter group explodes first, and at which step (session-0013).

Session-0012 established *that* the v2 family diverges (val loss 3.35 -> 25.2, grad norm 4.8 -> 648)
but not *where*. A single global gradient norm says the model blew up; it does not say whether the
bilinear (from, to) head, the history mixer, or the shared encoder was the first to go. This module
records the per-group gradient norm, the embedding norms and ``max|logit|`` at every step, so the
first exploding group and the first exploding step are read off the artifact rather than guessed.

The hypothesis under test — stated, not assumed — is that the **bilinear head** is the first group to
explode: its multiplicative ``(from, to)`` interaction produces unbounded logits, so a large update
to ``head_from``/``head_to`` feeds back into the logits and then into the gradient. The history mixer
is the second suspect. The report either confirms or refutes this; it is an **observation, not a
verdict** (session-0013 step 1).

Nothing here decides a verdict. It is instrumentation: it measures, and the LOG reads the numbers.
"""

from __future__ import annotations

import math

import numpy as np

# The learned tensors, grouped by the organ they belong to. The grouping is the whole point: a
# global norm cannot attribute a blow-up, and the session-0013 question is exactly attribution.
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "encoder": ("w_enc", "b_enc"),
    "mixer": ("w_mix", "b_mix"),
    "head": ("head_from", "head_to"),
    "geometry": ("geom_scale",),
}

# The embedding tensors whose NORM (not gradient) is tracked: a weight matrix whose norm runs away
# is the other face of the same divergence, and it is visible even on a step whose gradient is small.
EMBEDDING_TENSORS: tuple[str, ...] = ("w_enc", "w_mix", "head_from", "head_to", "geom_scale")

# A group is called "exploded" once its gradient norm exceeds this multiple of its own first-step
# norm. Recorded as a constant so the threshold is reproducible and cannot be tuned after the fact.
EXPLOSION_FACTOR = 10.0


def _l2(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=float)
    return math.sqrt(float(np.sum(arr * arr)))


def group_grad_norms(grads: dict[str, np.ndarray]) -> dict[str, float]:
    """L2 norm of the gradient, per parameter group.

    A group absent from ``grads`` (e.g. ``geometry`` when the bias is off) is reported as ``0.0``
    rather than omitted, so every configuration's record has the same keys and a missing group is
    never mistaken for a zero one.
    """
    out: dict[str, float] = {}
    for group, names in PARAM_GROUPS.items():
        total = 0.0
        for name in names:
            if name in grads:
                arr = np.asarray(grads[name], dtype=float)
                total += float(np.sum(arr * arr))
        out[group] = math.sqrt(total)
    return out


def embedding_norms(model: object) -> dict[str, float]:
    """L2 norm of each learned tensor — the weight-side view of the same divergence."""
    out: dict[str, float] = {}
    for name in EMBEDDING_TENSORS:
        value = getattr(model, name, None)
        out[name] = _l2(value) if value is not None else 0.0
    return out


def max_abs_logit(model: object, x_hist: np.ndarray) -> float:
    """``max|logit|`` over the (from, to) grid — the quantity the bilinear head can drive to infinity.

    Read through the model's own ``logits`` so the geometry bias is included exactly as the loss
    sees it; a separate forward here would measure a different function (the session-0008 defect).
    """
    logits = np.asarray(model.logits(x_hist), dtype=float)  # type: ignore[attr-defined,type-var]
    return float(np.max(np.abs(logits)))


class ForensicsRecorder:
    """Per-step record of group gradient norms, embedding norms and ``max|logit|``.

    Keeps the full per-step series in memory (a handful of floats per step) so the FIRST exploding
    step is exact, and exposes a downsampled curve for the artifact so a 10k-step run does not
    publish 10k rows. The summary is what the LOG reads; the curve is what a reader can plot.
    """

    def __init__(self, *, factor: float = EXPLOSION_FACTOR, curve_points: int = 40) -> None:
        if factor <= 1.0:
            raise ValueError("explosion factor must be > 1")
        self.factor = float(factor)
        self.curve_points = int(curve_points)
        self.steps: list[int] = []
        self.group_norms: dict[str, list[float]] = {g: [] for g in PARAM_GROUPS}
        self.embed_norms: dict[str, list[float]] = {n: [] for n in EMBEDDING_TENSORS}
        self.logit_max: list[float] = []

    def record(
        self, step: int, grads: dict[str, np.ndarray], model: object, x_hist: np.ndarray
    ) -> None:
        """Record one step. Called from the training loop, on the real forward/backward path."""
        self.steps.append(int(step))
        norms = group_grad_norms(grads)
        for group in PARAM_GROUPS:
            self.group_norms[group].append(norms[group])
        emb = embedding_norms(model)
        for name in EMBEDDING_TENSORS:
            self.embed_norms[name].append(emb[name])
        self.logit_max.append(max_abs_logit(model, x_hist))

    def _first_explosion(self, series: list[float]) -> int | None:
        """The first step index (1-based) at which ``series`` exceeds ``factor`` x its first value.

        A group whose first value is zero (a bias that starts at zero, or the geometry scalar when
        the bias is off) has no meaningful ratio; it is reported as ``None`` rather than a spurious
        explosion at step 1.
        """
        if not series:
            return None
        base = series[0]
        if base <= 0.0:
            return None
        for i, value in enumerate(series):
            if value > self.factor * base:
                return i + 1
        return None

    def first_explosion(self) -> dict[str, int | None]:
        """Per-group first exploding step, plus ``"any"`` — the earliest across all groups."""
        per_group = {g: self._first_explosion(self.group_norms[g]) for g in PARAM_GROUPS}
        found = [v for v in per_group.values() if v is not None]
        per_group["any"] = min(found) if found else None
        return per_group

    def _downsample(self, series: list[float]) -> list[float]:
        if len(series) <= self.curve_points:
            return list(series)
        stride = max(1, len(series) // self.curve_points)
        return series[::stride]

    def as_dict(self) -> dict[str, object]:
        """A JSON-safe summary: per-group first/last/max, the first explosion, and a curve."""
        summary: dict[str, object] = {
            "factor": self.factor,
            "steps_recorded": len(self.steps),
            "first_explosion_step": self.first_explosion(),
            "groups": {},
            "embeddings": {},
            "logit_max": {
                "first": self.logit_max[0] if self.logit_max else None,
                "last": self.logit_max[-1] if self.logit_max else None,
                "max": max(self.logit_max) if self.logit_max else None,
                "curve": self._downsample(self.logit_max),
            },
        }
        groups = summary["groups"]
        assert isinstance(groups, dict)
        for group in PARAM_GROUPS:
            series = self.group_norms[group]
            groups[group] = {
                "first": series[0] if series else None,
                "last": series[-1] if series else None,
                "max": max(series) if series else None,
                "curve": self._downsample(series),
            }
        embeddings = summary["embeddings"]
        assert isinstance(embeddings, dict)
        for name in EMBEDDING_TENSORS:
            series = self.embed_norms[name]
            embeddings[name] = {
                "first": series[0] if series else None,
                "last": series[-1] if series else None,
                "max": max(series) if series else None,
            }
        return summary


__all__ = [
    "EMBEDDING_TENSORS",
    "EXPLOSION_FACTOR",
    "PARAM_GROUPS",
    "ForensicsRecorder",
    "embedding_norms",
    "group_grad_norms",
    "max_abs_logit",
]
