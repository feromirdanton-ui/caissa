"""Metric, baseline and divergence-guard helpers for the reloaded scaled vertical (session-0010).

Split out of `smoke_train_v2.py` so the driver stays small; the pieces here are the ones the
pre-registration names.

  * `DivergenceError` + `parameters_finite` -- the NaN guard. Session-0009 produced a GREEN run
    whose numbers were NaN for the main config *and* both ablations, all three reporting an
    identical top-1, and it still finished green with an INCONCLUSIVE verdict. The number was not
    a measurement: the model was destroyed while the run looked fine. This guard removes that
    state at two levels (the training loop and the metrics writer), so a diverged run can never be
    mistaken for a finding.

  * `set_local_baseline` -- the most-common-move baseline measured on the EVAL SET ITSELF. A
    constant policy cannot beat the most common move of its own set, so baseline and model then
    share ONE ruler. The old behaviour scored the *training* set's most-common move on the
    validation set: 0.0925 vs 0.0132 was a comparison of two different quantities.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from caissa.model.numpy_history import NumpyHistoryModel
from caissa.model.numpy_tiny import NumpyTinyModel
from caissa.model.numpy_tiny import encode as encode_board
from caissa.training.dataset import HistorySample, Position, move_index


class DivergenceError(RuntimeError):
    """A non-finite loss/parameter/metric: the instrument is destroyed, not a result.

    A *defect guard*, not a criterion -- it does not decide SUCCESS/DEAD/INCONCLUSIVE, it removes
    a state in which those words are meaningless (session-0010 pre-registration, the INVALID
    branch). `run()` fails the process so a diverged run cannot be read as a result.

    Session-0013 extends the same guard to a **finite** divergence: a curve whose final validation
    loss is more than `DIVERGENCE_FACTOR` times its own best is a blow-up even though every number
    is finite, and it must fail the run exactly as a NaN does. Session-0012's run was the case in
    point -- val loss 3.35 -> 25.2, all finite, reported `PLATEAUED`.
    """


# A curve whose final validation loss exceeds this multiple of its own BEST is a divergence, not a
# plateau. Recorded as a constant so the threshold is reproducible and cannot be tuned after the
# fact. The value is chosen against the two real curves of session-0012: v1 rises 3.1663 -> 3.5408
# (1.12x -- overfitting, NOT a blow-up) while v2 rises 3.3453 -> 25.1846 (7.53x -- a blow-up).
DIVERGENCE_FACTOR = 2.0


# The pre-registered convergence rule of session-0013 (LOG/session-0013-preregistration.md).
# CONVERGES = final val loss < first AND grad norm never exceeds GRAD_EXPLOSION_FACTOR x its first
#             AND legality is 100%; DIVERGES = final val loss > DIVERGENCE_FACTOR x the best;
# anything else is recorded as facts, not a verdict. Recorded as constants so the thresholds are
# reproducible and cannot be tuned after the shot.
GRAD_EXPLOSION_FACTOR = 10.0


def convergence_verdict(
    val_curve: list[float],
    grad_norms: list[float],
    legality: float,
    *,
    divergence_factor: float = DIVERGENCE_FACTOR,
    grad_factor: float = GRAD_EXPLOSION_FACTOR,
) -> dict[str, object]:
    """Apply the pre-registered CONVERGES / DIVERGES / BETWEEN rule to one configuration's run.

    This is the session-0013 question made mechanical: does this configuration converge at all?
    It is deliberately NOT the duel verdict -- a configuration that converges has earned a rematch
    with v1 under its own pre-registration (0014), and one that does not has not.
    """
    if not val_curve:
        return {"status": "BETWEEN", "reason": "no validation curve was recorded"}
    first, final, best = val_curve[0], val_curve[-1], min(val_curve)
    grad_first = grad_norms[0] if grad_norms else 0.0
    grad_max = max(grad_norms) if grad_norms else 0.0
    grad_ok = grad_first <= 0.0 or grad_max <= grad_factor * grad_first
    loss_ok = final < first
    if loss_ok and grad_ok and legality >= 1.0:
        return {
            "status": "CONVERGES",
            "reason": (
                f"final val loss {final:.4f} < first {first:.4f}; grad norm max {grad_max:.3g} "
                f"<= {grad_factor}x first {grad_first:.3g}; legality 100%"
            ),
        }
    if best > 0.0 and final > divergence_factor * best:
        return {
            "status": "DIVERGES",
            "reason": (
                f"final val loss {final:.4f} > {divergence_factor}x best {best:.4f} -- a blow-up"
            ),
        }
    return {
        "status": "BETWEEN",
        "reason": (
            f"neither converged nor diverged: final {final:.4f} vs first {first:.4f}, "
            f"grad max {grad_max:.3g} vs {grad_factor}x first {grad_first:.3g}, legality {legality:.3f}"
        ),
    }


def classify_curve(val_curve: list[float], *, factor: float = DIVERGENCE_FACTOR) -> str:
    """The ONE function that decides PLATEAUED vs DIVERGED, shared by the detector and the guard.

    A single implementation is deliberate: session-0012's defect was that the plateau detector and
    the divergence guard disagreed about the same curve (the detector said `PLATEAUED`, the guard
    said nothing). Two code paths for "what happened to this curve" is how that disagreement gets
    in; one path is how it stays out.

    Returns ``"DIVERGED"`` when the final value exceeds ``factor`` x the best value, else
    ``"PLATEAUED"``. The comparison is against the BEST, not the first: v1's own curve rises from
    its first reading (3.1663 -> 3.5408) because it overfits, and calling that a divergence would
    be wrong -- the best is what the model actually achieved.
    """
    if not val_curve:
        return "PLATEAUED"
    best = min(val_curve)
    final = val_curve[-1]
    if best > 0.0 and final > factor * best:
        return "DIVERGED"
    return "PLATEAUED"


def parameters_finite(model: NumpyHistoryModel) -> bool:
    """True iff every learned tensor (and the geometry scale) is finite.

    Exposed so the training loop can assert finiteness every N steps and so the mutation control
    can drive the model into a non-finite state deliberately.
    """
    tensors = (
        model.w_enc,
        model.b_enc,
        model.w_mix,
        model.b_mix,
        model.head_from,
        model.head_to,
        model.geom_scale,
    )
    return all(bool(np.all(np.isfinite(t))) for t in tensors)


def assert_finite(obj: object, path: str = "metrics") -> None:
    """Refuse to persist a metrics artifact that contains a non-finite number.

    Walks the structure and raises on the first non-finite float, so the artifact that lands is
    either honest or absent -- never a plausible-looking lie (session-0009's green NaN artifact).
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_finite(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            assert_finite(value, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise DivergenceError(f"{path} is not finite ({obj}) -- refusing to write the artifact")


def write_failure(path: Path, label: str, exc: BaseException, partial: dict) -> None:
    """Persist a FAILURE artifact (not metrics) carrying the cause, so a red run is legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "caissa-smoke-train-v2-failure/1",
        "label": label,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "verdict": {
            "status": "INVALID",
            "reason": "the instrument was destroyed (non-finite loss/parameters) -- not a finding",
        },
        "partial": partial,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def set_local_baseline(samples: list[HistorySample]) -> tuple[str, float, int]:
    """Most-common move OF THE EVAL SET, as ``("frm:to", accuracy, index)``."""
    counts = Counter(move_index(s.move) for s in samples)
    index, count = counts.most_common(1)[0]
    accuracy = count / max(1, len(samples))
    return f"{index // 64}:{index % 64}", accuracy, index


def cross_ruler_baseline(
    train_samples: list[HistorySample], val_samples: list[HistorySample]
) -> tuple[str, float]:
    """The earlier, cross-ruler number (training set's most-common move on the val set).

    Kept for transparency only; NOT used for the DEAD threshold. A reader can see both rulers and
    why the verdict moved to the set-local one.
    """
    counts = Counter(move_index(s.move) for s in train_samples)
    index, _ = counts.most_common(1)[0]
    move = f"{index // 64}:{index % 64}"
    accuracy = sum(1 for s in val_samples if move_index(s.move) == index) / max(1, len(val_samples))
    return move, accuracy


def samples_matrix(sample: HistorySample, history: int) -> np.ndarray:
    """Encode the LAST ``history + 1`` boards of a sample's window.

    The window is built once with the MAIN run's ``K``, so every configuration is scored on the
    SAME positions (session-0009 preflight). A model with a smaller ``K`` consumes only the
    suffix it can read -- ``K=0`` sees the current board alone -- so an ablation changes the
    model's input, never the set it is measured on.
    """
    boards = sample.history[-(history + 1) :]
    return np.stack([encode_board(board) for board in boards], axis=0)


def accuracy_v2(model: NumpyHistoryModel, samples: list[HistorySample], history: int) -> float:
    """Top-1 (from, to) accuracy of the masked policy over ``samples``."""
    if not samples:
        return 0.0
    hits = 0
    for sample in samples:
        probs = model.masked_probs(sample.board, samples_matrix(sample, history))
        if int(np.argmax(probs)) == move_index(sample.move):
            hits += 1
    return hits / len(samples)


def mean_loss_v2(model: NumpyHistoryModel, samples: list[HistorySample], history: int) -> float:
    """Mean masked cross-entropy over ``samples`` without touching the weights."""
    if not samples:
        return float("nan")
    total = sum(model.loss(s.board, samples_matrix(s, history), s.move) for s in samples)
    return total / len(samples)


def tiny_accuracy(model: NumpyTinyModel, positions: list[Position]) -> float:
    """Top-1 accuracy of the v1 (tiny, K=0) model over explicit ``(board, move)`` pairs."""
    if not positions:
        return 0.0
    hits = 0
    for board, move in positions:
        if int(np.argmax(model.masked_probs(board))) == move_index(move):
            hits += 1
    return hits / len(positions)


def tiny_mean_loss(model: NumpyTinyModel, positions: list[Position]) -> float:
    """Mean masked cross-entropy of the v1 model (``lr=0.0`` step reads without updating)."""
    if not positions:
        return float("nan")
    return sum(model.train_step(b, m, lr=0.0) for b, m in positions) / len(positions)


def correctness_v2(
    model: NumpyHistoryModel, samples: list[HistorySample], history: int
) -> list[bool]:
    """Top-1 correct/incorrect per position, IN ORDER — the input McNemar needs.

    `accuracy_v2` collapses this to a single number; the pairing is what a significance test
    requires, so the vector is exposed rather than re-derived. The order is the eval set's own
    order, identical for every configuration, which is what makes the pairing valid.
    """
    out: list[bool] = []
    for sample in samples:
        probs = model.masked_probs(sample.board, samples_matrix(sample, history))
        out.append(int(np.argmax(probs)) == move_index(sample.move))
    return out


def correctness_tiny(model: NumpyTinyModel, positions: list[Position]) -> list[bool]:
    """The v1 model's correctness vector over the SAME positions, in the same order."""
    return [
        int(np.argmax(model.masked_probs(board))) == move_index(move) for board, move in positions
    ]


class PlateauTracker:
    """Stop training when the validation loss stops improving — the session-0011 budget.

    Equal step counts judge the stopwatch: two configurations trained for the same number of steps
    are not being compared if one is still falling. This tracker stops a configuration at its own
    plateau, and reports the step and epoch at which that happened, so "trained to convergence" is
    a visible fact rather than an assertion.

    `eps` (absolute) is deliberate: without a margin, float jitter on a flat curve would register
    as improvement forever and the patience would never fire.
    """

    def __init__(self, *, patience: int = 5, eps: float = 1e-4, min_evals: int = 3) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_evals < 1:
            raise ValueError("min_evals must be >= 1")
        self.patience = patience
        self.eps = eps
        self.min_evals = min_evals
        self.best = float("inf")
        self.best_eval = 0
        self.since_improvement = 0
        self.evals = 0
        # The full curve, kept so the DIVERGED/PLATEAUED decision (session-0013) is made by the
        # SAME function the guard uses, on the same numbers -- not re-derived from `best` alone.
        self.values: list[float] = []

    def update(self, value: float) -> bool:
        """Record an evaluation; return True once patience has been exhausted."""
        self.evals += 1
        if not math.isfinite(value):
            raise DivergenceError(
                f"a non-finite validation loss ({value}) -- the instrument is gone"
            )
        self.values.append(value)
        if value < self.best - self.eps:
            self.best = value
            self.best_eval = self.evals
            self.since_improvement = 0
        else:
            self.since_improvement += 1
        return self.evals >= self.min_evals and self.since_improvement >= self.patience

    @property
    def reached(self) -> bool:
        return self.evals >= self.min_evals and self.since_improvement >= self.patience

    @property
    def diverged(self) -> bool:
        """True when the curve blew up (final > factor x best) -- the session-0013 status."""
        return classify_curve(self.values) == "DIVERGED"

    @property
    def status(self) -> str:
        """``DIVERGED`` / ``PLATEAUED`` / ``TRAINING`` -- the one word the detector and guard share."""
        if self.diverged:
            return "DIVERGED"
        return "PLATEAUED" if self.reached else "TRAINING"

    def as_dict(self) -> dict[str, object]:
        return {
            "patience": self.patience,
            "eps": self.eps,
            "min_evals": self.min_evals,
            "evaluations": self.evals,
            "best_val_loss": self.best,
            "best_eval": self.best_eval,
            "evals_without_improvement": self.since_improvement,
            "plateau_reached": self.reached,
            "diverged": self.diverged,
            "status": self.status,
            "divergence_factor": DIVERGENCE_FACTOR,
        }


__all__ = [
    "DIVERGENCE_FACTOR",
    "GRAD_EXPLOSION_FACTOR",
    "DivergenceError",
    "PlateauTracker",
    "accuracy_v2",
    "assert_finite",
    "classify_curve",
    "convergence_verdict",
    "correctness_tiny",
    "correctness_v2",
    "cross_ruler_baseline",
    "mean_loss_v2",
    "parameters_finite",
    "samples_matrix",
    "set_local_baseline",
    "tiny_accuracy",
    "tiny_mean_loss",
    "write_failure",
]
