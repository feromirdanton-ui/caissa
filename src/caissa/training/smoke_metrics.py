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
    """


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


__all__ = [
    "DivergenceError",
    "accuracy_v2",
    "assert_finite",
    "cross_ruler_baseline",
    "mean_loss_v2",
    "parameters_finite",
    "samples_matrix",
    "set_local_baseline",
    "tiny_accuracy",
    "tiny_mean_loss",
    "write_failure",
]
