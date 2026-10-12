"""Data preparation and the v1 reference for the convergence duel (session-0011).

Split out of `smoke_train_v2.py` so every published file stays small enough to move over the GitHub
API without a transcription slip -- ADR-0009 makes the publish the risky step, and a small blob's
sha is cheap to verify.

Two things changed in session-0011, both recorded in `LOG/session-0011-preregistration.md`:

  * `Prepared` now carries the **v1 position lists** alongside the v2 samples, so the v1 reference
    and the v2 family are guaranteed to be scored on positions derived from the SAME split, in the
    SAME order. Deriving them twice with the same seed would *usually* agree; carrying them makes
    it impossible for them not to, which is what a paired significance test requires.
  * the v1 reference is trained **to its own plateau** under the same plateau rule as the v2 family,
    because session-0011's question is about convergence limits, not step counts. v1 keeps its own
    optimizer and rate (SGD, lr=0.05): a learning rate is part of an architecture's price.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from caissa.model.numpy_tiny import NumpyTinyModel
from caissa.training.dataset import (
    HistorySample,
    Position,
    build_history_samples,
    count_short_games,
    flatten,
    split_games,
)
from caissa.training.smoke_metrics import PlateauTracker, tiny_accuracy, tiny_mean_loss


def merge(base: dict, overrides: dict) -> dict:
    """Deep-merge `overrides` into `base` (one level of nesting is all the configs use)."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


@dataclass
class Prepared:
    """The shared data surface: one split, one set of v2 windows, reused by every configuration."""

    train_games: list[list[Position]]
    val_games: list[list[Position]]
    train_samples: list[HistorySample]
    val_samples: list[HistorySample]
    short_games: int
    # The v2 samples projected to (board, move) in the SAME order, for the v1 reference. Carried
    # rather than re-derived so the pairing a McNemar test relies on cannot drift.
    v1_on_v2_set: list[Position] = field(default_factory=list)


def prepare(cfg: dict, games: list[list[Position]]) -> Prepared:
    """Build the split and the v2 history windows ONCE, so every configuration shares them.

    The window is built with the MAIN run's `K`, so an ablation that changes `history` still reads
    the SAME positions (session-0009 preflight): a smaller-`K` model consumes only the suffix it can
    read, so the ablation changes the model's input, never the set it is measured on.
    """
    history = int(cfg.get("history", 8))
    window_history = int(cfg.get("window_history", history))
    train_games, val_games = split_games(
        games,
        val_fraction=float(cfg.get("val_fraction", 0.05)),
        seed=int(cfg.get("seed", 0)),
    )
    train = build_history_samples(
        train_games, history=window_history, limit=int(cfg.get("max_train_positions", 60000))
    )
    val = build_history_samples(
        val_games, history=window_history, limit=int(cfg.get("max_val_positions", 2000))
    )
    if not train or not val:
        raise SystemExit(
            f"[smoke-train-v2] the history window (K={window_history}) left no samples. "
            "Lower `history` or use longer games -- never pad a window."
        )
    return Prepared(
        train_games=train_games,
        val_games=val_games,
        train_samples=train,
        val_samples=val,
        short_games=count_short_games(games, history=window_history),
        v1_on_v2_set=[(s.board, s.move) for s in val],
    )


def train_v1_reference(cfg: dict, prepared: Prepared) -> tuple[NumpyTinyModel, dict]:
    """Train v1 (tiny linear model) to its own plateau and return ``(model, metrics)``.

    The model is returned because the caller needs its **correctness vector** on the v2 eval set to
    run McNemar against the v2 family -- a single accuracy number cannot be tested for significance.
    """
    v1_lr = float(cfg.get("v1_lr", 0.05))
    max_epochs = int(cfg.get("max_epochs", 20))
    patience = int(cfg.get("plateau_patience", 5))
    eps = float(cfg.get("plateau_eps", 1e-4))
    min_evals = int(cfg.get("min_evals", 3))
    v1_train = flatten(prepared.train_games, limit=int(cfg.get("max_train_positions", 60000)))
    v1_val = flatten(prepared.val_games, limit=int(cfg.get("max_val_positions", 10000)))
    v1_on_v2 = prepared.v1_on_v2_set

    model = NumpyTinyModel(seed=int(cfg.get("seed", 0)))
    rng = random.Random(int(cfg.get("seed", 0)))
    tracker = PlateauTracker(patience=patience, eps=eps, min_evals=min_evals)
    val_curve = [tiny_mean_loss(model, v1_val)]
    tracker.update(val_curve[-1])
    epochs_run = 0
    while epochs_run < max_epochs:
        order = list(range(len(v1_train)))
        rng.shuffle(order)
        for idx in order:
            board, move = v1_train[idx]
            model.train_step(board, move, lr=v1_lr)
        epochs_run += 1
        val_curve.append(tiny_mean_loss(model, v1_val))
        if tracker.update(val_curve[-1]):
            break

    metrics = {
        "label": "v1_reference",
        "model": "numpy_tiny",
        "optimizer": "sgd",
        "lr": v1_lr,
        "epochs": epochs_run,
        "max_epochs": max_epochs,
        "converged": tracker.as_dict(),
        "val_loss_curve": val_curve,
        "train_positions": len(v1_train),
        "v1_at_v1_set": {
            "top1_accuracy": tiny_accuracy(model, v1_val),
            "val_loss": tiny_mean_loss(model, v1_val),
            "val_positions": len(v1_val),
        },
        "v1_at_v2_set": {
            "top1_accuracy": tiny_accuracy(model, v1_on_v2),
            "val_loss": tiny_mean_loss(model, v1_on_v2),
            "val_positions": len(v1_on_v2),
        },
    }
    return model, metrics
