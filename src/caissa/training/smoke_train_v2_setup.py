"""Data preparation and the v1 reference for the reloaded scaled vertical (session-0010).

Split out of `smoke_train_v2.py` so every published file stays small enough to move over the GitHub
API without a transcription slip -- ADR-0009 makes the publish the risky step, and a small blob's
sha is cheap to verify. Nothing here is new logic; it is the same `_prepare` / v1 reference that
session-0008 ran, now in its own module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from caissa.model.numpy_tiny import NumpyTinyModel
from caissa.training.dataset import (
    HistorySample,
    Position,
    build_history_samples,
    count_short_games,
    flatten,
    split_games,
)
from caissa.training.smoke_metrics import tiny_accuracy, tiny_mean_loss


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
    )


def train_v1_reference(cfg: dict, prepared: Prepared) -> dict:
    """Retrain v1 (tiny linear model) in THIS run and score it on BOTH eval sets.

    v1's historical 0.1839 was measured on the ply>=0 set; v2 is measured on the ply>=K set, which
    drops the opening half-moves and is *harder*. Comparing v2 to the historical v1 number would mix
    the organ effect with a change of ruler, so we report:

      * `v1_at_v1_set` -- historical reading, reproduced (sanity);
      * `v1_at_v2_set` -- v1 on the SAME positions v2 is scored on. **This is the honest v1->v2
        comparison**, because only the model differs, not the ruler.

    v1 keeps its own stable learning rate (`v1_lr`): a learning rate is part of an architecture's
    price, so neither model is handicapped to flatter the other.
    """
    v1_lr = float(cfg.get("v1_lr", 0.05))
    epochs = int(cfg.get("v1_epochs", cfg.get("epochs", 2)))
    v1_train = flatten(prepared.train_games, limit=int(cfg.get("max_train_positions", 60000)))
    v1_val = flatten(prepared.val_games, limit=int(cfg.get("max_val_positions", 2000)))
    v1_on_v2 = [(s.board, s.move) for s in prepared.val_samples]

    model = NumpyTinyModel(seed=int(cfg.get("seed", 0)))
    rng = random.Random(int(cfg.get("seed", 0)))
    for _ in range(epochs):
        order = list(range(len(v1_train)))
        rng.shuffle(order)
        for idx in order:
            board, move = v1_train[idx]
            model.train_step(board, move, lr=v1_lr)
    return {
        "label": "v1_reference",
        "model": "numpy_tiny",
        "lr": v1_lr,
        "epochs": epochs,
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
