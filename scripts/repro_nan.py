"""Session-0009 diagnosis: why did the v2 run diverge to NaN?

The run's metrics artifact shows `train_loss_curve: [NaN, NaN]` and an IDENTICAL top-1 (0.0925)
for the main run and both ablations — the signature of a model that diverged, not a measurement.
This script names the cause on a small real subset: it trains the SAME model at several learning
rates and reports where the loss goes NaN. It changes nothing in the repo; it is a diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caissa.model.numpy_history import NumpyHistoryModel  # noqa: E402
from caissa.model.numpy_tiny import encode  # noqa: E402
from caissa.training.dataset import build_history_samples, load_games, split_games  # noqa: E402


def main() -> int:
    games = load_games("data/shards", max_positions=20_000, min_moves=10)
    train, _ = split_games(games, val_fraction=0.05, seed=0)
    samples = build_history_samples(train, history=8, limit=1_500)
    print(f"[repro] {len(samples)} samples")

    for lr in (0.05, 0.01, 0.005, 0.001):
        model = NumpyHistoryModel(history=8, d_model=256, n_heads=8, geometry_bias=True, seed=0)
        losses: list[float] = []
        first_nan = None
        for i, s in enumerate(samples[:400]):
            xh = np.stack([encode(b) for b in s.history], axis=0)
            loss = model.train_step(s.board, xh, s.move, lr=lr)
            losses.append(loss)
            if first_nan is None and not np.isfinite(loss):
                first_nan = i
        finite = [x for x in losses if np.isfinite(x)]
        print(
            f"[repro] lr={lr:<6} first={losses[0]:.4f} "
            f"last_finite={finite[-1] if finite else float('nan'):.4f} "
            f"first_nan_step={first_nan} nan_count={sum(1 for x in losses if not np.isfinite(x))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
