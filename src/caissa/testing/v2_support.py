"""Shared fixtures for the `smoke-train-v2` tests (session-0010; extended in 0011).

Kept in its own small module so `test_smoke_train_v2.py` and `test_smoke_train_v2_guard.py` share
one shard/config builder without duplicating it -- and so every published file stays small enough to
move over the GitHub API without a transcription slip (ADR-0009).

Session-0011 changed the config's shape: the two ablations became a **factorial** (`configurations`,
each with its own overrides), the budget became a **plateau** rather than an epoch count, and every
delta is now a **McNemar** comparison. The fixture below is deliberately tiny (d_model 8, K in {0,2})
so the whole machinery runs in seconds; the SHIPPED config's own shape -- the real
{128,256} x {0,8} factorial -- is asserted separately, against `configs/smoke-train-v2.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from caissa.processing.pgn import iter_games
from caissa.processing.sharding import shard_games
from caissa.testing.realism import build_valid_pgn


def make_shards(tmp_path: Path, games: int = 40, shard_size: int = 10) -> Path:
    """A deterministic shard directory built from the realism fixture (no network, no clock)."""
    pgn = tmp_path / "sample.pgn"
    pgn.write_bytes(build_valid_pgn(games, seed=11))
    out = tmp_path / "shards"
    report = shard_games(iter_games(pgn), out, shard_size=shard_size)
    assert report.kept == games
    return out


def make_config(tmp_path: Path, shards: Path, **overrides: object) -> Path:
    """A small but complete v2 config: one split, a plateau budget, Adam, and a factorial of three.

    `plateau_patience: 1` with `min_evals: 1` makes the plateau fire after a single non-improving
    evaluation, so the plateau MACHINERY is exercised without a long run. The shipped config uses
    patience 5 and a 20-epoch ceiling; the number is a fixture choice, and it is stated here rather
    than hidden in a magic constant.
    """
    cfg: dict = {
        "scale": "small",
        "lane": "deep",
        "shards": str(shards),
        "metrics": str(tmp_path / "qa" / "metrics-v2.json"),
        "ablations_out": str(tmp_path / "qa" / "ablations-v2.json"),
        "history": 2,
        "window_history": 2,
        "plateau_patience": 1,
        "plateau_eps": 0.0001,
        "min_evals": 1,
        "max_epochs": 2,
        "model": {"name": "numpy_history", "d_model": 8, "n_heads": 2, "geometric_bias": True},
        "val_fraction": 0.2,
        "seed": 0,
        "max_positions_to_load": 4000,
        "max_train_positions": 200,
        "max_val_positions": 80,
        "budget_minutes": 90,
        "optimizer": "adam",
        "lr": 0.005,
        "betas": [0.9, 0.999],
        "alpha": 0.05,
        "v1_reference": True,
        "v1_lr": 0.05,
        "guard_every": 50,
        "sample_moves": 60,
        "progress_marker": 0.30,
        "configurations": [
            {
                "name": "lower_capacity",
                "overrides": {"model": {"d_model": 4, "n_heads": 1}},
            },
            {"name": "no_history", "overrides": {"history": 0, "window_history": 2}},
            {
                "name": "lower_capacity_no_history",
                "overrides": {
                    "history": 0,
                    "window_history": 2,
                    "model": {"d_model": 4, "n_heads": 1},
                },
            },
        ],
    }
    cfg.update(overrides)
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def poison(model: object) -> None:
    """Drive every learned encoder weight of a `NumpyHistoryModel` to NaN -- the mutation's hand."""
    import numpy as np

    model.w_enc = np.full_like(model.w_enc, np.nan)  # type: ignore[attr-defined]
