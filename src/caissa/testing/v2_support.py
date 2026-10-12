"""Shared fixtures for the `smoke-train-v2` tests (session-0010).

Kept in its own small module so `test_smoke_train_v2.py` and `test_smoke_train_v2_guard.py` share
one shard/config builder without duplicating it -- and so every published file stays small enough to
move over the GitHub API without a transcription slip (ADR-0009).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from caissa.processing.pgn import iter_games
from caissa.processing.sharding import shard_games
from caissa.testing.realism import build_valid_pgn


def make_shards(tmp_path: Path, games: int = 60, shard_size: int = 15) -> Path:
    """A deterministic shard directory built from the realism fixture (no network, no clock)."""
    pgn = tmp_path / "sample.pgn"
    pgn.write_bytes(build_valid_pgn(games, seed=11))
    out = tmp_path / "shards"
    report = shard_games(iter_games(pgn), out, shard_size=shard_size)
    assert report.kept == games
    return out


def make_config(tmp_path: Path, shards: Path) -> Path:
    """A small but complete v2 config: two ablations, both scored on the same split as the main."""
    cfg = {
        "scale": "small",
        "shards": str(shards),
        "metrics": str(tmp_path / "qa" / "metrics-v2.json"),
        "ablations_out": str(tmp_path / "qa" / "ablations-v2.json"),
        "history": 4,
        "model": {"name": "numpy_history", "d_model": 16, "n_heads": 2, "geometric_bias": True},
        "val_fraction": 0.05,
        "seed": 0,
        "max_positions_to_load": 5000,
        "max_train_positions": 400,
        "max_val_positions": 120,
        "epochs": 2,
        "lr": 0.01,
        "v1_reference": True,
        "v1_lr": 0.05,
        "v1_epochs": 1,
        "sample_moves": 120,
        "budget_minutes": 90,
        "success_threshold": 0.30,
        "ablations": [
            {
                "name": "no_geometry_bias",
                "overrides": {"model": {"geometric_bias": False}, "epochs": 1},
            },
            {"name": "no_history", "overrides": {"history": 0, "window_history": 4, "epochs": 1}},
        ],
    }
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def poison(model: object) -> None:
    """Drive every learned encoder weight of a `NumpyHistoryModel` to NaN -- the mutation's hand."""
    import numpy as np

    model.w_enc = np.full_like(model.w_enc, np.nan)  # type: ignore[attr-defined]
