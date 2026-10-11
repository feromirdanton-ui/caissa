"""Geometric attention bias — precomputed 64x64 distance tables (GOAL.md §4.2).

The policy head over (from, to) pairs is a *board* question: how far a move travels and along
which geometry matters, and a learned bias from a precomputed table gives the model that
structure for free instead of asking it to discover it. Two geometries are provided:

- **Chebyshev** (king-move distance) — the natural metric for a king, and a good general one.
- **Manhattan** — the rook/bishop-family metric.

The bias is **behind an ablation flag**: with the flag off, :func:`piece_distance_bias` returns a
zero table, so a run with the bias and a run without it differ in exactly one thing and are
therefore comparable. Nothing here is learned or magic — the tables are pure functions of the
square indices, computed once.

This module is deliberately numpy-only and torch-free, so the CPU smoke and the tiny model can use
it on any runner.
"""

from __future__ import annotations

import numpy as np

# A table is indexed [from_square, to_square] with python-chess square numbers (a1=0 .. h8=63).
_DISTANCE_KINDS = ("chebyshev", "manhattan")


def _rank(square: int) -> int:
    return square // 8


def _file(square: int) -> int:
    return square % 8


def chebyshev_table() -> np.ndarray:
    """64x64 table of king-move (Chebyshev) distances between squares."""
    table = np.zeros((64, 64), dtype=float)
    for a in range(64):
        for b in range(64):
            table[a, b] = max(abs(_rank(a) - _rank(b)), abs(_file(a) - _file(b)))
    return table


def manhattan_table() -> np.ndarray:
    """64x64 table of Manhattan distances between squares."""
    table = np.zeros((64, 64), dtype=float)
    for a in range(64):
        for b in range(64):
            table[a, b] = abs(_rank(a) - _rank(b)) + abs(_file(a) - _file(b))
    return table


def distance_table(kind: str = "chebyshev") -> np.ndarray:
    """Return the 64x64 distance table for ``kind`` (``chebyshev`` or ``manhattan``)."""
    if kind == "chebyshev":
        return chebyshev_table()
    if kind == "manhattan":
        return manhattan_table()
    raise ValueError(f"unknown distance kind {kind!r}; expected one of {_DISTANCE_KINDS}")


def bias_enabled(config: dict) -> bool:
    """The ablation flag: is the geometric bias switched on for this run?

    Default is **off** — a run must opt in explicitly, so the baseline is the unbiased model and
    an ablation is a config change rather than a code change.
    """
    model = config.get("model") or {}
    return bool(model.get("geometric_bias", False))


def piece_distance_bias(
    piece_type: int = 0,
    kind: str = "chebyshev",
    *,
    enabled: bool = True,
    weight: float = 1.0,
) -> np.ndarray:
    """A 64x64 bias table for one piece type, or a zero table when the ablation flag is off.

    ``piece_type`` is a python-chess piece type (1..6); it only scales the table today (all piece
    types share the same geometry) and exists so a per-piece table can be fitted later without
    changing the call site. With ``enabled=False`` the result is exactly zeros, so the model's
    logits are unchanged — that is what makes the ablation honest.
    """
    if not enabled:
        return np.zeros((64, 64), dtype=float)
    if piece_type < 0:
        raise ValueError("piece_type must be >= 0")
    return weight * distance_table(kind)


def apply_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Add a geometric bias to (from, to) logits, leaving them otherwise untouched.

    Shape-checked: a silent broadcast would quietly bias the wrong axis, which is exactly the
    class of defect that is invisible in a loss curve.
    """
    if logits.shape != bias.shape:
        raise ValueError(f"shape mismatch: logits {logits.shape} vs bias {bias.shape}")
    return logits + bias
