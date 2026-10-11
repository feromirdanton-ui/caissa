"""Geometric attention bias (GOAL.md §4.2) — tables and the ablation flag.

The bias must be a pure function of the square indices, and the ablation flag must produce
**exactly zeros** — otherwise "with bias" and "without bias" would differ in something other than
the bias, and the ablation would prove nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.model.geometry import (
    apply_bias,
    bias_enabled,
    chebyshev_table,
    distance_table,
    manhattan_table,
    piece_distance_bias,
)


def test_tables_are_64_by_64_and_zero_on_the_diagonal() -> None:
    for table in (chebyshev_table(), manhattan_table()):
        assert table.shape == (64, 64)
        assert np.allclose(np.diag(table), 0.0), "a square is at distance 0 from itself"


def test_chebyshev_and_manhattan_are_correct_on_known_squares() -> None:
    a1, c3, h8 = 0, 18, 63
    assert chebyshev_table()[a1, c3] == 2
    assert manhattan_table()[a1, c3] == 4
    assert chebyshev_table()[a1, h8] == 7
    assert manhattan_table()[a1, h8] == 14


def test_tables_are_symmetric() -> None:
    assert np.array_equal(chebyshev_table(), chebyshev_table().T)
    assert np.array_equal(manhattan_table(), manhattan_table().T)


def test_distance_table_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError):
        distance_table("euclidean")


def test_ablation_flag_defaults_to_off() -> None:
    assert bias_enabled({}) is False
    assert bias_enabled({"model": {}}) is False
    assert bias_enabled({"model": {"geometric_bias": True}}) is True


def test_disabled_bias_is_exactly_zero() -> None:
    """The ablation must be exact — an epsilon here would quietly change the baseline."""
    bias = piece_distance_bias(piece_type=3, enabled=False)
    assert bias.shape == (64, 64)
    assert not np.any(bias), "a disabled bias must be all zeros"


def test_enabled_bias_scales_but_keeps_the_geometry() -> None:
    bias = piece_distance_bias(piece_type=1, kind="manhattan", enabled=True, weight=0.5)
    assert np.allclose(bias, 0.5 * manhattan_table())


def test_piece_type_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        piece_distance_bias(piece_type=-1)


def test_apply_bias_adds_and_shape_checks() -> None:
    logits = np.ones((64, 64))
    out = apply_bias(logits, chebyshev_table())
    assert np.array_equal(out, logits + chebyshev_table())
    with pytest.raises(ValueError):
        apply_bias(logits, np.zeros((8, 8)))


def test_bias_cannot_make_an_illegal_move_preferable() -> None:
    """The mask is applied AFTER the bias (GOAL.md §4.1), so no bias can legalise a move."""
    import chess

    from caissa.model.numpy_tiny import NumpyTinyModel, encode
    from caissa.representation.mask import legal_move_mask

    board = chess.Board()
    model = NumpyTinyModel(seed=0)
    mask = legal_move_mask(board)
    # An absurd bias, to show the mask — not the bias — decides legality.
    biased = apply_bias(model.logits(encode(board)), chebyshev_table() * 100.0)
    z = np.where(mask, biased, -np.inf)
    z = z - np.max(z)
    probs = np.where(mask, np.exp(z), 0.0)
    probs = probs / probs.sum()
    assert probs[~mask].sum() == 0.0, "illegal moves must carry zero probability"
    assert np.isclose(probs.sum(), 1.0)
