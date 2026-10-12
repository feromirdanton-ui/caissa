"""Set-locality of the most-common-move baseline (session-0010 pre-registration, item 3).

The defect this pins lived in session 0009: the model's top-1 (0.0925) was compared against a
baseline (0.0132) computed as "the **training** set's most common move, scored on the validation
set". Those are two different quantities measured with two different rulers, and a model can appear
to "beat the baseline" simply by playing a move the eval set barely contains.

The property that must hold: a **constant** policy can never beat the most common move of **its own
set** -- that move's frequency in the set is the strict upper bound for any constant predictor. So
the baseline has to be measured on the eval set itself, and a *foreign* set's most common move must
score no better on this set than the set's own baseline.
"""

from __future__ import annotations

import chess
import pytest

from caissa.training import smoke_train_v2
from caissa.training.dataset import HistorySample, move_index


def _label(move: chess.Move) -> str:
    """The artifact's move label: the flat (from, to) index, ``"frm:to"`` (indices, not SAN/UCI)."""
    index = move_index(move)
    return f"{index // 64}:{index % 64}"


class _Sample:
    """A minimal stand-in carrying only what `set_local_baseline` reads (the target move)."""

    def __init__(self, move: chess.Move) -> None:
        self.move = move
        self.history = (chess.Board(),)


def _uniform_set(move: chess.Move, n: int) -> list[_Sample]:
    return [_Sample(move) for _ in range(n)]


def test_baseline_is_the_most_common_move_of_the_given_set() -> None:
    a = chess.Move.from_uci("e2e4")
    b = chess.Move.from_uci("d2d4")
    # 3x e2e4, 1x d2d4 -> the set-local baseline is e2e4 at 0.75.
    samples = _uniform_set(a, 3) + _uniform_set(b, 1)
    label, accuracy, index = smoke_train_v2.set_local_baseline(samples)  # type: ignore[arg-type]

    assert label == _label(a)
    assert index == move_index(a)
    assert accuracy == pytest.approx(0.75)


def test_a_foreign_sets_move_scores_no_better_than_the_local_baseline() -> None:
    """The exact confounder: set B's most common move is rarer in A than A's own baseline.

    Under the old (cross-ruler) scheme the DEAD threshold could be set by B's move while the model
    was scored on A. Here we prove that move is strictly worse on A than A's own baseline -- i.e.
    the two rulers are not the same quantity, which is why the threshold had to become set-local.
    """
    a_common = chess.Move.from_uci("e2e4")
    b_common = chess.Move.from_uci("h2h4")
    set_a = _uniform_set(a_common, 8) + _uniform_set(chess.Move.from_uci("g1f3"), 2)

    # set B, in which h2h4 is the most common move and e2e4 never appears.
    set_b = _uniform_set(b_common, 9) + _uniform_set(chess.Move.from_uci("a2a3"), 1)

    a_label, a_baseline, _ = smoke_train_v2.set_local_baseline(set_a)  # type: ignore[arg-type]
    b_label, _, _ = smoke_train_v2.set_local_baseline(set_b)  # type: ignore[arg-type]

    assert a_label == _label(a_common)
    assert b_label == _label(b_common)
    # Scoring B's most common move on A yields exactly ZERO -- the "foreign ruler" the old baseline
    # used is a different quantity from A's own baseline (0.8 here).
    foreign_on_a = sum(1 for s in set_a if move_index(s.move) == move_index(b_common)) / len(set_a)
    assert foreign_on_a == 0.0
    assert a_baseline == pytest.approx(0.8)
    assert foreign_on_a < a_baseline


def test_baseline_follows_this_sets_own_counts() -> None:
    """The baseline is a property of THE SET: change which move dominates and it changes with it."""
    e4, d4 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    d4_dominant = _uniform_set(d4, 3) + _uniform_set(e4, 1)  # d2d4 at 0.75
    e4_dominant = _uniform_set(e4, 3) + _uniform_set(d4, 1)  # e2e4 at 0.75

    d4_label, d4_acc, _ = smoke_train_v2.set_local_baseline(d4_dominant)  # type: ignore[arg-type]
    e4_label, e4_acc, _ = smoke_train_v2.set_local_baseline(e4_dominant)  # type: ignore[arg-type]

    assert d4_label == _label(d4) and d4_acc == pytest.approx(0.75)
    assert e4_label == _label(e4) and e4_acc == pytest.approx(0.75)

    # And the frequency itself is local: more mass on the same move raises the baseline.
    heavier = _uniform_set(e4, 7) + _uniform_set(d4, 3)  # e2e4 at 0.70
    _, heavy_acc, _ = smoke_train_v2.set_local_baseline(heavier)  # type: ignore[arg-type]
    assert heavy_acc == pytest.approx(0.7)


def test_history_sample_is_accepted_directly() -> None:
    """The function is typed for `HistorySample`; a real instance must work, not only the stub."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    samples = [HistorySample(history=(board,), move=move) for _ in range(4)]
    label, accuracy, _ = smoke_train_v2.set_local_baseline(samples)
    assert label == _label(move)
    assert accuracy == pytest.approx(1.0)
