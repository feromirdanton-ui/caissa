"""McNemar's test — the significance instrument behind every session-0011 delta.

A top-1 gap on ten thousand positions can be carried by a handful of positions. These tests pin the
statistic on cases whose p-value can be worked out by hand, so the number the run reports is a
number a reader can check:

  * **perfect agreement** -> nothing to test: method "degenerate", p = 1.0 (never a division by zero);
  * **a clean sweep** (k discordant, all one way) -> the two-sided exact p is ``2 / 2**k``;
  * **the exact/normal boundary** -> the method switches at exactly 25 discordant pairs;
  * **symmetry** -> swapping the two models swaps the winner and leaves p unchanged;
  * **pairing is mandatory** -> unequal lengths are a loud error, not a silent truncation.

The hand-computable cases are the point: a statistic whose test only checks "it returned a float"
would pass while being wrong.
"""

from __future__ import annotations

import math

import pytest

from caissa.training.mcnemar import EXACT_MAX_DISCORDANT, mcnemar


def test_perfect_agreement_is_not_a_result() -> None:
    """Two models agreeing on every position produce no evidence, and no division by zero."""
    votes = [True, True, False, False, True]
    result = mcnemar(votes, list(votes))

    assert result.method == "degenerate"
    assert result.p_value == 1.0
    assert result.z == 0.0
    assert result.favours == "tie"
    assert result.significant is False
    assert result.both_right == 3
    assert result.both_wrong == 2
    assert result.only_a_right == 0 and result.only_b_right == 0


def test_a_clean_sweep_has_a_hand_computable_exact_p_value() -> None:
    """10 discordant pairs, all won by `a`: exact two-sided p = 2 / 2**10 = 0.001953125."""
    a = [True] * 10 + [True] * 3  # wins the 10, agrees on 3
    b = [False] * 10 + [True] * 3
    result = mcnemar(a, b)

    assert result.only_a_right == 10
    assert result.only_b_right == 0
    assert result.both_right == 3
    assert result.method == "exact"
    assert result.p_value == pytest.approx(2 / 2**10)
    assert result.favours == "a"
    assert result.significant is True
    # z is the normal-approximation shadow of the same table, reported for completeness.
    assert result.z == pytest.approx(10 / math.sqrt(10))


def test_a_split_table_is_not_significant() -> None:
    """5 each way is the most balanced possible disagreement: p must be ~1, not "half significant"."""
    a = [True] * 5 + [False] * 5
    b = [False] * 5 + [True] * 5
    result = mcnemar(a, b)

    assert result.only_a_right == 5
    assert result.only_b_right == 5
    assert result.favours == "tie"
    assert result.significant is False
    # min(5,5)=5 -> 2 * sum_{i<=5} C(5,i) / 2**5 = 2 * (32) / 32 = 2, capped at 1.
    assert result.p_value == 1.0


def test_the_method_switches_at_the_recorded_boundary() -> None:
    """The exact binomial form is used below 25 discordant pairs, the normal form at and above."""
    below = 24
    a = [True] * below + [False] * 1
    b = [False] * below + [False] * 1
    r_below = mcnemar(a, b)
    assert r_below.only_a_right == below
    assert r_below.method == "exact"

    at = EXACT_MAX_DISCORDANT
    a2 = [True] * at + [False] * 1
    b2 = [False] * at + [True] * 1
    r_at = mcnemar(a2, b2)
    assert r_at.only_a_right == at and r_at.only_b_right == 1
    assert r_at.method == "normal"


def test_normal_form_matches_a_hand_computed_z() -> None:
    """20 vs 5 discordant: z = 15 / sqrt(25) = 3.0, p = erfc(3 / sqrt(2)) ≈ 0.0027."""
    a = [True] * 20 + [False] * 5 + [True] * 7
    b = [False] * 20 + [True] * 5 + [True] * 7
    result = mcnemar(a, b)

    assert result.only_a_right == 20 and result.only_b_right == 5
    assert result.method == "normal"
    assert result.z == pytest.approx(3.0)
    assert result.p_value == pytest.approx(math.erfc(3.0 / math.sqrt(2.0)))
    assert result.significant is True


def test_swapping_the_models_swaps_the_winner_and_keeps_p() -> None:
    """The test is symmetric: who is 'a' changes the direction, never the evidence."""
    a = [True] * 14 + [False] * 4 + [True] * 6
    b = [False] * 14 + [True] * 4 + [True] * 6

    forward = mcnemar(a, b)
    backward = mcnemar(b, a)

    assert forward.p_value == pytest.approx(backward.p_value)
    assert forward.z == pytest.approx(-backward.z)
    assert forward.favours == "a" and backward.favours == "b"
    # The discordant table is the same table, read from the other side.
    assert forward.only_a_right == backward.only_b_right
    assert forward.only_b_right == backward.only_a_right
    assert forward.both_right == backward.both_right


def test_alpha_decides_significance_not_the_point_estimate() -> None:
    """A 12-1 sweep is significant at alpha=0.05 and not at alpha=0.001 -- the threshold is explicit."""
    a = [True] * 12 + [False] * 1
    b = [False] * 12 + [True] * 1

    assert mcnemar(a, b, alpha=0.05).significant is True
    assert mcnemar(a, b, alpha=0.001).significant is False
    # ... while the direction of the point estimate does not move with alpha.
    assert mcnemar(a, b, alpha=0.001).favours == "a"


def test_unpaired_vectors_are_a_loud_error() -> None:
    """The pairing IS the experiment: a length mismatch must raise, never zip-truncate."""
    with pytest.raises(ValueError, match="paired"):
        mcnemar([True, False, True], [True, False])


def test_an_empty_eval_set_proves_nothing() -> None:
    with pytest.raises(ValueError, match="no positions"):
        mcnemar([], [])


def test_as_dict_is_json_safe() -> None:
    """The artifact is JSON: the view must carry plain numbers and names, nothing exotic."""
    result = mcnemar([True] * 6 + [False] * 2, [False] * 6 + [True] * 2)
    payload = result.as_dict()

    assert payload["n"] == 8
    assert isinstance(payload["p_value"], float)
    assert payload["method"] in {"exact", "normal", "degenerate"}
    assert payload["favours"] in {"a", "b", "tie"}
    assert isinstance(payload["significant"], bool)
    import json

    json.dumps(payload)  # must not raise
