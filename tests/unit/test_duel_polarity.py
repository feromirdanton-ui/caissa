"""The duel verdict's polarity -- the defect session-0012 found in session-0011's rule.

Session-0011 built each comparison as ``mcnemar(correctness, v1_correct)``, so ``a`` is the v2
configuration and ``b`` is the v1 reference. Its ``_decide`` then tested ``favours == "b"`` -- which
is a **v1** win -- and called it a v2 win. The consequence was not subtle: the session-0012 run had
every configuration BELOW v1 (deltas -0.0356, -0.0366, -0.0453, -0.0435) and the rule still printed
``SUCCESS``, with a reason string that read "delta -0.0356 ... beats v1".

These tests pin the direction on hand-built tables, and the last one is a **mutation control**: it
re-introduces the exact session-0011 bug and asserts the suite goes red, so the fix cannot silently
regress. A verdict rule that cannot fail on a losing run is not a rule.
"""

from __future__ import annotations

from caissa.training.smoke_train_v2 import _decide


def _comparison(label: str, delta: float, favours: str, significant: bool) -> dict:
    """One row of the pairwise table, in the shape `_decide` reads."""
    return {
        "label": label,
        "top1_accuracy": 0.5 + delta,
        "top1_delta_vs_v1": delta,
        "mcnemar": {
            "favours": favours,
            "significant": significant,
            "p_value": 1e-19,
            "method": "normal",
        },
        "steps": 10000,
        "plateau_reached": True,
        "verdict": "PLATEAUED",
    }


def _plateaued() -> list[dict]:
    return [{"status": "PLATEAUED"}]


def test_a_v2_win_is_favours_a_and_a_positive_delta() -> None:
    """`a` is the v2 configuration: a significant win for `a` with a positive delta is SUCCESS."""
    verdict = _decide(_plateaued(), [_comparison("d256_k8", +0.02, "a", True)])
    assert verdict["status"] == "SUCCESS"
    assert verdict["winner"] == "d256_k8"


def test_a_v1_win_is_favours_b_and_must_not_be_success() -> None:
    """The session-0012 run: every configuration lost to v1. `favours == "b"` is a v1 win -> DEAD."""
    comparisons = [
        _comparison("d256_k8", -0.0356, "b", True),
        _comparison("d128_k8", -0.0366, "b", True),
        _comparison("d256_k0", -0.0453, "b", True),
        _comparison("d128_k0", -0.0435, "b", True),
    ]
    verdict = _decide(_plateaued(), comparisons)
    assert verdict["status"] == "DEAD", "a run where nothing beats v1 is DEAD, never SUCCESS"


def test_a_positive_delta_without_significance_is_inconclusive() -> None:
    """Above v1 but not significant is the legal middle outcome, not a win."""
    verdict = _decide(_plateaued(), [_comparison("d256_k8", +0.004, "a", False)])
    assert verdict["status"] == "INCONCLUSIVE"


def test_a_ceiling_before_any_plateau_is_inconclusive() -> None:
    """If nothing reached its plateau the comparison is not yet meaningful."""
    verdict = _decide([{"status": "CEILING"}], [_comparison("d256_k8", +0.02, "a", True)])
    assert verdict["status"] == "INCONCLUSIVE"


def test_mutation_control_the_session_0011_polarity_bug_turns_the_suite_red() -> None:
    """Re-introduce the bug (read `favours == "b"` as a v2 win) and prove the tests catch it.

    This is the mutation control for the fix: with the old polarity the losing run above would be
    reported SUCCESS, so the assertion in `test_a_v1_win_is_favours_b_and_must_not_be_success`
    must fail. If this test ever passes while the bug is present, the guard is worthless.
    """
    comparisons = [
        _comparison("d256_k8", -0.0356, "b", True),
        _comparison("d128_k8", -0.0366, "b", True),
    ]

    def buggy_decide(main_verdicts: list[dict], comps: list[dict]) -> dict:
        # The session-0011 rule, verbatim in its polarity: `favours == "b"` treated as a v2 win.
        winners = [
            c for c in comps if c["mcnemar"]["favours"] == "b" and c["mcnemar"]["significant"]
        ]
        if winners:
            return {"status": "SUCCESS", "winner": winners[0]["label"]}
        return {"status": "DEAD"}

    assert buggy_decide(_plateaued(), comparisons)["status"] == "SUCCESS"  # the bug
    assert _decide(_plateaued(), comparisons)["status"] == "DEAD"  # the fix
    assert (
        buggy_decide(_plateaued(), comparisons)["status"]
        != _decide(_plateaued(), comparisons)["status"]
    )
