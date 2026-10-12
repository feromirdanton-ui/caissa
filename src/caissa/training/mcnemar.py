"""McNemar's test — is a top-1 delta real, or is it dice?

Session-0011 compares several configurations on ONE eval set and reports a top-1 accuracy each. A
two-point gap on ten thousand positions looks like progress and can be nothing: two models that
agree on almost every position produce a delta carried by a handful of positions, and a table of
those disagreements is what decides whether the gap is a signal.

McNemar's test is exactly the right instrument here because the two models are scored on the SAME
positions — the observations are *paired*, so the only informative cells are the discordant ones:

                     v1 wrong    v1 right
    v2 right          n01         both right
    v2 wrong          both wrong  n10

``n01`` and ``n10`` are the positions where the two models disagree, and they are the entire test.
The concordant cells are reported too, because a reader should be able to recompute the p-value
without the code — a statistic that cannot be checked is a claim, not a measurement.

Two methods, chosen by the discordant count so the p-value is never a crude approximation when it
need not be:

  * **exact** (discordant < 25): the two-sided binomial p-value
    ``2 * sum_{i=0}^{min(n01,n10)} C(n,i) * 0.5**n``, capped at 1.0 — the pre-computed form for a
    small sample, where the normal approximation is known to be poor;
  * **normal** (discordant >= 25): ``z = (n01 - n10) / sqrt(n01 + n10)``, two-sided
    ``p = erfc(|z| / sqrt(2))``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Below this many discordant pairs the normal approximation is unreliable, so the exact
# binomial form is used instead. Recorded as a constant so the method is reproducible.
EXACT_MAX_DISCORDANT = 25


@dataclass(frozen=True)
class McNemarResult:
    """The full discordant table plus the test statistic, so the number can be audited."""

    n: int
    only_a_right: int  # n01: a correct, b wrong
    only_b_right: int  # n10: b correct, a wrong
    both_right: int
    both_wrong: int
    z: float
    p_value: float
    method: str
    favours: str  # "a", "b" or "tie" — which model the point estimate prefers
    significant: bool  # p_value < alpha, in the direction of `favours`

    def as_dict(self) -> dict[str, object]:
        """A JSON-safe view: numbers where a number exists, names for the method."""
        return {
            "n": self.n,
            "only_a_right": self.only_a_right,
            "only_b_right": self.only_b_right,
            "both_right": self.both_right,
            "both_wrong": self.both_wrong,
            "z": self.z,
            "p_value": self.p_value,
            "method": self.method,
            "favours": self.favours,
            "significant": self.significant,
        }


def _exact_two_sided(n: int, k: int) -> float:
    """Two-sided exact binomial p-value for ``k`` or fewer successes in ``n`` fair trials."""
    total = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * total * (0.5**n))


def mcnemar(a: Sequence[bool], b: Sequence[bool], *, alpha: float = 0.05) -> McNemarResult:
    """Compare two boolean correctness vectors measured on the SAME positions, in order.

    ``a`` and ``b`` must be the same length: the pairing *is* the experiment. A length mismatch is
    a loud ``ValueError`` rather than a silent zip-truncation, because a truncated pairing would
    quietly change the sample size and therefore the p-value.
    """
    if len(a) != len(b):
        raise ValueError(f"correctness vectors must be paired: len(a)={len(a)} != len(b)={len(b)}")
    if not a:
        raise ValueError("no positions to compare — an empty eval set proves nothing")

    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    both_right = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    both_wrong = len(a) - only_a - only_b - both_right

    discordant = only_a + only_b
    if discordant == 0:
        # The models agree on every position: no evidence either way, and no division by zero.
        z, p, method = 0.0, 1.0, "degenerate"
    elif discordant < EXACT_MAX_DISCORDANT:
        z = (only_a - only_b) / math.sqrt(discordant)
        p = _exact_two_sided(discordant, min(only_a, only_b))
        method = "exact"
    else:
        z = (only_a - only_b) / math.sqrt(discordant)
        p = math.erfc(abs(z) / math.sqrt(2.0))
        method = "normal"

    favours = "a" if only_a > only_b else ("b" if only_b > only_a else "tie")
    return McNemarResult(
        n=len(a),
        only_a_right=only_a,
        only_b_right=only_b,
        both_right=both_right,
        both_wrong=both_wrong,
        z=z,
        p_value=p,
        method=method,
        favours=favours,
        significant=bool(p < alpha and favours != "tie"),
    )


__all__ = ["EXACT_MAX_DISCORDANT", "McNemarResult", "mcnemar"]
