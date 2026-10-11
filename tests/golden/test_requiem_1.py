"""Requiem-1 — the eternal regression test (GOAL.md §8).

THIS TEST MUST NEVER BE DELETED.

Position: r3kbnr/ppp2ppp/2n5/3q4/4P1b1/2B2N2/PPP2PPP/R2QKB1R w KQkq - 0 10
Candidate Ne5 must raise an alarm (knight f3 is the only shield on the g4–d1 diagonal;
leaving exposes the queen to Bxd1). Be2 and Qe2 must pass.
"""

from __future__ import annotations

import chess

from caissa.guards.queens_guard import inspect_move


def test_requiem_1_ne5_raises_alarm(requiem_1: chess.Board) -> None:
    report = inspect_move(requiem_1, chess.Move.from_uci("f3e5"))
    assert report.legal
    assert not report.passed, "Requiem-1: Ne5 must NOT pass the guard"
    assert any(t.kind == "capture" for t in report.threats), report.summary()


def test_requiem_1_be2_passes(requiem_1: chess.Board) -> None:
    report = inspect_move(requiem_1, chess.Move.from_uci("f1e2"))
    assert report.legal
    assert report.passed, report.summary()


def test_requiem_1_qe2_passes(requiem_1: chess.Board) -> None:
    report = inspect_move(requiem_1, chess.Move.from_uci("d1e2"))
    assert report.legal
    assert report.passed, report.summary()


def test_requiem_1_fen_is_stable(requiem_1: chess.Board) -> None:
    assert requiem_1.fen().startswith("r3kbnr/ppp2ppp/2n5/3q4/4P1b1/2B2N2/PPP2PPP/R2QKB1R w")
