"""Queens Guard v0 — move hygiene (Article 2).

A pure function: position + candidate move -> enumeration of all threats after the move
(checks, captures of hanging pieces, pins, forks). No neural nets. python-chess only.

This is a HEURISTIC v0, honestly labeled as such. It is not a proof of safety; it is a
hygiene gate. Its first and eternal obligation is Requiem-1 (GOAL.md §8): the move Ne5 in
that position must raise an alarm, while Be2 and Qe2 must pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
}

# A threat is material-relevant if the opponent's best capture nets at least this much.
MATERIAL_THRESHOLD = 2


@dataclass(frozen=True)
class Threat:
    kind: str  # "capture" | "check" | "pin" | "fork"
    move: str  # UCI of the opponent move that realises the threat
    detail: str
    value: int = 0


@dataclass
class GuardReport:
    fen: str
    candidate: str
    legal: bool
    threats: list[Threat] = field(default_factory=list)

    @property
    def material_threats(self) -> list[Threat]:
        """Threats that actually cost material (captures and material-winning forks)."""
        return [t for t in self.threats if t.value >= MATERIAL_THRESHOLD]

    @property
    def passed(self) -> bool:
        """Hygiene passes if the move is legal and no *material* threat remains.

        Checks and pins are reported for the record but do not by themselves fail the guard:
        a check is not a blunder. Only a threat that wins material raises the alarm.
        """
        return self.legal and not self.material_threats

    def summary(self) -> str:
        if not self.legal:
            return f"ILLEGAL: {self.candidate} is not a legal move"
        if self.passed:
            return f"PASS: {self.candidate} — no material threat detected"
        lines = [f"ALARM: {self.candidate} leaves {len(self.material_threats)} material threat(s):"]
        lines += [f"  - [{t.kind}] {t.move}: {t.detail}" for t in self.material_threats]
        return "\n".join(lines)


def _least_valuable_legal_capturer(board: chess.Board, square: int) -> chess.Move | None:
    """Least valuable legal capture onto `square` for the side to move.

    Uses legal_moves so pins and king safety are respected (python-chess is the truth).
    """
    best: tuple[int, chess.Move] | None = None
    for move in board.legal_moves:
        if move.to_square != square:
            continue
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        value = PIECE_VALUES[piece.piece_type]
        if best is None or value < best[0]:
            best = (value, move)
    return best[1] if best else None


def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation of a capture, from the mover's perspective.

    Returns the net material the mover wins by initiating the exchange on move.to_square.
    Positive = good for the mover. Non-captures return 0.
    """
    target = board.piece_at(move.to_square)
    if target is None:
        return 0
    gain = [PIECE_VALUES[target.piece_type]]
    b = board.copy()
    b.push(move)
    while True:
        recapture = _least_valuable_legal_capturer(b, move.to_square)
        if recapture is None:
            break
        captured = b.piece_at(move.to_square)  # the piece the recapture takes
        assert captured is not None
        gain.append(PIECE_VALUES[captured.piece_type] - gain[-1])
        b.push(recapture)
    d = len(gain) - 1
    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1
    return gain[0]


def _pinned_squares(board: chess.Board) -> list[int]:
    """Squares of the side-to-move's pieces that are pinned to their king."""
    king = board.king(board.turn)
    if king is None:
        return []
    pinned: list[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != board.turn or piece.piece_type == chess.KING:
            continue
        if board.is_pinned(board.turn, square):
            pinned.append(square)
    return pinned


def _fork_targets(board: chess.Board, move: chess.Move) -> list[int]:
    """Squares of the mover's pieces (value >= 3) attacked by the piece after `move`."""
    b = board.copy()
    b.push(move)
    attacker_sq = move.to_square
    attacker = b.piece_at(attacker_sq)
    if attacker is None:
        return []
    targets: list[int] = []
    for square in b.attacks(attacker_sq):
        piece = b.piece_at(square)
        if piece is None or piece.color == attacker.color:
            continue
        if PIECE_VALUES[piece.piece_type] < 3:
            continue
        targets.append(square)
    return targets


def inspect_move(board: chess.Board, move: chess.Move) -> GuardReport:
    """Inspect a candidate move and enumerate the threats it leaves behind."""
    report = GuardReport(fen=board.fen(), candidate=move.uci(), legal=move in board.legal_moves)
    if not report.legal:
        return report

    after = board.copy()
    after.push(move)

    # 1. Captures of hanging pieces: opponent's best capture nets material.
    for opp_move in after.legal_moves:
        if after.piece_at(opp_move.to_square) is None:
            continue
        see = static_exchange_eval(after, opp_move)
        if see >= MATERIAL_THRESHOLD:
            captured = after.piece_at(opp_move.to_square)
            name = chess.piece_name(captured.piece_type) if captured else "piece"
            report.threats.append(
                Threat(
                    kind="capture",
                    move=opp_move.uci(),
                    detail=f"wins material (SEE +{see}) by capturing the {name}",
                    value=see,
                )
            )

    # 2. Checks available to the opponent after the move.
    for opp_move in after.legal_moves:
        probe = after.copy()
        probe.push(opp_move)
        if probe.is_check():
            report.threats.append(Threat(kind="check", move=opp_move.uci(), detail="gives check"))

    # 3. Pins created against us by the move (pieces of ours now pinned).
    for square in _pinned_squares(after):
        piece = after.piece_at(square)
        name = chess.piece_name(piece.piece_type) if piece else "piece"
        report.threats.append(
            Threat(
                kind="pin",
                move=chess.square_name(square),
                detail=f"our {name} on {chess.square_name(square)} is pinned to the king",
            )
        )

    # 4. Forks: opponent moves that attack two or more of our valuable pieces AND win
    #    material (at least one attacked piece is not adequately defended). A double attack
    #    on defended pieces is not a threat, so it is not reported.
    for opp_move in after.legal_moves:
        targets = _fork_targets(after, opp_move)
        if len(targets) < 2:
            continue
        probe = after.copy()
        probe.push(opp_move)
        wins = 0
        for square in targets:
            capture = chess.Move(opp_move.to_square, square)
            if capture in probe.legal_moves:
                wins = max(wins, static_exchange_eval(probe, capture))
        if wins >= MATERIAL_THRESHOLD:
            names = ", ".join(chess.square_name(s) for s in targets)
            report.threats.append(
                Threat(
                    kind="fork",
                    move=opp_move.uci(),
                    detail=f"forks {names} (wins material, SEE +{wins})",
                    value=wins,
                )
            )

    return report
