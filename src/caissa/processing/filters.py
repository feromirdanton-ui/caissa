"""Game filters: rated standard, Elo buckets, time control (GOAL.md §5 Phase 2).

Policy is fixed by ADR-0010. A filter is a pure predicate over a parsed game's headers —
no board is replayed — so it is cheap enough to run on every game of a multi-million-game
archive.

The session-0002 audit found the old ``is_rated_standard`` accepted *any* game carrying two
numeric Elos, which is not the same thing as a rated standard game: a rated variant game
(Chess960, Antichess) also carries Elos. The predicate below requires the Event to say
"Rated" **and** the Variant to be standard/absent, and it is tested against both.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess.pgn

ELO_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 1200),
    (1200, 1600),
    (1600, 2000),
    (2000, 2400),
    (2400, 4000),
)

# Lichess time-control categories, by estimated game length in seconds
# (base + 40 * increment), which is the convention Lichess itself uses.
BULLET_MAX_S = 180
BLITZ_MAX_S = 480
RAPID_MAX_S = 1500

STANDARD_VARIANTS = frozenset({"", "standard", "chess", "from position"})


def elo_bucket(elo: int) -> str:
    """The Elo bucket label for a rating, or ``"unknown"`` outside every bucket."""
    for lo, hi in ELO_BUCKETS:
        if lo <= elo < hi:
            return f"{lo}-{hi}"
    return "unknown"


def parse_time_control(header: str) -> tuple[int, int] | None:
    """Parse a ``TimeControl`` header into ``(base_seconds, increment_seconds)``.

    Returns ``None`` for correspondence (``-``) and for anything unparseable — the caller
    decides what to do with an unknown, it is never silently treated as a real control.
    """
    text = header.strip()
    if not text or text == "-":
        return None
    base_text, _, inc_text = text.partition("+")
    try:
        return int(base_text), int(inc_text or 0)
    except ValueError:
        return None


def time_control_category(header: str) -> str:
    """Classify a ``TimeControl`` header: bullet / blitz / rapid / classical / correspondence."""
    if header.strip() == "-":
        return "correspondence"
    parsed = parse_time_control(header)
    if parsed is None:
        return "unknown"
    base, increment = parsed
    estimated = base + 40 * increment
    if estimated < BULLET_MAX_S:
        return "bullet"
    if estimated < BLITZ_MAX_S:
        return "blitz"
    if estimated < RAPID_MAX_S:
        return "rapid"
    return "classical"


def is_rated_standard(game: chess.pgn.Game) -> bool:
    """True only for a *rated standard* game — Event says Rated, Variant is standard."""
    headers = game.headers
    if not headers.get("Event", "").lower().startswith("rated"):
        return False
    if headers.get("Variant", "").strip().lower() not in STANDARD_VARIANTS:
        return False
    return headers.get("WhiteElo", "?").isdigit() and headers.get("BlackElo", "?").isdigit()


def game_elo(game: chess.pgn.Game) -> int | None:
    """The mean of the two players' ratings, or ``None`` when either is unreadable."""
    try:
        white = int(game.headers.get("WhiteElo", ""))
        black = int(game.headers.get("BlackElo", ""))
    except ValueError:
        return None
    return (white + black) // 2


def game_elo_bucket(game: chess.pgn.Game) -> str:
    elo = game_elo(game)
    return elo_bucket(elo) if elo is not None else "unknown"


@dataclass(frozen=True)
class GameFilter:
    """The processing policy of ADR-0010, as a pure predicate.

    ``accepts`` returns ``(accepted, reason)`` so the caller can count *why* a game was
    rejected — a rejection counter is a QA fact, not a silent drop.
    """

    rated_standard_only: bool = True
    min_elo: int = 0
    max_elo: int = 4000
    time_controls: frozenset[str] = frozenset()  # empty = every category
    min_plies: int = 0

    def accepts(self, game: chess.pgn.Game) -> tuple[bool, str]:
        if self.rated_standard_only and not is_rated_standard(game):
            return False, "not_rated_standard"
        elo = game_elo(game)
        if elo is None:
            return False, "elo_unreadable"
        if elo < self.min_elo:
            return False, "elo_below_min"
        if elo >= self.max_elo:
            return False, "elo_above_max"
        if self.time_controls:
            category = time_control_category(game.headers.get("TimeControl", ""))
            if category not in self.time_controls:
                return False, f"time_control_{category}"
        if self.min_plies and game.end().ply() < self.min_plies:
            return False, "too_short"
        return True, "accepted"
