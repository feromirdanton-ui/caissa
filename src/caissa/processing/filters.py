"""Game filters: rated standard, Elo buckets (GOAL.md §5 Phase 2)."""

from __future__ import annotations

import chess.pgn

ELO_BUCKETS = [(0, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 4000)]


def elo_bucket(elo: int) -> str:
    for lo, hi in ELO_BUCKETS:
        if lo <= elo < hi:
            return f"{lo}-{hi}"
    return "unknown"


def is_rated_standard(game: chess.pgn.Game) -> bool:
    headers = game.headers
    if headers.get("Event", "").lower().startswith("rated"):
        return True
    # Lichess bulk games are all rated standard; accept when both Elos are present.
    return headers.get("WhiteElo", "?").isdigit() and headers.get("BlackElo", "?").isdigit()


def game_elo_bucket(game: chess.pgn.Game) -> str:
    try:
        white = int(game.headers.get("WhiteElo", "0"))
        black = int(game.headers.get("BlackElo", "0"))
    except ValueError:
        return "unknown"
    return elo_bucket((white + black) // 2)
