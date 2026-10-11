"""Game identity and deduplication (GOAL.md §5 Phase 2).

A game's identity is its ``Site`` URL when present (Lichess: ``https://lichess.org/<id>``),
else a sha256 over the headers that define the game plus its move text. The id is stable
across runs, so a re-run of the pipeline recognises the same game.

Dedup state is a set of ids — the only state the streaming pipeline keeps. A sha256 hex
digest is 64 bytes in a Python set; a million games is ~100 MB, which is affordable and is
recorded in the QA report rather than assumed.
"""

from __future__ import annotations

import hashlib

import chess.pgn

# Headers that define a game's identity when there is no Site URL.
_IDENTITY_HEADERS = ("White", "Black", "Date", "Result", "WhiteElo", "BlackElo", "TimeControl")


def game_id(game: chess.pgn.Game) -> str:
    """A stable identity for a game: its Site URL, else a hash of headers + moves."""
    site = game.headers.get("Site", "").strip()
    if site and site != "?":
        return site
    parts = [f"{h}={game.headers.get(h, '')}" for h in _IDENTITY_HEADERS]
    parts.append(game.end().board().fen())
    parts.append(" ".join(move.uci() for move in game.mainline_moves()))
    return "sha256:" + hashlib.sha256("\n".join(parts).encode()).hexdigest()


class Deduplicator:
    """Remembers the ids it has seen and counts the repeats."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.duplicates = 0

    def is_new(self, game: chess.pgn.Game) -> bool:
        """True the first time a game is seen; False for every later copy."""
        gid = game_id(game)
        if gid in self.seen:
            self.duplicates += 1
            return False
        self.seen.add(gid)
        return True

    def __len__(self) -> int:
        return len(self.seen)
