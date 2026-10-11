"""Streaming PGN parser — generators, no RAM loading (GOAL.md §5 Phase 2).

Uses python-chess's streaming reader. Never loads a whole file into memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import chess.pgn


def stream_pgn(handle: TextIO) -> Iterator[chess.pgn.Game]:
    """Yield games one at a time from an open text handle."""
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return
        yield game


def iter_games(path: str | Path) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN file (plain text; .zst handled by the caller)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        yield from stream_pgn(fh)
