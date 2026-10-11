"""Streaming PGN parser — generators, no RAM loading (GOAL.md §5 Phase 2).

python-chess is the single source of truth for rules (Article 1); this module only
*streams* games out of a file — plain or zstd-compressed — one at a time.

Memory is O(1) in the number of games: ``chess.pgn.read_game`` reads one game and the
generator yields it before the next is read. A whole archive is never materialised.

Malformed input is a **loud, localised** failure, not a silent skip. With ``strict=True``
a game the parser flagged raises :class:`PgnParseError` naming its index and the parser's
own error list, so a corrupt archive is diagnosed where it broke. With ``strict=False``
the bad games are counted in :class:`ParseStats` and the run continues — the count is the
caller's to report, never to hide (Article 5).

A truncated zstd stream (the Phase 1 day-prefix) is a legitimate input: ``truncated_ok``
ends iteration at the last complete game and sets ``stats.truncated``, so the truncation is
*named* rather than passed off as a complete parse.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import chess.pgn
import zstandard as zstd


class PgnParseError(RuntimeError):
    """A game could not be parsed. Carries the game index and the parser's errors."""

    def __init__(self, source: str, game_index: int, errors: list[str]) -> None:
        self.source = source
        self.game_index = game_index
        self.errors = errors
        detail = "; ".join(errors) if errors else "unknown parse error"
        super().__init__(f"{source}: game #{game_index} failed to parse: {detail}")


class TruncatedStreamError(RuntimeError):
    """A zstd stream ended before its frame was complete (a byte prefix, not a whole file)."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(
            f"{source}: zstd stream is truncated (the frame never ended). "
            "Pass truncated_ok=True to accept a byte prefix and record it."
        )


class _ZstdRaw(io.RawIOBase):
    """A raw byte stream over a zstd file that *knows* whether the frame ended.

    ``zstandard``'s ``stream_reader`` returns 0 bytes for a truncated frame **without
    raising** — a silent truncation, exactly the failure class this project exists to
    refuse. ``decompressobj().eof`` is the reliable signal: it is ``True`` only when the
    frame ended cleanly. This wrapper exposes that as ``truncated`` so the caller can
    *name* the truncation instead of reading an empty archive as a complete one.
    """

    def __init__(self, fh, chunk: int = 1 << 20) -> None:
        self._fh = fh
        self._decomp = zstd.ZstdDecompressor().decompressobj()
        self._chunk = chunk
        self._buf = b""
        self._done = False
        self._at_frame_start = True
        self.truncated = False

    def readable(self) -> bool:  # pragma: no cover - io contract
        return True

    def _feed(self, data: bytes) -> None:
        """Feed bytes to the current frame, rolling over to the next frame on ``eof``.

        A Lichess archive is a **concatenation of zstd frames**, and a ``decompressobj``
        stops at the end of the first one — feeding it more raises
        ``ZstdError: cannot use a decompressobj multiple times``. The bytes of the next
        frame are handed back in ``unused_data``, so a fresh object is started with them.
        """
        while data:
            out = self._decomp.decompress(data)
            if out:
                self._buf += out
            if self._decomp.eof:
                data = self._decomp.unused_data
                self._decomp = zstd.ZstdDecompressor().decompressobj()
                self._at_frame_start = True
            else:
                self._at_frame_start = False
                return

    def readinto(self, b) -> int:
        while not self._buf:
            if self._done:
                return 0
            data = self._fh.read(self._chunk)
            if not data:
                self._done = True
                # A stream that ends mid-frame is truncated; one that ends exactly on a
                # frame boundary is complete. `_at_frame_start` is the honest signal.
                if not self._at_frame_start:
                    self.truncated = True
                return 0
            self._feed(data)
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


@dataclass
class ParseStats:
    """What a parse pass actually saw — measured, never assumed."""

    games: int = 0
    skipped: int = 0
    truncated: bool = False
    errors: list[str] = field(default_factory=list)


def stream_pgn(
    handle: TextIO,
    *,
    strict: bool = True,
    source: str = "<stream>",
    stats: ParseStats | None = None,
) -> Iterator[chess.pgn.Game]:
    """Yield games one at a time from an open text handle.

    ``strict=True`` raises :class:`PgnParseError` on the first game the parser flagged.
    ``strict=False`` skips it and increments ``stats.skipped``.
    """
    stats = stats if stats is not None else ParseStats()
    index = 0
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        index += 1
        if game.errors:
            messages = [str(e) for e in game.errors]
            if strict:
                raise PgnParseError(source, index, messages)
            stats.skipped += 1
            stats.errors.extend(messages)
            continue
        stats.games += 1
        yield game


def iter_games(
    path: str | Path, *, strict: bool = True, stats: ParseStats | None = None
) -> Iterator[chess.pgn.Game]:
    """Yield games from a plain-text PGN file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        yield from stream_pgn(fh, strict=strict, source=str(path), stats=stats)


def iter_games_zst(
    path: str | Path,
    *,
    strict: bool = True,
    stats: ParseStats | None = None,
    truncated_ok: bool = False,
) -> Iterator[chess.pgn.Game]:
    """Yield games from a zstd-compressed PGN, decompressing as a stream.

    ``truncated_ok`` tolerates a stream that ends mid-frame (a byte prefix of a larger
    archive) and records ``stats.truncated`` — the truncation is reported, not hidden.
    Without it, a truncated stream raises :class:`TruncatedStreamError` rather than
    yielding an empty archive that looks complete.
    """
    stats = stats if stats is not None else ParseStats()
    with open(path, "rb") as fh:
        raw = _ZstdRaw(fh)
        text = io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8", errors="replace")
        try:
            yield from stream_pgn(text, strict=strict, source=str(path), stats=stats)
        finally:
            if raw.truncated:
                if not truncated_ok:
                    raise TruncatedStreamError(str(path))
                stats.truncated = True


def iter_games_any(
    path: str | Path,
    *,
    strict: bool = True,
    stats: ParseStats | None = None,
    truncated_ok: bool = False,
) -> Iterator[chess.pgn.Game]:
    """Dispatch on the file suffix: ``.zst`` streams through zstd, anything else is text."""
    if str(path).endswith(".zst"):
        yield from iter_games_zst(path, strict=strict, stats=stats, truncated_ok=truncated_ok)
    else:
        yield from iter_games(path, strict=strict, stats=stats)
