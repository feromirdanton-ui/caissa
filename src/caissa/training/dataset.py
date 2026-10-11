"""Position dataset from jsonl.zst shards (session-0007, milestone «Vertical»).

The shards written by Phase 2 now carry each game's moves (as UCI), so a shard is not merely a
statistic about games — it is **training data**. This module turns shards into the (position, move)
pairs the tiny model consumes, streaming and bounded: memory is O(positions actually kept), never
O(archive).

Honest assumptions, stated rather than hidden:

- A game is replayed from the standard initial position unless it carries a ``start_fen`` header.
  A game whose moves do not replay legally is **stopped at the break and counted** — never patched.
- The 95/5 split is made **by game**, not by position: positions from one game appear on exactly
  one side, so the validation split cannot leak a position's own predecessor to the training set.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import zstandard as zstd

Position = tuple[chess.Board, chess.Move]


@dataclass
class ShardStats:
    """What the loader actually read — measured, for the metrics artifact."""

    shards: list[str]
    games: int = 0
    positions: int = 0
    skipped_games: int = 0
    broken_replays: int = 0


def iter_shard_records(shards_dir: str | Path) -> Iterator[dict]:
    """Yield one dict per game record from every ``*.jsonl.zst`` shard, in file order."""
    paths = sorted(Path(shards_dir).glob("*.jsonl.zst"))
    for path in paths:
        with open(path, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            pending = b""
            while True:
                block = reader.read(1 << 20)
                if not block:
                    break
                pending += block
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.strip():
                        yield json.loads(line)
            if pending.strip():
                yield json.loads(pending)


def game_positions(record: dict) -> list[Position]:
    """Replay a record into ``(board, move)`` pairs, stopping at the first illegal move.

    Returns a list (not a generator) because the caller splits *by game* and needs the game's
    extent. A game that cannot be replayed is truncated at the break; the caller counts it.
    """
    uci_moves = record.get("uci") or []
    if not uci_moves:
        return []
    start_fen = record.get("start_fen") or ""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    positions: list[Position] = []
    for uci in uci_moves:
        try:
            move = chess.Move.from_uci(uci) if isinstance(uci, str) else uci
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        positions.append((board.copy(), move))
        board.push(move)
    return positions


def load_games(
    shards_dir: str | Path,
    *,
    max_positions: int,
    min_moves: int = 4,
    stats: ShardStats | None = None,
) -> list[list[Position]]:
    """Load whole games from the shards until ``max_positions`` positions are collected.

    Stops on a *game* boundary, so the last game is never half-loaded — the split downstream is
    by game and a half game would corrupt it.
    """
    shards_path = Path(shards_dir)
    stats = (
        stats
        if stats is not None
        else ShardStats(shards=sorted(p.name for p in shards_path.glob("*.jsonl.zst")))
    )
    games: list[list[Position]] = []
    total = 0
    for record in iter_shard_records(shards_path):
        positions = game_positions(record)
        if len(positions) != len(record.get("uci") or []):
            stats.broken_replays += 1
        if len(positions) < min_moves:
            stats.skipped_games += 1
            continue
        games.append(positions)
        stats.games += 1
        total += len(positions)
        if total >= max_positions:
            break
    stats.positions = total
    return games


def split_games(
    games: list[list[Position]], *, val_fraction: float = 0.05, seed: int = 0
) -> tuple[list[list[Position]], list[list[Position]]]:
    """Shuffle games (seeded) and split them into (train, val) **by game**."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    if val_fraction > 0 and len(shuffled) > 1 and n_val == 0:
        n_val = 1
    return shuffled[n_val:], shuffled[:n_val]


def flatten(games: list[list[Position]], limit: int | None = None) -> list[Position]:
    """Concatenate games into one position list, optionally capped at ``limit``."""
    out: list[Position] = []
    for game in games:
        out.extend(game)
        if limit is not None and len(out) >= limit:
            return out[:limit]
    return out


def move_index(move: chess.Move) -> int:
    """The flat (from, to) index of a move: ``from * 64 + to``."""
    return move.from_square * 64 + move.to_square
