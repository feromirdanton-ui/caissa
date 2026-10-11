"""Sharding to jsonl.zst + QA report (GOAL.md §5 Phase 2, §11 task 6).

Writes shards atomically (Article 4): a crash leaves previous shards intact.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chess.pgn
import zstandard as zstd

from caissa.processing.filters import game_elo_bucket, is_rated_standard


@dataclass
class QAReport:
    games: int = 0
    kept: int = 0
    elo_buckets: Counter = field(default_factory=Counter)
    results: Counter = field(default_factory=Counter)

    def to_markdown(self) -> str:
        lines = [
            "# QA report",
            "",
            f"- games seen: {self.games}",
            f"- games kept: {self.kept}",
            "",
            "## Elo buckets",
            "",
        ]
        for bucket, count in sorted(self.elo_buckets.items()):
            lines.append(f"- {bucket}: {count}")
        lines += ["", "## Results", ""]
        for result, count in sorted(self.results.items()):
            lines.append(f"- {result}: {count}")
        return "\n".join(lines) + "\n"


def _game_record(game: chess.pgn.Game) -> dict:
    return {
        "white": game.headers.get("White", "?"),
        "black": game.headers.get("Black", "?"),
        "white_elo": game.headers.get("WhiteElo", "?"),
        "black_elo": game.headers.get("BlackElo", "?"),
        "result": game.headers.get("Result", "*"),
        "elo_bucket": game_elo_bucket(game),
        "moves": game.end().ply(),
    }


def shard_games(
    games, out_dir: Path, shard_size: int = 10_000, report: QAReport | None = None
) -> QAReport:
    """Write games to jsonl.zst shards; return a QA report."""
    report = report or QAReport()
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buffer: list[dict] = []

    def flush() -> None:
        nonlocal shard_idx, buffer
        if not buffer:
            return
        path = out_dir / f"shard-{shard_idx:05d}.jsonl.zst"
        tmp = path.with_suffix(path.suffix + ".part")
        cctx = zstd.ZstdCompressor(level=10)
        with open(tmp, "wb") as fh:
            fh.write(cctx.compress(("\n".join(json.dumps(r) for r in buffer)).encode()))
        tmp.replace(path)  # atomic
        shard_idx += 1
        buffer = []

    for game in games:
        report.games += 1
        if not is_rated_standard(game):
            continue
        report.kept += 1
        report.elo_buckets[game_elo_bucket(game)] += 1
        report.results[game.headers.get("Result", "*")] += 1
        buffer.append(_game_record(game))
        if len(buffer) >= shard_size:
            flush()
    flush()
    return report
