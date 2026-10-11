"""Sharding to jsonl.zst + QA report (GOAL.md §5 Phase 2, §11 task 6).

Writes shards atomically (Article 4): a crash leaves previous shards intact, and a shard is
published by an atomic rename only after its bytes are on disk. Every shard carries its own
sha256 in the QA report, so a shard is a *certified* artifact, not a hopeful file (Article 14).

The QA report is measured from the games that actually passed the filter — Elo, time-control
and opening distributions, plus a counter for every rejection reason. Nothing is estimated.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import chess.pgn
import zstandard as zstd

from caissa.processing.dedup import Deduplicator
from caissa.processing.filters import (
    GameFilter,
    game_elo_bucket,
    time_control_category,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class ShardInfo:
    name: str
    rows: int
    bytes: int
    sha256: str


@dataclass
class QAReport:
    """Everything the pipeline measured, in one place."""

    games_seen: int = 0
    kept: int = 0
    duplicates: int = 0
    rejected: Counter = field(default_factory=Counter)
    elo_buckets: Counter = field(default_factory=Counter)
    time_controls: Counter = field(default_factory=Counter)
    openings: Counter = field(default_factory=Counter)
    results: Counter = field(default_factory=Counter)
    shards: list[ShardInfo] = field(default_factory=list)
    parse_skipped: int = 0
    truncated: bool = False

    @property
    def shard_rows(self) -> int:
        return sum(s.rows for s in self.shards)

    def to_dict(self) -> dict:
        return {
            "games_seen": self.games_seen,
            "kept": self.kept,
            "duplicates": self.duplicates,
            "rejected": dict(sorted(self.rejected.items())),
            "elo_buckets": dict(sorted(self.elo_buckets.items())),
            "time_controls": dict(sorted(self.time_controls.items())),
            "openings": dict(self.openings.most_common(20)),
            "results": dict(sorted(self.results.items())),
            "shards": [
                {"name": s.name, "rows": s.rows, "bytes": s.bytes, "sha256": s.sha256}
                for s in self.shards
            ],
            "shard_rows": self.shard_rows,
            "parse_skipped": self.parse_skipped,
            "truncated": self.truncated,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Processing QA report",
            "",
            f"- games seen: {self.games_seen}",
            f"- games kept: {self.kept}",
            f"- duplicates dropped: {self.duplicates}",
            f"- parse-skipped: {self.parse_skipped}",
            f"- truncated input: {self.truncated}",
            f"- shards: {len(self.shards)} ({self.shard_rows} rows)",
            "",
            "## Rejections",
            "",
        ]
        lines += [f"- {reason}: {count}" for reason, count in sorted(self.rejected.items())]
        lines += ["", "## Elo buckets", ""]
        lines += [f"- {b}: {c}" for b, c in sorted(self.elo_buckets.items())]
        lines += ["", "## Time controls", ""]
        lines += [f"- {t}: {c}" for t, c in sorted(self.time_controls.items())]
        lines += ["", "## Openings (top 20)", ""]
        lines += [f"- {o}: {c}" for o, c in self.openings.most_common(20)]
        lines += ["", "## Results", ""]
        lines += [f"- {r}: {c}" for r, c in sorted(self.results.items())]
        lines += ["", "## Shards", ""]
        lines += [
            f"- {s.name}: {s.rows} rows, {s.bytes} B, sha256={s.sha256[:12]}…" for s in self.shards
        ]
        return "\n".join(lines) + "\n"


def _game_record(game: chess.pgn.Game) -> dict:
    return {
        "white": game.headers.get("White", "?"),
        "black": game.headers.get("Black", "?"),
        "white_elo": game.headers.get("WhiteElo", "?"),
        "black_elo": game.headers.get("BlackElo", "?"),
        "result": game.headers.get("Result", "*"),
        "elo_bucket": game_elo_bucket(game),
        "time_control": game.headers.get("TimeControl", ""),
        "opening": game.headers.get("Opening", ""),
        "eco": game.headers.get("ECO", ""),
        "moves": game.end().ply(),
        "site": game.headers.get("Site", ""),
    }


def shard_games(
    games: Iterable[chess.pgn.Game],
    out_dir: Path,
    shard_size: int = 10_000,
    *,
    report: QAReport | None = None,
    game_filter: GameFilter | None = None,
    dedup: Deduplicator | None = None,
) -> QAReport:
    """Filter, dedup and write games to ``jsonl.zst`` shards; return the QA report.

    Each shard is written to ``<name>.part`` and renamed into place only after its bytes are
    flushed, so a crash never leaves a half-written shard under its final name.
    """
    report = report if report is not None else QAReport()
    game_filter = game_filter if game_filter is not None else GameFilter()
    dedup = dedup if dedup is not None else Deduplicator()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buffer: list[dict] = []

    def flush() -> None:
        nonlocal shard_idx, buffer
        if not buffer:
            return
        name = f"shard-{shard_idx:05d}.jsonl.zst"
        path = out_dir / name
        tmp = path.with_suffix(path.suffix + ".part")
        payload = ("\n".join(json.dumps(r) for r in buffer) + "\n").encode()
        compressed = zstd.ZstdCompressor(level=10).compress(payload)
        with open(tmp, "wb") as fh:
            fh.write(compressed)
        tmp.replace(path)  # atomic
        report.shards.append(
            ShardInfo(
                name=name, rows=len(buffer), bytes=len(compressed), sha256=_sha256_bytes(compressed)
            )
        )
        shard_idx += 1
        buffer = []

    for game in games:
        report.games_seen += 1
        accepted, reason = game_filter.accepts(game)
        if not accepted:
            report.rejected[reason] += 1
            continue
        if not dedup.is_new(game):
            continue
        report.kept += 1
        report.elo_buckets[game_elo_bucket(game)] += 1
        report.time_controls[time_control_category(game.headers.get("TimeControl", ""))] += 1
        opening = game.headers.get("Opening", "") or game.headers.get("ECO", "") or "unknown"
        report.openings[opening] += 1
        report.results[game.headers.get("Result", "*")] += 1
        buffer.append(_game_record(game))
        if len(buffer) >= shard_size:
            flush()
    flush()
    report.duplicates = dedup.duplicates
    return report
