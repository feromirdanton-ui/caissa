"""`caissa-shard` and `caissa-label` console scripts (GOAL.md §5 Phase 2).

``caissa-shard`` is the whole processing pipeline in one command: stream a PGN (plain or
zstd) -> filter (ADR-0010) -> dedup -> shard to ``jsonl.zst`` -> QA report. It never loads
the archive into memory and it never writes data into git.

``caissa-label`` derives blunder labels from a *measured* engine delta. It uses a real
Stockfish when one is available and otherwise the MOCK engine — and a MOCK report says so
in its own output (Article 5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from caissa.guards.counterfactual import Engine
from caissa.processing.filters import GameFilter
from caissa.processing.labeling import LabelReport, default_engine, label_games
from caissa.processing.pgn import ParseStats, iter_games_any
from caissa.processing.sharding import shard_games


def _parse_time_controls(text: str) -> frozenset[str]:
    if not text:
        return frozenset()
    return frozenset(part.strip() for part in text.split(",") if part.strip())


def shard_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="caissa-shard", description="Caïssa-1 PGN sharder")
    parser.add_argument("pgn", help="path to a PGN file (plain or .zst)")
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--qa", default="data/qa/processing.json")
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--max-elo", type=int, default=4000)
    parser.add_argument("--time-controls", default="", help="comma list, e.g. blitz,rapid")
    parser.add_argument("--min-plies", type=int, default=0)
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="skip malformed games and count them instead of failing loudly",
    )
    parser.add_argument(
        "--truncated-ok",
        action="store_true",
        help="tolerate a truncated zstd stream (a byte prefix) and record it",
    )
    args = parser.parse_args(argv)

    stats = ParseStats()
    game_filter = GameFilter(
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        time_controls=_parse_time_controls(args.time_controls),
        min_plies=args.min_plies,
    )
    games = iter_games_any(
        args.pgn,
        strict=not args.lenient,
        stats=stats,
        truncated_ok=args.truncated_ok,
    )
    report = shard_games(games, Path(args.out), args.shard_size, game_filter=game_filter)
    report.parse_skipped = stats.skipped
    report.truncated = stats.truncated

    qa_path = Path(args.qa)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(report.to_markdown())
    print(f"[shard] QA written to {qa_path}")
    return 0


def label_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="caissa-label", description="Caïssa-1 blunder labeler")
    parser.add_argument("pgn", help="path to a PGN file (plain or .zst)")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--out", default="data/qa/labels.json")
    parser.add_argument(
        "--engine",
        choices=["auto", "mock", "stockfish"],
        default="auto",
        help="auto = real Stockfish when present, else MOCK (labeled)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="stop after this many seconds and record the report as PARTIAL (never silent)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write a progress checkpoint every N positions (0 = off)",
    )
    args = parser.parse_args(argv)

    if args.engine == "mock":
        from caissa.guards.counterfactual import MockEngine

        engine: Engine = MockEngine()
    elif args.engine == "stockfish":
        from caissa.guards.counterfactual import StockfishEngine

        engine = StockfishEngine()
    else:
        engine = default_engine()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out.with_suffix(".checkpoint.json")

    def _write(report: LabelReport) -> None:
        out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    def _checkpoint(report: LabelReport) -> None:
        checkpoint_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        eta = report.eta_seconds(max(0, args.limit - report.positions))
        eta_text = f"{eta:.0f}s" if eta is not None else "unknown (rate not yet measured)"
        print(
            f"[label] checkpoint: {report.positions} positions, "
            f"{report.positions_per_second:.2f}/s, ETA {eta_text}"
        )

    stats = ParseStats()
    games = iter_games_any(args.pgn, strict=False, stats=stats, truncated_ok=True)
    report = label_games(
        games,
        engine,
        limit=args.limit,
        timeout_s=args.timeout_s,
        on_checkpoint=_checkpoint if args.checkpoint_every > 0 else None,
        checkpoint_every=args.checkpoint_every,
    )

    _write(report)
    print(report.to_markdown())
    if report.is_mock:
        print("[label] MOCK engine — this is a wiring test, not a strength result.")
    if report.timed_out:
        print(f"[label] PARTIAL report: {report.timeout_reason}")
    print(f"[label] report written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch on the first argument: ``shard`` (default) or ``label``."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] == "label":
        return label_main(argv[1:])
    if argv and argv[0] == "shard":
        return shard_main(argv[1:])
    return shard_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
