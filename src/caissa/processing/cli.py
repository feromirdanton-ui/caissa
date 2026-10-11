"""`caissa-shard` console script."""

from __future__ import annotations

import argparse
from pathlib import Path

from caissa.processing.pgn import iter_games
from caissa.processing.sharding import shard_games


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 PGN sharder")
    parser.add_argument("pgn", help="path to a plain-text PGN file")
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args(argv)
    report = shard_games(iter_games(args.pgn), Path(args.out), args.shard_size)
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
