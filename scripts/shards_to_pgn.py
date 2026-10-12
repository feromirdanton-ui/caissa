"""Reconstruct a PGN from jsonl.zst shards, so the labeler can consume shard data.

The labeler (`caissa-label`) takes a PGN path; the Phase-2 pipeline produces shards. This bridge
replays each shard record's UCI moves into a `chess.pgn.Game` and writes standard PGN, so the
fast-lane labeling run measures the SAME games the vertical trains on — not a separate fixture.

Usage: python scripts/shards_to_pgn.py <shards_dir> <out.pgn> [--max-games N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caissa.training.dataset import iter_shard_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="shards -> PGN bridge for the labeler")
    parser.add_argument("shards_dir")
    parser.add_argument("out")
    parser.add_argument("--max-games", type=int, default=0, help="0 = all")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w", encoding="utf-8") as fh:
        for record in iter_shard_records(args.shards_dir):
            uci_moves = record.get("uci") or []
            if not uci_moves:
                continue
            start_fen = record.get("start_fen") or ""
            board = chess.Board(start_fen) if start_fen else chess.Board()
            game = chess.pgn.Game()
            game.headers["White"] = record.get("white", "?")
            game.headers["Black"] = record.get("black", "?")
            game.headers["WhiteElo"] = record.get("white_elo", "?")
            game.headers["BlackElo"] = record.get("black_elo", "?")
            game.headers["Result"] = record.get("result", "*")
            game.headers["TimeControl"] = record.get("time_control", "")
            game.headers["Opening"] = record.get("opening", "")
            game.headers["ECO"] = record.get("eco", "")
            game.headers["Site"] = record.get("site", "")
            if start_fen:
                game.headers["FEN"] = start_fen
            node = game
            ok = True
            for uci in uci_moves:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    ok = False
                    break
                if move not in board.legal_moves:
                    ok = False
                    break
                node = node.add_variation(move)
                board.push(move)
            if not ok:
                continue
            print(game, file=fh, end="\n\n")
            written += 1
            if args.max_games and written >= args.max_games:
                break
    print(f"[shards_to_pgn] wrote {written} games to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
