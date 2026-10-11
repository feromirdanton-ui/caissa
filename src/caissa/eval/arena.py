"""Arena stub vs Stockfish + report (GOAL.md §11 task 10).

STATUS: wiring stub, honestly labelled. Both sides currently play uniformly random legal moves,
so the W/D/L figure measures nothing about strength; every report is tagged "MOCK — not a result"
(Article 5). The `opponent` field is recorded and reported but does not yet influence play — a real
Stockfish path is not implemented (docs/ISSUES-0002.md P1-5).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import chess
import yaml


@dataclass
class ArenaResult:
    games: int
    engine: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    mock: bool = True

    def to_markdown(self) -> str:
        tag = "MOCK — not a result" if self.mock else "real engine"
        return (
            f"# Arena report ({tag})\n\n"
            f"- opponent: {self.engine}\n"
            f"- games: {self.games}\n"
            f"- W/D/L: {self.wins}/{self.draws}/{self.losses}\n"
        )


def _random_opponent_move(board: chess.Board, rng: random.Random) -> chess.Move:
    return rng.choice(list(board.legal_moves))


def play_game(opponent: str, rng: random.Random, max_plies: int = 200) -> str:
    """Play one game: both sides move randomly (stub); `opponent` is recorded, not consulted.

    Returns 'win' | 'draw' | 'loss' from our (white) perspective.
    """
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over():
            break
        board.push(_random_opponent_move(board, rng))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == chess.WHITE else "loss"


def run(config_path: str) -> ArenaResult:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    games = int(cfg.get("games", 20))
    opponent = cfg.get("opponent", "stockfish-level-1")
    rng = random.Random(int(cfg.get("seed", 0)))

    result = ArenaResult(games=games, engine=opponent, mock=True)
    for _ in range(games):
        outcome = play_game(opponent, rng)
        if outcome == "win":
            result.wins += 1
        elif outcome == "loss":
            result.losses += 1
        else:
            result.draws += 1
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 arena stub")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args(argv)
    result = run(args.config)
    print(result.to_markdown())
    print("[arena] MOCK opponent — this is a wiring test, not a strength result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
