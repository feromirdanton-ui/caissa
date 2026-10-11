"""`make smoke` — CPU micro-benchmark (GOAL.md §10).

DoD: CPU <=10 min, loss decreasing, 1000 sampled moves 100% legal.

This trains the tiny numpy model on a deterministic teacher signal over random positions.
It is a REAL training loop on a REAL (tiny) model — not a mock. It produces no claim about
chess strength; it only proves the pipeline is wired and the mask forbids illegal moves.

Every position that is already finished is skipped rather than counted, so the reported
"legal moves" figure is over moves that were actually sampled (a finished position has no
move to sample and used to inflate the percentage).
"""

from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np
import yaml

from caissa.model.numpy_tiny import NumpyTinyModel, teacher_move


def _random_position(rng: random.Random, plies: int) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        board.push(rng.choice(moves))
    return board


def run(config_path: str) -> int:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    steps = int(cfg.get("steps", 100))
    sample_moves = int(cfg.get("sample_moves", 1000))
    seed = int(cfg.get("seed", 0))
    dataset_size = int(cfg.get("dataset_size", 64))

    rng = random.Random(seed)
    model = NumpyTinyModel(seed=seed)

    print(f"[smoke] config={config_path} steps={steps} sample_moves={sample_moves}")
    print("[smoke] model=numpy_tiny (REAL tiny model, CPU fallback — not a mock)")

    # A FIXED dataset so the loss is a real learning signal, not sampling noise.
    dataset: list[tuple[chess.Board, chess.Move]] = []
    while len(dataset) < dataset_size:
        board = _random_position(rng, plies=rng.randint(0, 40))
        if board.is_game_over():
            continue
        dataset.append((board, teacher_move(board)))

    def mean_loss() -> float:
        return sum(model.train_step(b, m, lr=0.0) for b, m in dataset) / len(dataset)

    t0 = time.time()
    first = mean_loss()
    for step in range(steps):
        board, target = dataset[step % len(dataset)]
        model.train_step(board, target)
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[smoke] step {step + 1}/{steps} loss={mean_loss():.4f}")

    elapsed = time.time() - t0
    last = mean_loss()
    decreasing = last < first

    # Sample moves and verify legality — the Article 1 guarantee.
    sampled = 0
    legal = 0
    while sampled < sample_moves:
        board = _random_position(rng, plies=rng.randint(0, 40))
        if board.is_game_over():
            continue  # nothing to sample here: do not count it as a legal move
        move = model.sample_move(board, np.random.default_rng(rng.randint(0, 1 << 30)))
        sampled += 1
        if move in board.legal_moves:
            legal += 1

    print(
        f"[smoke] elapsed={elapsed:.1f}s loss_first={first:.4f} loss_last={last:.4f} "
        f"decreasing={decreasing}"
    )
    print(f"[smoke] legal moves: {legal}/{sampled} ({100.0 * legal / sampled:.1f}%)")

    ok = decreasing and legal == sampled and elapsed <= 600
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 CPU smoke test")
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
