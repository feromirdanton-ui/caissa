"""`make smoke-train` — the deep-lane experiment of the «Vertical» milestone (ADR-0011).

The vertical (representation + tiny model) had never been exercised on **real data**. This trains
the REAL tiny model on positions from REAL rated standard Lichess games and records the metrics
that decide the pre-registered criterion (``LOG/session-0007-preregistration.md``):

    ALIVE  ⇔  val loss decreases  AND  top-1 move accuracy strictly beats the baseline
    DEAD   ⇔  a plateau, a rising val loss, or accuracy at/below the baseline
    INVALID ⇔  sampling legality is not 100% (a defect to fix, not a finding)

The baseline is the frequency of the single most common (from, to) move in the **training** split,
measured on the **validation** split — so it is the "always play the most common move" straw man,
not a straw man of our own making.

This is a real training loop, not a mock: SGD, a masked softmax policy head, and a measured
validation. It makes **no claim about chess strength**. Weights are a disposable run artifact and
are never written into git (Article 8).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from caissa.model.numpy_tiny import NumpyTinyModel
from caissa.training.dataset import (
    Position,
    ShardStats,
    flatten,
    load_games,
    move_index,
    split_games,
)


def _accuracy(model: NumpyTinyModel, positions: list[Position]) -> float:
    """Top-1 (from, to) accuracy of the masked policy over ``positions``."""
    if not positions:
        return 0.0
    hits = 0
    for board, move in positions:
        probs = model.masked_probs(board)
        pred = int(np.argmax(probs))
        if pred == move_index(move):
            hits += 1
    return hits / len(positions)


def _mean_loss(model: NumpyTinyModel, positions: list[Position]) -> float:
    """Mean masked cross-entropy over ``positions``, without touching the weights."""
    if not positions:
        return float("nan")
    return sum(model.train_step(b, m, lr=0.0) for b, m in positions) / len(positions)


def run(config_path: str) -> dict:
    """Train, evaluate and return the metrics dict (also written to the QA directory)."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    shards_dir = cfg.get("shards", "data/shards")
    val_fraction = float(cfg.get("val_fraction", 0.05))
    seed = int(cfg.get("seed", 0))
    max_train = int(cfg.get("max_train_positions", 30_000))
    max_val = int(cfg.get("max_val_positions", 2_000))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 0.5))
    sample_moves = int(cfg.get("sample_moves", 1_000))
    budget_minutes = int(cfg.get("budget_minutes", 60))
    metrics_path = Path(cfg.get("metrics", "data/qa/metrics.json"))

    started = time.time()
    print(f"[smoke-train] config={config_path} shards={shards_dir} seed={seed}")

    stats = ShardStats(shards=sorted(p.name for p in Path(shards_dir).glob("*.jsonl.zst")))
    games = load_games(shards_dir, max_positions=max_train + max_val, stats=stats)
    if not games:
        raise SystemExit(
            f"[smoke-train] no positions found under {shards_dir!r}. Run the data sample and the "
            "sharder first (see docs/TRAINING.md)."
        )
    train_games, val_games = split_games(games, val_fraction=val_fraction, seed=seed)
    train = flatten(train_games, limit=max_train)
    val = flatten(val_games, limit=max_val)
    print(
        f"[smoke-train] data: {stats.games} games ({len(train_games)} train / {len(val_games)} val), "
        f"{len(train)} train positions, {len(val)} val positions, "
        f"broken replays={stats.broken_replays} skipped={stats.skipped_games}"
    )

    # --- the baseline: always play the single most common training move ---
    counts = Counter(move_index(move) for _, move in train)
    baseline_index, baseline_count = counts.most_common(1)[0]
    baseline_move = f"{baseline_index // 64}:{baseline_index % 64}"
    baseline_acc = sum(1 for _, mv in val if move_index(mv) == baseline_index) / max(1, len(val))
    print(
        f"[smoke-train] baseline: most common train move {baseline_move} "
        f"({baseline_count}/{len(train)}) → val accuracy {baseline_acc:.4f}"
    )

    # --- train ---
    model = NumpyTinyModel(seed=seed)
    rng = random.Random(seed)
    val_curve: list[float] = []
    train_curve: list[float] = []
    loss_first = _mean_loss(model, val)
    val_curve.append(loss_first)

    for epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        running, seen = 0.0, 0
        for idx in order:
            board, move = train[idx]
            running += model.train_step(board, move, lr=lr)
            seen += 1
            if seen % max(1, len(order) // 5) == 0:
                print(
                    f"[smoke-train] epoch {epoch + 1} step {seen}/{len(order)} loss={running / seen:.4f}"
                )
        epoch_val = _mean_loss(model, val)
        val_curve.append(epoch_val)
        train_curve.append(running / max(1, seen))
        print(
            f"[smoke-train] epoch {epoch + 1} done: train={train_curve[-1]:.4f} val={epoch_val:.4f}"
        )

    val_loss_first = val_curve[0]
    val_loss_last = val_curve[-1]
    top1 = _accuracy(model, val)
    loss_decreasing = val_loss_last < val_loss_first
    beats_baseline = top1 > baseline_acc

    # --- the Article 1 gate: 1000 sampled moves must be 100% legal ---
    sampled = legal = 0
    sample_rng = np.random.default_rng(seed)
    while sampled < sample_moves:
        board, _ = train[sample_rng.integers(0, len(train))]
        if board.is_game_over():
            continue
        move = model.sample_move(board, sample_rng)
        sampled += 1
        if move in board.legal_moves:
            legal += 1
    legality_rate = legal / max(1, sampled)

    elapsed = time.time() - started
    within_budget = elapsed <= budget_minutes * 60

    if legality_rate < 1.0:
        verdict, reason = "INVALID", f"sampling legality {legal}/{sampled} is not 100% (Article 1)"
    elif loss_decreasing and beats_baseline:
        verdict, reason = "ALIVE", "val loss decreases and top-1 accuracy beats the baseline"
    else:
        verdict, reason = (
            "DEAD",
            (
                "val loss did not decrease"
                if not loss_decreasing
                else "accuracy did not beat baseline"
            ),
        )

    metrics = {
        "schema": "caissa-smoke-train-metrics/1",
        "config": config_path,
        "seed": seed,
        "data": {
            "shards": stats.shards,
            "games": stats.games,
            "train_games": len(train_games),
            "val_games": len(val_games),
            "train_positions": len(train),
            "val_positions": len(val),
            "broken_replays": stats.broken_replays,
            "skipped_games": stats.skipped_games,
        },
        "baseline": {"move": baseline_move, "val_accuracy": baseline_acc},
        "model": {
            "name": "numpy_tiny",
            "val_loss_first": val_loss_first,
            "val_loss_last": val_loss_last,
            "val_loss_curve": val_curve,
            "train_loss_curve": train_curve,
            "top1_accuracy": top1,
            "epochs": epochs,
            "lr": lr,
        },
        "legality": {"sampled": sampled, "legal": legal, "rate": legality_rate},
        "verdict": {"status": verdict, "reason": reason, "within_budget": within_budget},
        "elapsed_seconds": round(elapsed, 2),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print(
        f"[smoke-train] val loss {val_loss_first:.4f} → {val_loss_last:.4f} "
        f"(decreasing={loss_decreasing})"
    )
    print(f"[smoke-train] top-1 {top1:.4f} vs baseline {baseline_acc:.4f} (beats={beats_baseline})")
    print(f"[smoke-train] legality {legal}/{sampled} ({100.0 * legality_rate:.1f}%)")
    print(f"[smoke-train] elapsed={elapsed:.1f}s within_budget={within_budget}")
    print(f"[smoke-train] VERDICT: {verdict} — {reason}")
    print(f"[smoke-train] metrics written to {metrics_path} (weights are not persisted: Article 8)")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 smoke-train (deep lane)")
    parser.add_argument("--config", default="configs/smoke-train.yaml")
    args = parser.parse_args(argv)
    metrics = run(args.config)
    # INVALID is a defect, not a finding: it fails the run so it cannot be mistaken for a result.
    return 1 if metrics["verdict"]["status"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
