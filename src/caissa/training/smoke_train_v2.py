"""`make smoke-train-v2` — the scaled vertical (session-0008, deep lane #2, ADR-0011).

Session-0007 proved the composition on real data with `numpy_tiny` (K=0, linear head, d_model
implicit). This experiment asks a sharper question on the **same** day-prefix: does raising the
model's capacity and giving it move **history** raise top-1 move accuracy — and if so, which organ
pays for it?

The answer is decided by the pre-registered thresholds in
``LOG/session-0008-preregistration.md``, written **before** dispatch:

    SUCCESS       top-1 accuracy >= 0.30
    DEAD          top-1 accuracy <= the most-common-move baseline
    INCONCLUSIVE  strictly between (interesting, not convincing)
    INVALID       sampling legality < 100% (a defect to fix, not a finding)

The same file also carries the two ablations' questions and thresholds. They are run **in this
script**, from the same shards and the same split seed, so a component's contribution is a
controlled difference and not two runs compared across time. Ablations run at a reduced position
budget — stated here and in the log, because they answer a *relative* question about a component
and an absolute accuracy number from a smaller sample would be a weaker claim than the one being
made.

Weights are a disposable run artifact, never a git object (Article 8).
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

from caissa.model.numpy_history import NumpyHistoryModel
from caissa.model.numpy_tiny import encode
from caissa.training.dataset import (
    HistorySample,
    Position,
    ShardStats,
    build_history_samples,
    count_short_games,
    load_games,
    move_index,
    split_games,
)


def _merge(base: dict, overrides: dict) -> dict:
    """Deep-merge ``overrides`` into ``base`` (one level of nesting is all the configs use)."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out


def _build_model(cfg: dict) -> NumpyHistoryModel:
    model_cfg = cfg.get("model") or {}
    return NumpyHistoryModel(
        history=int(cfg.get("history", 8)),
        d_model=int(model_cfg.get("d_model", 256)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        geometry_bias=bool(model_cfg.get("geometric_bias", False)),
        seed=int(cfg.get("seed", 0)),
    )


def _hist_matrix(sample: HistorySample, history: int) -> np.ndarray:
    """Encode the LAST ``history + 1`` boards of a sample's window.

    The window is built once with the MAIN run's ``K`` so every configuration is scored on the
    SAME positions (session-0009 preflight: the eval set is the controlled comparison surface).
    A model with a smaller ``K`` consumes only the suffix it can read — ``K=0`` sees the current
    board alone — so an ablation changes the model's input, never the set it is measured on.
    """
    boards = sample.history[-(history + 1) :]
    return np.stack([encode(board) for board in boards], axis=0)


def _accuracy(model: NumpyHistoryModel, samples: list[HistorySample], history: int) -> float:
    """Top-1 (from, to) accuracy of the masked policy over ``samples``."""
    if not samples:
        return 0.0
    hits = 0
    for sample in samples:
        probs = model.masked_probs(sample.board, _hist_matrix(sample, history))
        if int(np.argmax(probs)) == move_index(sample.move):
            hits += 1
    return hits / len(samples)


def _mean_loss(model: NumpyHistoryModel, samples: list[HistorySample], history: int) -> float:
    """Mean masked cross-entropy over ``samples`` without touching the weights."""
    if not samples:
        return float("nan")
    return sum(model.loss(s.board, _hist_matrix(s, history), s.move) for s in samples) / len(
        samples
    )


def _train_once(cfg: dict, games: list[list[Position]], *, label: str) -> dict:
    """Train one configuration end to end and return its metrics dict."""
    history = int(cfg.get("history", 8))
    # The window is built with the MAIN run's K for every configuration, so the eval set is the
    # SAME positions for the main run and both ablations (session-0009 preflight). Without this,
    # `no_history` (K=0) would be scored on a different, larger set (ply >= 0) than the main run
    # (ply >= K) — a confounder, not an ablation. `window_history` defaults to `history`, so the
    # main run is unchanged; only a smaller-K ablation reads a suffix of the same window.
    window_history = int(cfg.get("window_history", history))
    val_fraction = float(cfg.get("val_fraction", 0.05))
    seed = int(cfg.get("seed", 0))
    max_train = int(cfg.get("max_train_positions", 60_000))
    max_val = int(cfg.get("max_val_positions", 2_000))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 0.05))
    sample_moves = int(cfg.get("sample_moves", 1_000))
    budget_minutes = int(cfg.get("budget_minutes", 90))
    success_threshold = float(cfg.get("success_threshold", 0.30))

    started = time.time()
    train_games, val_games = split_games(games, val_fraction=val_fraction, seed=seed)
    train = build_history_samples(train_games, history=window_history, limit=max_train)
    val = build_history_samples(val_games, history=window_history, limit=max_val)
    if not train or not val:
        raise SystemExit(
            f"[smoke-train-v2:{label}] the history window (K={window_history}) left no samples. "
            "Lower `history` or use longer games — never pad a window."
        )
    short = count_short_games(games, history=window_history)
    print(
        f"[smoke-train-v2:{label}] K={history} d_model={cfg.get('model', {}).get('d_model')} "
        f"heads={cfg.get('model', {}).get('n_heads')} geom={cfg.get('model', {}).get('geometric_bias')} "
        f"| {len(train_games)} train / {len(val_games)} val games | "
        f"{len(train)} train / {len(val)} val samples | games too short for a window: {short}"
    )

    # --- the baseline: always play the single most common training move ---
    counts = Counter(move_index(s.move) for s in train)
    baseline_index, baseline_count = counts.most_common(1)[0]
    baseline_move = f"{baseline_index // 64}:{baseline_index % 64}"
    baseline_acc = sum(1 for s in val if move_index(s.move) == baseline_index) / max(1, len(val))
    print(
        f"[smoke-train-v2:{label}] baseline: most common train move {baseline_move} "
        f"({baseline_count}/{len(train)}) -> val accuracy {baseline_acc:.4f}"
    )

    model = _build_model(cfg)
    rng = random.Random(seed)
    val_curve: list[float] = [_mean_loss(model, val, history)]
    train_curve: list[float] = []

    for epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        running, seen = 0.0, 0
        for idx in order:
            sample = train[idx]
            running += model.train_step(
                sample.board, _hist_matrix(sample, history), sample.move, lr=lr
            )
            seen += 1
            if seen % max(1, len(order) // 4) == 0:
                print(
                    f"[smoke-train-v2:{label}] epoch {epoch + 1} step {seen}/{len(order)} "
                    f"loss={running / seen:.4f} elapsed={time.time() - started:.0f}s"
                )
        train_curve.append(running / max(1, seen))
        val_curve.append(_mean_loss(model, val, history))
        print(
            f"[smoke-train-v2:{label}] epoch {epoch + 1} done: train={train_curve[-1]:.4f} "
            f"val={val_curve[-1]:.4f}"
        )

    val_loss_first, val_loss_last = val_curve[0], val_curve[-1]
    top1 = _accuracy(model, val, history)
    loss_decreasing = val_loss_last < val_loss_first

    # --- the Article 1 gate: 1000 top-1 moves must be 100% legal ---
    # Deterministic top-1, not a sampled draw: from a peaked softmax a sample is the argmax almost
    # always, so a "100% legal" claim built on sampling would be true but weak. This is the exact
    # move the policy plays, and it must be legal on every distinct position it sees (not a
    # reshuffled window) — hence the dedup by FEN.
    sampled = legal = 0
    seen_fens: set[str] = set()
    for sample in train:
        board = sample.board
        if board.is_game_over() or board.fen() in seen_fens:
            continue
        seen_fens.add(board.fen())
        move = model.argmax_move(board, _hist_matrix(sample, history))
        sampled += 1
        if move in board.legal_moves:
            legal += 1
        if sampled >= sample_moves:
            break
    legality_rate = legal / max(1, sampled)

    elapsed = time.time() - started
    within_budget = elapsed <= budget_minutes * 60

    if legality_rate < 1.0:
        verdict = "INVALID"
        reason = f"sampling legality {legal}/{sampled} is not 100% (Article 1)"
    elif top1 >= success_threshold:
        verdict = "SUCCESS"
        reason = f"top-1 {top1:.4f} >= the pre-registered {success_threshold:.2f}"
    elif top1 <= baseline_acc:
        verdict = "DEAD"
        reason = f"top-1 {top1:.4f} <= baseline {baseline_acc:.4f}"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"top-1 {top1:.4f} beats the baseline but is below {success_threshold:.2f}"

    return {
        "label": label,
        "config": {
            "history": history,
            "d_model": int((cfg.get("model") or {}).get("d_model", 256)),
            "n_heads": int((cfg.get("model") or {}).get("n_heads", 8)),
            "geometric_bias": bool((cfg.get("model") or {}).get("geometric_bias", False)),
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "max_train_positions": max_train,
            "val_fraction": val_fraction,
        },
        "data": {
            "train_games": len(train_games),
            "val_games": len(val_games),
            "train_samples": len(train),
            "val_samples": len(val),
            "games_too_short_for_window": short,
        },
        "baseline": {"move": baseline_move, "val_accuracy": baseline_acc},
        "model": {
            "name": "numpy_history",
            "val_loss_first": val_loss_first,
            "val_loss_last": val_loss_last,
            "val_loss_curve": val_curve,
            "train_loss_curve": train_curve,
            "top1_accuracy": top1,
            "loss_decreasing": loss_decreasing,
        },
        "legality": {"sampled": sampled, "legal": legal, "rate": legality_rate},
        "verdict": {"status": verdict, "reason": reason, "within_budget": within_budget},
        "elapsed_seconds": round(elapsed, 2),
    }


def run(config_path: str) -> dict:
    """Train the main configuration, then each ablation, and write both artifacts."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    shards_dir = cfg.get("shards", "data/shards")
    metrics_path = Path(cfg.get("metrics", "data/qa/metrics-v2.json"))
    ablations_path = Path(cfg.get("ablations_out", "data/qa/ablations-v2.json"))
    history = int(cfg.get("history", 8))
    max_load = int(cfg.get("max_positions_to_load", 200_000))

    print(f"[smoke-train-v2] config={config_path} shards={shards_dir} K={history}")
    stats = ShardStats(shards=sorted(p.name for p in Path(shards_dir).glob("*.jsonl.zst")))
    games = load_games(shards_dir, max_positions=max_load, min_moves=history + 2, stats=stats)
    if not games:
        raise SystemExit(
            f"[smoke-train-v2] no usable games under {shards_dir!r}. Run the data sample and the "
            "sharder first (see docs/TRAINING.md)."
        )
    print(
        f"[smoke-train-v2] loaded {stats.games} games / {stats.positions} positions "
        f"(broken replays={stats.broken_replays} skipped={stats.skipped_games})"
    )

    main = _train_once(cfg, games, label="v2")
    main["schema"] = "caissa-smoke-train-v2-metrics/1"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(main, indent=2, sort_keys=True))

    print(
        f"[smoke-train-v2] MAIN: val loss {main['model']['val_loss_first']:.4f} -> "
        f"{main['model']['val_loss_last']:.4f} | top-1 {main['model']['top1_accuracy']:.4f} "
        f"vs baseline {main['baseline']['val_accuracy']:.4f} | "
        f"legality {main['legality']['legal']}/{main['legality']['sampled']} | "
        f"VERDICT: {main['verdict']['status']}"
    )
    print(f"[smoke-train-v2] metrics written to {metrics_path}")

    ablation_specs = cfg.get("ablations") or []
    if not ablation_specs:
        return main

    results_ablations: list[dict] = []
    results: dict = {
        "schema": "caissa-smoke-train-v2-ablations/1",
        "baseline": main,
        "ablations": results_ablations,
    }
    for spec in ablation_specs:
        name = spec.get("name")
        overrides = spec.get("overrides") or {}
        if not name:
            raise SystemExit("[smoke-train-v2] an ablation entry has no `name`")
        print(
            f"[smoke-train-v2] ablation '{name}': overrides={json.dumps(overrides, sort_keys=True)}"
        )
        run_cfg = _merge(cfg, overrides)
        result = _train_once(run_cfg, games, label=str(name))
        result["delta_top1_vs_main"] = (
            result["model"]["top1_accuracy"] - main["model"]["top1_accuracy"]
        )
        results_ablations.append(result)
        print(
            f"[smoke-train-v2] ablation '{name}': top-1 {result['model']['top1_accuracy']:.4f} "
            f"(delta {result['delta_top1_vs_main']:+.4f}) verdict {result['verdict']['status']}"
        )

    ablations_path.parent.mkdir(parents=True, exist_ok=True)
    ablations_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[smoke-train-v2] ablations written to {ablations_path}")
    return main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 smoke-train-v2 (deep lane #2)")
    parser.add_argument("--config", default="configs/smoke-train-v2.yaml")
    args = parser.parse_args(argv)
    metrics = run(args.config)
    # INVALID is a defect, not a finding: fail the run so it cannot be read as a result.
    return 1 if metrics["verdict"]["status"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
