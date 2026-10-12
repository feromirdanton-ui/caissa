"""The guarded training loop and the pre-registered verdict (session-0010).

Split out of `smoke_train_v2.py` so each published blob stays small enough to move over the GitHub
API without a transcription slip (ADR-0009). This module owns the two things the reload added:

  * the NaN guard wrapping the epoch loop -- the first non-finite loss or parameter aborts the run
    RED and writes a FAILURE artifact, so a diverged run can never be read as a green non-result
    (the exact state session-0009 fell into);
  * the verdict, decided INVALID first (non-finite metric, or legality < 100%), and only then
    SUCCESS / DEAD / INCONCLUSIVE against the pre-registered thresholds.
"""

from __future__ import annotations

import math
import random
import sys
import time

from caissa.model.numpy_history import NumpyHistoryModel
from caissa.training.smoke_metrics import (
    DivergenceError,
    accuracy_v2,
    cross_ruler_baseline,
    mean_loss_v2,
    parameters_finite,
    samples_matrix,
    set_local_baseline,
    write_failure,
)


def build_model(cfg: dict) -> NumpyHistoryModel:
    model_cfg = cfg.get("model") or {}
    return NumpyHistoryModel(
        history=int(cfg.get("history", 8)),
        d_model=int(model_cfg.get("d_model", 256)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        geometry_bias=bool(model_cfg.get("geometric_bias", False)),
        seed=int(cfg.get("seed", 0)),
    )


def train_configuration(
    prepared, cfg: dict, *, label: str, failure_dir, model_factory=None
) -> dict:
    """Train one configuration end to end and return its metrics dict (or abort RED on divergence).

    `model_factory` is the seam the mutation-control test drives: it can inject a poisoned model so
    a divergence can be produced on purpose. Defaults to the real `build_model`.
    """
    history = int(cfg.get("history", 8))
    seed = int(cfg.get("seed", 0))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 0.01))
    sample_moves = int(cfg.get("sample_moves", 1000))
    budget_minutes = int(cfg.get("budget_minutes", 90))
    success_threshold = float(cfg.get("success_threshold", 0.30))
    guard_every = int(cfg.get("guard_every", 200))

    started = time.time()
    train, val = prepared.train_samples, prepared.val_samples
    model_cfg = cfg.get("model") or {}
    print(
        f"[smoke-train-v2:{label}] K={history} d_model={model_cfg.get('d_model')} "
        f"heads={model_cfg.get('n_heads')} geom={model_cfg.get('geometric_bias')} lr={lr} | "
        f"{len(prepared.train_games)} train / {len(prepared.val_games)} val games | "
        f"{len(train)} train / {len(val)} val samples | short: {prepared.short_games}"
    )

    baseline_move, baseline_acc, _ = set_local_baseline(val)
    cross_move, cross_acc = cross_ruler_baseline(train, val)
    print(
        f"[smoke-train-v2:{label}] baseline (set-local): {baseline_move} -> {baseline_acc:.4f}; "
        f"cross-ruler (train move {cross_move} on val) {cross_acc:.4f}"
    )

    model = (model_factory or build_model)(cfg)
    rng = random.Random(seed)
    val_curve: list[float] = [mean_loss_v2(model, val, history)]
    train_curve: list[float] = []
    seen_total = 0

    try:
        for epoch in range(epochs):
            order = list(range(len(train)))
            rng.shuffle(order)
            running, seen = 0.0, 0
            for idx in order:
                sample = train[idx]
                loss = model.train_step(
                    sample.board, samples_matrix(sample, history), sample.move, lr=lr
                )
                if not math.isfinite(loss):
                    raise DivergenceError(
                        f"non-finite loss ({loss}) at epoch {epoch + 1} step {seen + 1} -- the "
                        "model has diverged; INVALID, never a finding (session-0010)"
                    )
                running += loss
                seen += 1
                seen_total += 1
                # Every N steps, also check the WEIGHTS: a finite loss this step does not prove
                # finite weights -- a large update can push a weight to inf and the NEXT step would
                # be the first to notice.
                if seen % max(1, guard_every) == 0 and not parameters_finite(model):
                    raise DivergenceError(
                        f"non-finite parameters at epoch {epoch + 1} step {seen} -- diverged"
                    )
                if seen % max(1, len(order) // 4) == 0:
                    print(
                        f"[smoke-train-v2:{label}] epoch {epoch + 1} step {seen}/{len(order)} "
                        f"loss={running / seen:.4f} elapsed={time.time() - started:.0f}s"
                    )
            train_curve.append(running / max(1, seen))
            val_curve.append(mean_loss_v2(model, val, history))
            print(
                f"[smoke-train-v2:{label}] epoch {epoch + 1} done: train={train_curve[-1]:.4f} "
                f"val={val_curve[-1]:.4f}"
            )
    except DivergenceError as exc:
        partial = {
            "epochs_completed": len(train_curve),
            "train_loss_curve": train_curve,
            "val_loss_curve": list(val_curve),
            "steps": seen_total,
        }
        write_failure(failure_dir / f"FAILURE-{label}.json", label, exc, partial)
        print(f"::error::[smoke-train-v2:{label}] DIVERGED -- {exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    val_first, val_last = val_curve[0], val_curve[-1]
    top1 = accuracy_v2(model, val, history)

    # --- the Article 1 gate: 1000 top-1 moves must be 100% legal on distinct positions ---
    sampled = legal = 0
    seen_fens: set[str] = set()
    for sample in train:
        board = sample.board
        if board.is_game_over() or board.fen() in seen_fens:
            continue
        seen_fens.add(board.fen())
        move = model.argmax_move(board, samples_matrix(sample, history))
        sampled += 1
        if move in board.legal_moves:
            legal += 1
        if sampled >= sample_moves:
            break
    legality = legal / max(1, sampled)

    elapsed = time.time() - started
    within_budget = elapsed <= budget_minutes * 60

    # A non-finite metric of ANY kind is INVALID first: the instrument is destroyed, so
    # SUCCESS/DEAD/INCONCLUSIVE are all meaningless (the branch session-0009 lacked).
    non_finite = not all(
        math.isfinite(x)
        for x in (top1, baseline_acc, val_first, val_last, *val_curve, *train_curve)
    )
    if non_finite:
        verdict, reason = "INVALID", "a non-finite metric -- the instrument is destroyed"
    elif legality < 1.0:
        verdict, reason = "INVALID", f"legality {legal}/{sampled} is not 100% (Article 1)"
    elif top1 >= success_threshold:
        verdict, reason = "SUCCESS", f"top-1 {top1:.4f} >= pre-registered {success_threshold:.2f}"
    elif top1 <= baseline_acc:
        verdict, reason = "DEAD", f"top-1 {top1:.4f} <= set-local baseline {baseline_acc:.4f}"
    else:
        verdict, reason = (
            "INCONCLUSIVE",
            f"top-1 {top1:.4f} beats the baseline but is below {success_threshold:.2f}",
        )

    return {
        "label": label,
        "config": {
            "history": history,
            "window_history": int(cfg.get("window_history", history)),
            "d_model": int(model_cfg.get("d_model", 256)),
            "n_heads": int(model_cfg.get("n_heads", 8)),
            "geometric_bias": bool(model_cfg.get("geometric_bias", False)),
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "max_train_positions": int(cfg.get("max_train_positions", 60000)),
            "val_fraction": float(cfg.get("val_fraction", 0.05)),
        },
        "data": {
            "train_games": len(prepared.train_games),
            "val_games": len(prepared.val_games),
            "train_samples": len(train),
            "val_samples": len(val),
            "games_too_short_for_window": prepared.short_games,
        },
        "baseline": {
            "move": baseline_move,
            "val_accuracy": baseline_acc,
            "ruler": "set_local",
            "cross_ruler_move": cross_move,
            "cross_ruler_accuracy": cross_acc,
        },
        "model": {
            "name": "numpy_history",
            "val_loss_first": val_first,
            "val_loss_last": val_last,
            "val_loss_curve": val_curve,
            "train_loss_curve": train_curve,
            "top1_accuracy": top1,
            "loss_decreasing": val_last < val_first,
        },
        "legality": {"sampled": sampled, "legal": legal, "rate": legality},
        "verdict": {"status": verdict, "reason": reason, "within_budget": within_budget},
        "elapsed_seconds": round(elapsed, 2),
    }
