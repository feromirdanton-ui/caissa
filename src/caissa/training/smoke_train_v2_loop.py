"""The guarded training loop and the pre-registered verdict (session-0010; plateau in 0011).

This module owns the three things the reload and the duel added:

  * the **NaN guard** wrapping the epoch loop -- the first non-finite loss or parameter aborts the
    run RED and writes a FAILURE artifact, so a diverged run can never be read as a green
    non-result (the exact state session-0009 fell into);
  * the **plateau budget** (session-0011): a configuration trains until its validation loss stops
    improving for `plateau_patience` evaluations, so two configurations are compared at their own
    convergence limits rather than at a shared stopwatch reading;
  * the **verdict**, decided INVALID first (non-finite metric, or legality < 100%), and only then
    SUCCESS / DEAD / INCONCLUSIVE against the pre-registered thresholds.

`Adam` is used when the config names it; plain SGD remains available and is what the v1 reference
uses. The gradient is the same object either way -- the optimizer is a rule for consuming it, not a
second implementation of the model.
"""

from __future__ import annotations

import math
import random
import sys
import time

from caissa.model.numpy_history import NumpyHistoryModel
from caissa.training.adam import AdamState
from caissa.training.smoke_metrics import (
    DivergenceError,
    PlateauTracker,
    accuracy_v2,
    correctness_v2,
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


def _train_epochs(prepared, cfg: dict, *, label: str, started: float):
    """Run the plateau-bounded training loop; raise DivergenceError on a non-finite state.

    Returns ``(train_curve, val_curve, steps_total, epochs_run, tracker, grad_norm_curve)``.

    The guard has two nets, deliberately: the loss of the step just taken, and -- every
    `guard_every` steps -- the parameters themselves. A finite loss does not prove finite weights;
    a large update can push a weight to inf and only the NEXT step would notice.

    The initial evaluation (the loss of the untrained model) COUNTS in the budget, and
    `min_evals` is therefore counted the same way: "6 evaluations" means the untrained reading plus
    five epoch readings. Counting it any other way would let a configuration stop after five epoch
    evaluations while the config claims six, and the recorded `evaluations` would not match the
    patience the reader was promised.
    """
    history = int(cfg.get("history", 8))
    seed = int(cfg.get("seed", 0))
    max_epochs = int(cfg.get("max_epochs", 20))
    patience = int(cfg.get("plateau_patience", 5))
    eps = float(cfg.get("plateau_eps", 1e-4))
    min_evals = int(cfg.get("min_evals", 3))
    lr = float(cfg.get("lr", 0.005))
    guard_every = int(cfg.get("guard_every", 200))
    optimizer = str(cfg.get("optimizer", "adam")).lower()

    train, val = prepared.train_samples, prepared.val_samples
    # The v2 model reads this configuration's K; the EVAL SET was built with the main run's K.
    model = build_model(cfg)
    adam = AdamState(betas=tuple(cfg.get("betas", (0.9, 0.999)))) if optimizer == "adam" else None

    rng = random.Random(seed)
    val_curve: list[float] = [mean_loss_v2(model, val, history)]
    train_curve: list[float] = []
    grad_norms: list[float] = []
    tracker = PlateauTracker(patience=patience, eps=eps, min_evals=min_evals)
    tracker.update(val_curve[-1])
    seen_total = 0
    epochs_run = 0

    while epochs_run < max_epochs:
        order = list(range(len(train)))
        rng.shuffle(order)
        running, seen = 0.0, 0
        for idx in order:
            sample = train[idx]
            x_hist = samples_matrix(sample, history)
            if adam is not None:
                loss, norm = model.adam_step(sample.board, x_hist, sample.move, adam, lr)
                grad_norms.append(norm)
            else:
                loss = model.train_step(sample.board, x_hist, sample.move, lr=lr)
            if not math.isfinite(loss):
                raise DivergenceError(
                    f"non-finite loss ({loss}) at epoch {epochs_run + 1} step {seen + 1} -- the "
                    "model has diverged; INVALID, never a finding (session-0010)"
                )
            running += loss
            seen += 1
            seen_total += 1
            if seen % max(1, guard_every) == 0 and not parameters_finite(model):
                raise DivergenceError(
                    f"non-finite parameters at epoch {epochs_run + 1} step {seen} -- diverged"
                )
            if seen % max(1, len(order) // 4) == 0:
                norm_txt = f" grad_norm={grad_norms[-1]:.3e}" if grad_norms else ""
                print(
                    f"[smoke-train-v2:{label}] epoch {epochs_run + 1} step {seen}/{len(order)} "
                    f"loss={running / seen:.4f}{norm_txt} elapsed={time.time() - started:.0f}s"
                )
        epochs_run += 1
        train_curve.append(running / max(1, seen))
        val_curve.append(mean_loss_v2(model, val, history))
        print(
            f"[smoke-train-v2:{label}] epoch {epochs_run} done: train={train_curve[-1]:.4f} "
            f"val={val_curve[-1]:.4f} best={tracker.best:.4f}"
        )
        if tracker.update(val_curve[-1]):
            print(
                f"[smoke-train-v2:{label}] PLATEAU after {epochs_run} epochs "
                f"({tracker.since_improvement} evaluations without >{eps} improvement)"
            )
            break

    return model, train_curve, val_curve, seen_total, epochs_run, tracker, grad_norms


def train_configuration(
    prepared,
    cfg: dict,
    *,
    label: str,
    failure_dir,
    model_factory=None,
    optimizer_override: str | None = None,
) -> dict:
    """Train one configuration to its plateau and return its metrics dict (or abort RED).

    `model_factory` is the seam the mutation-control test drives: it can inject a poisoned model so
    a divergence can be produced on purpose. Defaults to the real `build_model`.

    The returned dict carries a private ``_correctness`` list (the ordered top-1 hits on the eval
    set) which the driver pops out to run McNemar. It is not JSON-serialisable state, it is the raw
    pairing, kept out of the artifact because ten thousand booleans per configuration belong in the
    test, not in the published summary.
    """
    history = int(cfg.get("history", 8))
    seed = int(cfg.get("seed", 0))
    optimizer = (optimizer_override or str(cfg.get("optimizer", "adam"))).lower()
    lr = float(cfg.get("lr", 0.005))
    sample_moves = int(cfg.get("sample_moves", 1000))
    budget_minutes = int(cfg.get("budget_minutes", 90))
    progress_marker = float(cfg.get("progress_marker", 0.30))
    max_epochs = int(cfg.get("max_epochs", 20))

    started = time.time()
    val = prepared.val_samples
    model_cfg = cfg.get("model") or {}
    print(
        f"[smoke-train-v2:{label}] K={history} d_model={model_cfg.get('d_model')} "
        f"heads={model_cfg.get('n_heads')} geom={model_cfg.get('geometric_bias')} "
        f"opt={optimizer} lr={lr} | {len(prepared.train_games)} train / "
        f"{len(prepared.val_games)} val games | {len(prepared.train_samples)} train / "
        f"{len(val)} val samples | short: {prepared.short_games}"
    )

    baseline_move, baseline_acc, _ = set_local_baseline(val)
    cross_move, cross_acc = cross_ruler_baseline(prepared.train_samples, val)
    print(
        f"[smoke-train-v2:{label}] baseline (set-local): {baseline_move} -> {baseline_acc:.4f}; "
        f"cross-ruler (train move {cross_move} on val) {cross_acc:.4f}"
    )

    original_factory = None
    if model_factory is not None:
        # Route the injected factory through this module's `build_model` so the mutation control
        # can poison the model the loop actually trains.
        original_factory = globals()["build_model"]
        globals()["build_model"] = model_factory

    try:
        try:
            model, train_curve, val_curve, seen_total, epochs_run, tracker, grad_norms = (
                _train_epochs(
                    prepared, {**cfg, "optimizer": optimizer}, label=label, started=started
                )
            )
        except DivergenceError as exc:
            partial = {
                "epochs_completed": 0,
                "train_loss_curve": [],
                "val_loss_curve": [],
                "steps": 0,
            }
            write_failure(failure_dir / f"FAILURE-{label}.json", label, exc, partial)
            print(f"::error::[smoke-train-v2:{label}] DIVERGED -- {exc}", file=sys.stderr)
            raise SystemExit(3) from exc
    finally:
        if original_factory is not None:
            globals()["build_model"] = original_factory

    val_first, val_last = val_curve[0], val_curve[-1]
    top1 = accuracy_v2(model, val, history)
    correctness = correctness_v2(model, val, history)

    # --- the Article 1 gate: 1000 top-1 moves must be 100% legal on distinct positions ---
    sampled = legal = 0
    seen_fens: set[str] = set()
    for sample in prepared.train_samples:
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
    plateau = tracker.as_dict()
    plateau_reached = bool(plateau["plateau_reached"])
    plateau_val_loss = float(plateau["best_val_loss"])

    # A non-finite metric of ANY kind is INVALID first: the instrument is destroyed, so
    # SUCCESS/DEAD/INCONCLUSIVE are all meaningless (the branch session-0009 lacked).
    non_finite = not all(
        math.isfinite(x)
        for x in (
            top1,
            baseline_acc,
            val_first,
            val_last,
            plateau_val_loss,
            *val_curve,
            *train_curve,
        )
    )
    if non_finite:
        verdict, reason = "INVALID", "a non-finite metric -- the instrument is destroyed"
    elif legality < 1.0:
        verdict, reason = "INVALID", f"legality {legal}/{sampled} is not 100% (Article 1)"
    elif not plateau_reached:
        verdict, reason = (
            "INCONCLUSIVE",
            f"the {max_epochs}-epoch ceiling was reached before any plateau -- the comparison "
            "is not yet meaningful (session-0011)",
        )
    else:
        # A plateau was reached, so the configuration's own limit is known. Whether it BEATS v1 is
        # a separate question, answered by McNemar in the driver; here we record the fact.
        verdict, reason = (
            "PLATEAUED",
            f"converged at epoch {epochs_run} with val loss {plateau_val_loss:.4f}; "
            f"top-1 {top1:.4f} ({'above' if top1 > baseline_acc else 'at/below'} the set-local "
            f"baseline {baseline_acc:.4f})",
        )

    return {
        "label": label,
        "config": {
            "history": history,
            "window_history": int(cfg.get("window_history", history)),
            "d_model": int(model_cfg.get("d_model", 256)),
            "n_heads": int(model_cfg.get("n_heads", 8)),
            "geometric_bias": bool(model_cfg.get("geometric_bias", False)),
            "optimizer": optimizer,
            "lr": lr,
            "seed": seed,
            "max_train_positions": int(cfg.get("max_train_positions", 60000)),
            "max_epochs": max_epochs,
            "val_fraction": float(cfg.get("val_fraction", 0.05)),
        },
        "data": {
            "train_games": len(prepared.train_games),
            "val_games": len(prepared.val_games),
            "train_samples": len(prepared.train_samples),
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
            "val_loss_plateau": plateau_val_loss,
            "val_loss_curve": val_curve,
            "train_loss_curve": train_curve,
            "top1_accuracy": top1,
            "loss_decreasing": val_last < val_first,
        },
        "convergence": {
            **plateau,
            "epochs_run": epochs_run,
            "steps": seen_total,
            "grad_norm_first": grad_norms[0] if grad_norms else None,
            "grad_norm_last": grad_norms[-1] if grad_norms else None,
            "grad_norm_max": max(grad_norms) if grad_norms else None,
            "grad_norm_samples": len(grad_norms),
        },
        "legality": {"sampled": sampled, "legal": legal, "rate": legality},
        "verdict": {
            "status": verdict,
            "reason": reason,
            "within_budget": within_budget,
            "progress_marker": progress_marker,
            "progress_marker_reached": bool(top1 >= progress_marker),
        },
        "elapsed_seconds": round(elapsed, 2),
        "_correctness": correctness,
    }
