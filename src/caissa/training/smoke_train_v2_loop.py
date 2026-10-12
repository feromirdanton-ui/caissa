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
from caissa.training.adam import AdamState, grad_norm
from caissa.training.forensics import ForensicsRecorder
from caissa.training.smoke_metrics import (
    DIVERGENCE_FACTOR,
    DivergenceError,
    PlateauTracker,
    accuracy_v2,
    convergence_verdict,
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
        # Session-0013 F2: cosine logits, a config flag rather than a code path.
        logit_norm=bool(model_cfg.get("logit_norm", False)),
        logit_scale=float(model_cfg.get("logit_scale", 8.0)),
    )


def _train_epochs(prepared, cfg: dict, *, label: str, started: float):
    """Run the plateau-bounded training loop; raise DivergenceError on a non-finite state.

    Returns ``(model, train_curve, val_curve, steps_total, epochs_run, tracker, grad_norm_curve,
    forensics)``. ``forensics`` is a `ForensicsRecorder` when the config sets ``forensics: true``
    (session-0013's diagnostic run), else ``None`` -- the per-group gradient norms are only paid for
    when a run asks for them.

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
    forensics_enabled = bool(cfg.get("forensics", False))

    train, val = prepared.train_samples, prepared.val_samples
    # The v2 model reads this configuration's K; the EVAL SET was built with the main run's K.
    model = build_model(cfg)
    # Session-0013 F1: global-norm gradient clipping, when the config sets `max_grad_norm`.
    max_grad_norm_raw = cfg.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm_raw) if max_grad_norm_raw is not None else None
    adam = (
        AdamState(betas=tuple(cfg.get("betas", (0.9, 0.999))), max_grad_norm=max_grad_norm)
        if optimizer == "adam"
        else None
    )

    rng = random.Random(seed)
    val_curve: list[float] = [mean_loss_v2(model, val, history)]
    train_curve: list[float] = []
    grad_norms: list[float] = []
    forensics = ForensicsRecorder() if forensics_enabled else None
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
                if forensics is not None:
                    # The diagnostic path needs the gradient OBJECT, not just its norm, so it takes
                    # the same `gradients` the optimizer would and records the per-group split
                    # before the update. Same arithmetic, one extra read.
                    loss, grads = model.gradients(sample.board, x_hist, sample.move)
                    norm = grad_norm(grads)
                    forensics.record(seen_total + 1, grads, model, x_hist)
                    adam.step(model, grads, lr)
                else:
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
        plateau_now = tracker.update(val_curve[-1])
        # Session-0013: a FINITE blow-up is a divergence too -- and it is checked BEFORE the plateau
        # break. Order matters: session-0012's curve blew up while never improving on its first
        # reading, so the plateau branch fired first and the divergence was never seen. A blow-up
        # that also happens to stall is still a blow-up; divergence wins.
        if tracker.diverged:
            exc = DivergenceError(
                f"validation loss diverged at epoch {epochs_run}: {val_curve[0]:.4f} -> "
                f"{val_curve[-1]:.4f} (best {tracker.best:.4f}, factor {DIVERGENCE_FACTOR}) -- "
                "a finite blow-up is INVALID, never a finding (session-0013)"
            )
            # Session-0013 step 1: the diagnostic run's whole point is the trace of the blow-up, so
            # the curves, the gradient norms and the per-group forensics ride on the exception into
            # the FAILURE artifact. A divergence that discards its own evidence is useless.
            exc.partial = {  # type: ignore[attr-defined]
                "epochs_completed": epochs_run,
                "steps": seen_total,
                "train_loss_curve": train_curve,
                "val_loss_curve": val_curve,
                "grad_norm_curve": grad_norms,
                "forensics": forensics.as_dict() if forensics is not None else None,
            }
            raise exc
        if plateau_now:
            print(
                f"[smoke-train-v2:{label}] PLATEAU after {epochs_run} epochs "
                f"({tracker.since_improvement} evaluations without >{eps} improvement)"
            )
            break

    return model, train_curve, val_curve, seen_total, epochs_run, tracker, grad_norms, forensics


def _diverged_metrics(label: str, cfg: dict, exc: Exception, partial: dict, started: float) -> dict:
    """The metrics record for a configuration that DIVERGED, in the factorial's tolerant mode.

    It carries the same shape as a converged record (so the artifact table has one schema) plus the
    divergence's own evidence: the curves, the gradient norms and the per-group forensics. The
    ``convergence_verdict`` is ``DIVERGES`` and ``_diverged`` is True, which the driver uses to make
    the run RED at the end -- a divergence is recorded, never hidden.
    """
    val_curve = list(partial.get("val_loss_curve") or [])
    grad_norms = list(partial.get("grad_norm_curve") or [])
    best = min(val_curve) if val_curve else float("nan")
    return {
        "label": label,
        "config": {
            "history": int(cfg.get("history", 8)),
            "d_model": (cfg.get("model") or {}).get("d_model"),
            "n_heads": (cfg.get("model") or {}).get("n_heads"),
            "geometric_bias": (cfg.get("model") or {}).get("geometric_bias"),
            "logit_norm": (cfg.get("model") or {}).get("logit_norm", False),
            "optimizer": str(cfg.get("optimizer", "adam")).lower(),
            "lr": float(cfg.get("lr", 0.005)),
            "max_grad_norm": cfg.get("max_grad_norm"),
        },
        "model": {
            "name": "numpy_history",
            "val_loss_first": val_curve[0] if val_curve else None,
            "val_loss_last": val_curve[-1] if val_curve else None,
            "val_loss_plateau": best,
            "val_loss_curve": val_curve,
            "train_loss_curve": list(partial.get("train_loss_curve") or []),
            "top1_accuracy": None,
            "loss_decreasing": False,
        },
        "convergence": {
            "steps": int(partial.get("steps", 0)),
            "epochs": int(partial.get("epochs_completed", 0)),
            "grad_norm_first": grad_norms[0] if grad_norms else None,
            "grad_norm_last": grad_norms[-1] if grad_norms else None,
            "grad_norm_max": max(grad_norms) if grad_norms else None,
            "plateau_reached": False,
        },
        "legality": {"rate": None, "sampled": 0},
        "verdict": {"status": "INVALID", "reason": str(exc)},
        "convergence_verdict": {"status": "DIVERGES", "reason": str(exc)},
        "forensics": partial.get("forensics"),
        "elapsed_seconds": round(time.time() - started, 2),
        "_diverged": True,
        "_correctness": [],
    }


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
            (
                model,
                train_curve,
                val_curve,
                seen_total,
                epochs_run,
                tracker,
                grad_norms,
                forensics,
            ) = _train_epochs(
                prepared, {**cfg, "optimizer": optimizer}, label=label, started=started
            )
        except DivergenceError as exc:
            partial = getattr(
                exc,
                "partial",
                {
                    "epochs_completed": 0,
                    "train_loss_curve": [],
                    "val_loss_curve": [],
                    "steps": 0,
                },
            )
            write_failure(failure_dir / f"FAILURE-{label}.json", label, exc, partial)
            print(f"::error::[smoke-train-v2:{label}] DIVERGED -- {exc}", file=sys.stderr)
            if not bool(cfg.get("tolerate_divergence", False)):
                # The duel path: a diverged contender destroys the comparison, so the run is RED.
                raise SystemExit(3) from exc
            # The factorial path (session-0013): a divergence is the ANSWER to "does this fix
            # converge?" -- it is recorded as DIVERGES and the factorial continues, so the other
            # fixes still get their turn. The run still ends RED (the driver exits non-zero), so a
            # divergence can never be read as a clean result.
            return _diverged_metrics(label, cfg, exc, partial, started)
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
        # Session-0013: the pre-registered CONVERGES / DIVERGES / BETWEEN rule, applied to this
        # configuration's own curve. It is the session's question (does anything converge?), kept
        # separate from the duel verdict so the two are never confused.
        "convergence_verdict": convergence_verdict(val_curve, grad_norms, legality),
        "elapsed_seconds": round(elapsed, 2),
        "_correctness": correctness,
        "_forensics": forensics.as_dict() if forensics is not None else None,
    }
