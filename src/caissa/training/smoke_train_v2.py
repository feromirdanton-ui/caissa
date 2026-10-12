"""`make smoke-train-v2` -- the convergence duel (session-0011; the reload was session-0010).

The question session-0011 asks is sharper than session-0010's: *does any configuration of the v2
family beat v1 at v1's own convergence limit, on the same eval set, by a statistically significant
margin?* Two things follow from the wording, and both are in `LOG/session-0011-preregistration.md`,
written before this file ran:

  * every configuration trains to **its own plateau** (patience 5, eps 1e-4, hard ceiling 20 epochs),
    because equal step counts judge the stopwatch rather than the players;
  * every delta is tested with **McNemar** on the paired correctness vectors, because a two-point
    top-1 gap on ten thousand positions can be carried by a handful of positions.

The v2 family runs with Adam at one recorded lr (0.005); the v1 reference keeps SGD at its own
stable 0.05. A learning rate is part of an architecture's price, so neither model is handicapped to
flatter the other. Weights are a disposable run artifact, never a git object (Article 8).

The pieces live in small sibling modules so each published file stays small enough to move over the
GitHub API without a transcription slip -- ADR-0009 makes the publish the risky step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from caissa.training.dataset import ShardStats, load_games
from caissa.training.mcnemar import mcnemar
from caissa.training.smoke_metrics import (
    assert_finite,
    correctness_tiny,
    set_local_baseline,
)
from caissa.training.smoke_train_v2_loop import build_model, train_configuration
from caissa.training.smoke_train_v2_setup import merge, prepare, train_v1_reference

# Names the earlier session's tests still reach for; keep them working as thin aliases. The
# mutation-control test monkeypatches `_build_model`, which the loop resolves at call time.
_assert_finite = assert_finite
_build_model = build_model
_merge = merge

__all__ = [
    "_assert_finite",
    "_build_model",
    "_merge",
    "main",
    "run",
    "set_local_baseline",
]


def _model_factory(cfg: dict):
    """Resolve `_build_model` at call time so a test can monkeypatch the driver's symbol."""
    return _build_model(cfg)


def _decide(main_verdicts: list[dict], comparisons: list[dict]) -> dict:
    """Apply the pre-registered rule to the collected comparisons. No threshold is read here.

    SUCCESS     -- at least one configuration beats v1 on the shared eval set with p < 0.05
    DEAD        -- no configuration's top-1 exceeds v1's on that set
    INCONCLUSIVE -- some configuration is above v1 but nothing is significant, OR a configuration
                    never reached its plateau (the ceiling fired first)

    Polarity (session-0012): the comparison is built as ``mcnemar(correctness, v1_correct)``, so
    ``a`` is the v2 configuration and ``b`` is the v1 reference -- ``favours == "a"`` is a v2 win
    and ``favours == "b"`` is a v1 win. Session-0011's rule tested ``favours == "b"``, which read a
    v1 win as a v2 win: a run in which EVERY configuration lost to v1 (all deltas negative) was
    reported as SUCCESS. The delta sign is the independent witness, so a winner must also have a
    positive delta; the two can no longer disagree.
    """
    if not any(v["status"] == "PLATEAUED" for v in main_verdicts):
        return {
            "status": "INCONCLUSIVE",
            "reason": (
                "no configuration reached its plateau within the epoch ceiling -- the comparison "
                "is not yet meaningful (session-0011)"
            ),
        }
    winners = [
        c
        for c in comparisons
        if c["mcnemar"]["favours"] == "a"
        and c["mcnemar"]["significant"]
        and c["top1_delta_vs_v1"] > 0
    ]
    if winners:
        best = max(winners, key=lambda c: c["top1_delta_vs_v1"])
        mc = best["mcnemar"]
        return {
            "status": "SUCCESS",
            "reason": (
                f"{best['label']} beats v1@set on the shared eval set: delta "
                f"{best['top1_delta_vs_v1']:+.4f}, McNemar p={mc['p_value']:.4g} "
                f"({mc['method']}, favours v2)"
            ),
            "winner": best["label"],
        }
    above = [c for c in comparisons if c["top1_delta_vs_v1"] > 0]
    if above:
        best = max(above, key=lambda c: c["top1_delta_vs_v1"])
        mc = best["mcnemar"]
        return {
            "status": "INCONCLUSIVE",
            "reason": (
                f"{best['label']} is above v1@set by {best['top1_delta_vs_v1']:+.4f} but the "
                f"difference is not significant (McNemar p={mc['p_value']:.4g}, {mc['method']})"
            ),
        }
    return {
        "status": "DEAD",
        "reason": "no v2-family configuration's top-1 exceeds the v1 reference on the shared eval set",
    }


def run(config_path: str) -> dict:
    """Train the v1 reference, the main configuration and each listed configuration; write artifacts."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    shards_dir = cfg.get("shards", "data/shards")
    metrics_path = Path(cfg.get("metrics", "data/qa/metrics-v2.json"))
    ablations_path = Path(cfg.get("ablations_out", "data/qa/ablations-v2.json"))
    history = int(cfg.get("history", 8))
    window_history = int(cfg.get("window_history", history))
    max_load = int(cfg.get("max_positions_to_load", 200_000))

    print(f"[smoke-train-v2] config={config_path} shards={shards_dir} K={history}")
    stats = ShardStats(shards=sorted(p.name for p in Path(shards_dir).glob("*.jsonl.zst")))
    games = load_games(
        shards_dir, max_positions=max_load, min_moves=window_history + 2, stats=stats
    )
    if not games:
        raise SystemExit(
            f"[smoke-train-v2] no usable games under {shards_dir}. Run the data sample and the "
            "sharder first (see docs/TRAINING.md)."
        )
    print(
        f"[smoke-train-v2] loaded {stats.games} games / {stats.positions} positions "
        f"(broken={stats.broken_replays} skipped={stats.skipped_games})"
    )

    prepared = prepare(cfg, games)

    # --- the v1 reference: trained to ITS plateau, scored on both eval sets -------------------
    # Session-0013's stability config sets `v1_reference: false`: the question is convergence, not
    # beating v1, so the duel machinery is skipped entirely rather than run against a reference
    # nobody is comparing to.
    use_v1 = bool(cfg.get("v1_reference", True))
    v1_metrics: dict = {}
    v1_correct: list[bool] = []
    if use_v1:
        v1_model, v1_metrics = train_v1_reference(cfg, prepared)
        v1_correct = correctness_tiny(v1_model, prepared.v1_on_v2_set)
        print(
            f"[smoke-train-v2] v1 reference (tiny, sgd lr={v1_metrics['lr']}, "
            f"{v1_metrics['epochs']} epochs): top-1 {v1_metrics['v1_at_v1_set']['top1_accuracy']:.4f} "
            f"on its own set ({v1_metrics['v1_at_v1_set']['val_positions']} pos), "
            f"{v1_metrics['v1_at_v2_set']['top1_accuracy']:.4f} on the v2 set "
            f"({v1_metrics['v1_at_v2_set']['val_positions']} pos)"
        )
    else:
        print("[smoke-train-v2] v1_reference disabled -- convergence-only run (session-0013)")

    def _run_one(name: str, this_cfg: dict) -> dict:
        result = train_configuration(
            prepared,
            this_cfg,
            label=name,
            failure_dir=metrics_path.parent,
            model_factory=_model_factory,
        )
        correctness = result.pop("_correctness")
        diverged = bool(result.pop("_diverged", False))
        forensics = result.pop("_forensics", None)
        if forensics is not None:
            # Session-0013 step 1: the diagnostic run's per-group gradient norms, embedding norms
            # and max|logit| go into the artifact, so the first exploding group and step are read
            # from the published numbers rather than from a console log that scrolls away.
            result["forensics"] = forensics
        if use_v1 and not diverged:
            comparison = mcnemar(correctness, v1_correct, alpha=float(cfg.get("alpha", 0.05)))
            result["mcnemar_vs_v1"] = comparison.as_dict()
            result["top1_delta_vs_v1"] = result["model"]["top1_accuracy"] - (
                v1_metrics["v1_at_v2_set"]["top1_accuracy"]
            )
        _correctness_by_label[name] = correctness
        cv = result["convergence_verdict"]["status"]
        top1 = result["model"]["top1_accuracy"]
        top1_str = f"{top1:.4f}" if top1 is not None else "n/a (diverged)"
        print(
            f"[smoke-train-v2] {name}: top-1 {top1_str} "
            f"steps={result['convergence']['steps']} plateau={result['convergence']['plateau_reached']} "
            f"CONVERGENCE={cv}"
        )
        return result

    _correctness_by_label: dict[str, list[bool]] = {}

    # --- the main configuration (d256@K8) ------------------------------------------------------
    main = _run_one("d256_k8", cfg)
    main["schema"] = "caissa-smoke-train-v2-metrics/1"
    main["v1_reference"] = v1_metrics

    # --- the rest of the frozen 2x2: {128,256} x {0,8} -----------------------------------------
    axis_specs = cfg.get("configurations") or []
    others: list[dict] = []
    for spec in axis_specs:
        name = spec.get("name")
        if not name:
            raise SystemExit("[smoke-train-v2] a configuration entry has no `name`")
        print(
            f"[smoke-train-v2] configuration '{name}': overrides={json.dumps(spec.get('overrides') or {}, sort_keys=True)}"
        )
        others.append(_run_one(str(name), merge(cfg, spec.get("overrides") or {})))

    comparisons = []
    if use_v1:
        comparisons = [
            {
                "label": r["label"],
                "top1_accuracy": r["model"]["top1_accuracy"],
                # A diverged configuration has no top-1 and no McNemar -- it is reported as such
                # rather than given a fabricated delta (session-0013).
                "top1_delta_vs_v1": r.get("top1_delta_vs_v1"),
                "mcnemar": r.get("mcnemar_vs_v1"),
                "steps": r["convergence"]["steps"],
                "plateau_reached": r["convergence"]["plateau_reached"],
                "verdict": r["verdict"]["status"],
            }
            for r in [main, *others]
        ]
        main["pairwise_vs_v1"] = comparisons
    # Session-0013 step 4: the WITHIN-FAMILY comparison -- K8 vs K0 at each width. The v1 duel asks
    # "does v2 beat v1"; this asks "does history help at fixed capacity", which is a different
    # question and needs its own paired test on the same eval set. Both are reported; neither is
    # inferred from the other.
    within_family: list[dict] = []
    for width, k8_label, k0_label in (
        (256, "d256_k8", "d256_k0"),
        (128, "d128_k8", "d128_k0"),
    ):
        if k8_label in _correctness_by_label and k0_label in _correctness_by_label:
            cmp = mcnemar(
                _correctness_by_label[k8_label],
                _correctness_by_label[k0_label],
                alpha=float(cfg.get("alpha", 0.05)),
            )
            within_family.append(
                {
                    "width": width,
                    "a": k8_label,
                    "b": k0_label,
                    "question": "does history (K8) beat no-history (K0) at this width",
                    "mcnemar": cmp.as_dict(),
                }
            )
    main["within_family"] = within_family
    # Session-0013: the convergence summary -- the session's actual question. Every configuration's
    # CONVERGES / DIVERGES / BETWEEN status, in one place, so the verdict is read off the artifact.
    main["convergence_summary"] = [
        {
            "label": r["label"],
            "status": r["convergence_verdict"]["status"],
            "reason": r["convergence_verdict"]["reason"],
            "val_loss_first": r["model"]["val_loss_first"],
            "val_loss_last": r["model"]["val_loss_last"],
            "val_loss_plateau": r["model"]["val_loss_plateau"],
            "grad_norm_first": r["convergence"]["grad_norm_first"],
            "grad_norm_last": r["convergence"]["grad_norm_last"],
            "grad_norm_max": r["convergence"]["grad_norm_max"],
            "steps": r["convergence"]["steps"],
            "legality": r["legality"]["rate"],
        }
        for r in [main, *others]
    ]
    if use_v1:
        main["duel_verdict"] = _decide([r["verdict"] for r in [main, *others]], comparisons)
        if not main["duel_verdict"]["reason"]:
            raise SystemExit("[smoke-train-v2] the duel verdict has no reason -- refusing to write")
    else:
        # No duel this session: the question is convergence. The verdict is the convergence summary.
        converged = [c for c in main["convergence_summary"] if c["status"] == "CONVERGES"]
        main["duel_verdict"] = {
            "status": "CONVERGENCE_ONLY",
            "reason": (
                f"{len(converged)} of {len(main['convergence_summary'])} configurations converge "
                "-- no duel with v1 this session (session-0013)"
            ),
            "converged": [c["label"] for c in converged],
        }

    assert_finite(main)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(main, indent=2, sort_keys=True))
    # Session-0013: a divergence is RECORDED, never hidden -- but the run still ends RED, so a
    # factorial containing a blow-up can never be read as a clean result. The artifact is written
    # first (the evidence must survive), then the process exits non-zero.
    diverged_labels = [c["label"] for c in main["convergence_summary"] if c["status"] == "DIVERGES"]
    # A divergence the pre-registration NAMED as expected (the F0 control) is the predicted
    # behaviour and does not redden the run; an unexpected one does. The list is written in the
    # config BEFORE the shot, so this is not a post-hoc exemption.
    expected = set(cfg.get("expected_divergence") or [])
    unexpected = [lbl for lbl in diverged_labels if lbl not in expected]
    if unexpected and not bool(cfg.get("diagnostic", False)):
        print(
            f"::error::[smoke-train-v2] {len(unexpected)} configuration(s) DIVERGED "
            f"UNEXPECTEDLY: {', '.join(unexpected)} -- the run is RED (session-0013)",
            file=sys.stderr,
        )
        raise SystemExit(3)
    if diverged_labels:
        # A DIAGNOSTIC run's contract is to produce a trace, not to converge -- its divergence is
        # the observation it exists to record, so it exits 0 because it succeeded at its job. And a
        # factorial whose ONLY divergence is the pre-registered control has also answered its
        # question. No failure is swallowed: the artifact records every status and is uploaded.
        print(
            f"[smoke-train-v2] {len(diverged_labels)} configuration(s) DIVERGED "
            f"({', '.join(diverged_labels)}) -- recorded as the observation; "
            f"expected={sorted(expected)} unexpected={unexpected}"
        )
    print(
        f"[smoke-train-v2] MAIN d256_k8: val loss {main['model']['val_loss_first']:.4f} -> "
        f"{main['model']['val_loss_last']:.4f} (plateau {main['model']['val_loss_plateau']:.4f}) | "
        f"top-1 {main['model']['top1_accuracy']:.4f} vs set-local baseline "
        f"{main['baseline']['val_accuracy']:.4f} | legality "
        f"{main['legality']['legal']}/{main['legality']['sampled']} | "
        f"DUEL: {main['duel_verdict']['status']}"
    )
    print(f"[smoke-train-v2] metrics written to {metrics_path}")

    if not others:
        return main

    results: dict = {
        "schema": "caissa-smoke-train-v2-ablations/1",
        "baseline": {
            "label": "d256_k8",
            "top1_accuracy": main["model"]["top1_accuracy"],
        },
        "v1_reference": (
            {
                "label": "v1_reference",
                "top1_accuracy_at_v2_set": v1_metrics["v1_at_v2_set"]["top1_accuracy"],
                "val_positions": v1_metrics["v1_at_v2_set"]["val_positions"],
            }
            if use_v1
            else None
        ),
        "duel_verdict": main["duel_verdict"],
        "convergence_summary": main["convergence_summary"],
        "ablations": others,
    }
    assert_finite(results)
    ablations_path.parent.mkdir(parents=True, exist_ok=True)
    ablations_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[smoke-train-v2] configurations written to {ablations_path}")
    return main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caissa-1 smoke-train-v2 (deep lane #2, duel)")
    parser.add_argument("--config", default="configs/smoke-train-v2.yaml")
    args = parser.parse_args(argv)
    metrics = run(args.config)
    # INVALID is a defect, not a finding: fail the run so it cannot be read as a result.
    statuses = {metrics["verdict"]["status"]}
    statuses.update(c["verdict"] for c in metrics.get("pairwise_vs_v1", []))
    return 1 if "INVALID" in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
