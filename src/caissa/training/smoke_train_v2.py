"""`make smoke-train-v2` -- the reloaded scaled vertical (session-0008; reload session-0010).

Session-0007 proved the composition on real data with `numpy_tiny` (K=0, linear head). This
experiment asks a sharper question on the **same** day-prefix: does raising the model's capacity
and giving it move **history** raise top-1 move accuracy -- and if so, which organ pays for it?

Session-0009 fired it once at `lr=0.05` and the run **diverged** (`train_loss_curve` `[NaN, NaN]`,
identical top-1 for the main run and both ablations) while still finishing green. Session-0010 is
the reload, with three additions named in `LOG/session-0010-preregistration.md`, written BEFORE
dispatch:

  * a **NaN guard** (in `smoke_metrics` + `smoke_train_v2_loop`) so a diverged run is RED, never a
    green non-result;
  * a **set-local baseline** measured on the eval set itself (one ruler for baseline and model);
  * a **v1 reference** retrained in the same run and scored on BOTH eval sets, so the set-shift is
    separable from the organ effect.

The verdict is decided by the pre-registered thresholds (SUCCESS >= 0.30 / DEAD <= set-local
baseline / INCONCLUSIVE between / INVALID on any non-finite metric or legality < 100%). Weights are
a disposable run artifact, never a git object (Article 8).

The pieces live in small sibling modules (`smoke_metrics`, `smoke_train_v2_setup`,
`smoke_train_v2_loop`) so each published file stays small enough to move over the GitHub API
without a transcription slip -- ADR-0009 makes the publish the risky step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from caissa.training.dataset import ShardStats, load_games
from caissa.training.smoke_metrics import assert_finite, set_local_baseline
from caissa.training.smoke_train_v2_loop import build_model, train_configuration
from caissa.training.smoke_train_v2_setup import merge, prepare, train_v1_reference

# Names the session-0008 tests still reach for; keep them working as thin aliases. `_build_model`
# is resolved at call time inside `_model_factory`, which is the seam the mutation-control test
# monkeypatches to inject a poisoned model and prove a divergence turns the run RED.
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


def run(config_path: str) -> dict:
    """Train the v1 reference, the main configuration, then each ablation; write the artifacts."""
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
            f"[smoke-train-v2] no usable games under {shards_dir}. Run the data sample and the "
            "sharder first (see docs/TRAINING.md)."
        )
    print(
        f"[smoke-train-v2] loaded {stats.games} games / {stats.positions} positions "
        f"(broken={stats.broken_replays} skipped={stats.skipped_games})"
    )

    prepared = prepare(cfg, games)

    # v1 reference first: it reuses the SAME split, and it is what makes the v1->v2 comparison a
    # comparison of models rather than of rulers (session-0010 pre-registration, item 5).
    v1_reference = train_v1_reference(cfg, prepared) if cfg.get("v1_reference", True) else None
    if v1_reference is not None:
        print(
            f"[smoke-train-v2] v1 reference (tiny, lr={v1_reference['lr']}): "
            f"top-1 {v1_reference['v1_at_v1_set']['top1_accuracy']:.4f} on its own set "
            f"({v1_reference['v1_at_v1_set']['val_positions']} pos), "
            f"{v1_reference['v1_at_v2_set']['top1_accuracy']:.4f} on the v2 set "
            f"({v1_reference['v1_at_v2_set']['val_positions']} pos)"
        )

    main = train_configuration(
        prepared, cfg, label="v2", failure_dir=metrics_path.parent, model_factory=_model_factory
    )
    main["schema"] = "caissa-smoke-train-v2-metrics/1"
    if v1_reference is not None:
        main["v1_reference"] = v1_reference
    assert_finite(main)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(main, indent=2, sort_keys=True))
    print(
        f"[smoke-train-v2] MAIN: val loss {main['model']['val_loss_first']:.4f} -> "
        f"{main['model']['val_loss_last']:.4f} | top-1 {main['model']['top1_accuracy']:.4f} "
        f"vs set-local baseline {main['baseline']['val_accuracy']:.4f} | "
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
        result = train_configuration(
            prepared,
            merge(cfg, overrides),
            label=str(name),
            failure_dir=metrics_path.parent,
            model_factory=_model_factory,
        )
        result["delta_top1_vs_main"] = (
            result["model"]["top1_accuracy"] - main["model"]["top1_accuracy"]
        )
        results_ablations.append(result)
        print(
            f"[smoke-train-v2] ablation '{name}': top-1 {result['model']['top1_accuracy']:.4f} "
            f"(delta {result['delta_top1_vs_main']:+.4f}) verdict {result['verdict']['status']}"
        )

    assert_finite(results)
    ablations_path.parent.mkdir(parents=True, exist_ok=True)
    ablations_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[smoke-train-v2] ablations written to {ablations_path}")
    return main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caissa-1 smoke-train-v2 (deep lane #2)")
    parser.add_argument("--config", default="configs/smoke-train-v2.yaml")
    args = parser.parse_args(argv)
    metrics = run(args.config)
    # INVALID is a defect, not a finding: fail the run so it cannot be read as a result.
    return 1 if metrics["verdict"]["status"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
