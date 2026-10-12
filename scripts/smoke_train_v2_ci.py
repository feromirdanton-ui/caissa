"""CI driver for smoke-train-v2.yml (session-0010).

The workflow's shell logic lives here, not inline: an ~8 KB YAML kept arriving at
the API with a differing blob sha while every smaller file published
byte-identically (ADR-0009 -- publish, verify, CI).

Commands (paths relative to the repo root, as the workflow runs):
    fetch-prefix   download the Lichess day prefix
    shard          shard the prefix into jsonl.zst + QA
    verify-shards  re-hash every shard against the QA manifest
    override       apply the dispatch position budget
    guard          fail the run on a FAILURE artifact
    summary        echo a markdown verdict and write the step summary
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

PREFIX_FILE = "data/raw/lichess/lichess_day_{month}.pgn.zst"


def fetch_prefix() -> int:
    from caissa.acquisition.lichess_bulk import LICHESS_STANDARD, download_prefix

    cfg = yaml.safe_load(Path("configs/data-sample.yaml").read_text())
    month = os.environ["MONTH"]
    target = Path(PREFIX_FILE.format(month=month))
    res = download_prefix(
        LICHESS_STANDARD.format(month=month), target, int(cfg["lichess_prefix_bytes"])
    )
    print(
        f"[data] {res.prefix_bytes} B prefix sha256={res.sha256[:16]}... "
        f"resumed_from={res.resumed_from}"
    )
    return 0


def shard() -> int:
    from caissa.processing import cli

    argv = [
        PREFIX_FILE.format(month=os.environ["MONTH"]),
        "--out",
        "data/shards",
        "--qa",
        "data/qa/processing.json",
        "--shard-size",
        os.environ.get("SHARD_SIZE", "10000"),
        "--truncated-ok",
    ]
    return int(cli.main(argv))


def verify_shards() -> int:
    qa = json.loads(Path("data/qa/processing.json").read_text())
    failures = 0
    for info in qa["shards"]:
        path = Path("data/shards") / info["name"]
        if not path.exists():
            print(f"::error::shard missing: {path}")
            failures += 1
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            print(f"::error::sha256 mismatch for {path}")
            failures += 1
        else:
            print(f"[verify] {info['name']}: {info['rows']} rows, sha256 ok")
    if failures:
        return 1
    print(f"[verify] all {len(qa['shards'])} shards certified")
    return 0


def override() -> int:
    cfg = yaml.safe_load(Path("configs/smoke-train-v2.yaml").read_text())
    cfg["max_train_positions"] = int(os.environ["TRAIN_POSITIONS"])
    Path("configs/smoke-train-v2.ci.yaml").write_text(yaml.safe_dump(cfg))
    print(f"effective train positions: {cfg['max_train_positions']}")
    return 0


def guard() -> int:
    failures = sorted(Path("data/qa").glob("FAILURE-*.json"))
    if failures:
        print("::error::the run diverged; a FAILURE artifact is present")
        for path in failures:
            print(path.read_text())
        return 1
    print("no FAILURE artifacts: the run did not diverge")
    return 0


def summary() -> int:
    main = json.loads(Path("data/qa/metrics-v2.json").read_text())
    abl = json.loads(Path("data/qa/ablations-v2.json").read_text())
    model = main["model"]
    conv = main["convergence"]
    duel = main["duel_verdict"]
    v1 = main["v1_reference"]
    lines = [
        "## smoke-train-v2 -- the convergence duel (session-0011)",
        "",
        f"- **duel verdict**: `{duel['status']}` -- {duel['reason']}",
        f"- main `{main['label']}` verdict: `{main['verdict']['status']}`",
        f"- val loss: {model['val_loss_first']:.4f} -> {model['val_loss_last']:.4f} "
        f"(plateau {model['val_loss_plateau']:.4f})",
        f"- top-1: {model['top1_accuracy']:.4f} vs set-local baseline "
        f"{main['baseline']['val_accuracy']:.4f}",
        f"- legality: {main['legality']['legal']}/{main['legality']['sampled']}",
        f"- plateau: reached={conv['plateau_reached']} after {conv['epochs_run']} epochs / "
        f"{conv['steps']} steps (patience {conv['patience']}, eps {conv['eps']})",
        f"- grad norm: first {conv['grad_norm_first']:.3e} -> last {conv['grad_norm_last']:.3e} "
        f"(max {conv['grad_norm_max']:.3e})",
        "",
        f"v1 reference (tiny, sgd lr={v1['lr']}, {v1['epochs']} epochs): "
        f"top-1 {v1['v1_at_v2_set']['top1_accuracy']:.4f} on the shared eval set "
        f"({v1['v1_at_v2_set']['val_positions']} positions)",
        "",
        "### the frozen 2x2 -- every delta tested with McNemar against v1 on the SAME eval set",
        "",
        "| configuration | top-1 | delta vs v1 | McNemar z | p | method | favours | significant |"
        " steps | plateau | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [(main["label"], main), *((a["label"], a) for a in abl["ablations"])]
    for label, entry in rows:
        mc = entry["mcnemar_vs_v1"]
        lines.append(
            f"| {label} | {entry['model']['top1_accuracy']:.4f} | "
            f"{entry['top1_delta_vs_v1']:+.4f} | {mc['z']:.3f} | {mc['p_value']:.4g} | "
            f"{mc['method']} | {mc['favours']} | {mc['significant']} | "
            f"{entry['convergence']['steps']} | {entry['convergence']['plateau_reached']} | "
            f"{entry['verdict']['status']} |"
        )
    text = "\n".join(lines)
    Path("data/qa/summary.md").write_text(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        Path(step_summary).write_text(text)
    print(text)
    return 0


COMMANDS = {
    "fetch-prefix": fetch_prefix,
    "shard": shard,
    "verify-shards": verify_shards,
    "override": override,
    "guard": guard,
    "summary": summary,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in COMMANDS:
        print(f"usage: {__name__} {{{'|'.join(COMMANDS)}}}", file=sys.stderr)
        return 2
    return int(COMMANDS[args[0]]())


if __name__ == "__main__":
    raise SystemExit(main())
