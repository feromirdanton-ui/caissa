"""CI driver for smoke-train-stability.yml (session-0013).

The F0-F4 fix factorial runs on the same real shards as the duel, but its question is different:
*does any configuration of the v2 family converge at all?* The data steps are shared with the duel
driver (imported, not copied -- one implementation of "fetch, shard, verify"), and only the summary
is factorial-specific.

Commands (paths relative to the repo root, as the workflow runs):
    fetch-prefix   download the Lichess day prefix
    shard          shard the prefix into jsonl.zst + QA
    verify-shards  re-hash every shard against the QA manifest
    override       apply the dispatch position budget
    guard          fail the run on a FAILURE artifact
    summary        echo the convergence table and write the step summary
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from scripts.smoke_train_v2_ci import fetch_prefix, shard, verify_shards

CONFIG = "configs/smoke-train-stability.yaml"
CI_CONFIG = "configs/smoke-train-stability.ci.yaml"
METRICS = "data/qa/metrics-stability.json"
ABLATIONS = "data/qa/ablations-stability.json"


def override() -> int:
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    cfg["max_train_positions"] = int(os.environ["TRAIN_POSITIONS"])
    Path(CI_CONFIG).write_text(yaml.safe_dump(cfg))
    print(f"effective train positions: {cfg['max_train_positions']}")
    return 0


def guard() -> int:
    """A FAILURE artifact means a configuration diverged -- the run is RED, never a clean result."""
    failures = sorted(Path("data/qa").glob("FAILURE-*.json"))
    if failures:
        print("::error::a configuration diverged; a FAILURE artifact is present")
        for path in failures:
            print(path.read_text())
        return 1
    print("no FAILURE artifacts: no configuration diverged")
    return 0


def summary() -> int:
    main = json.loads(Path(METRICS).read_text())
    summary_rows = main["convergence_summary"]
    lines = [
        "## smoke-train-stability -- the F0-F4 fix factorial (session-0013)",
        "",
        f"- **verdict**: `{main['duel_verdict']['status']}` -- {main['duel_verdict']['reason']}",
        "",
        "### does each fix converge? (pre-registered rule, LOG/session-0013-preregistration.md)",
        "",
        "| configuration | status | val loss first -> last (best) | grad norm first -> max | steps |"
        " legality |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        first = row["val_loss_first"]
        last = row["val_loss_last"]
        best = row["val_loss_plateau"]
        gfirst = row["grad_norm_first"]
        gmax = row["grad_norm_max"]
        legality = row["legality"]
        lines.append(
            f"| {row['label']} | **{row['status']}** | "
            f"{_fmt(first)} -> {_fmt(last)} ({_fmt(best)}) | "
            f"{_fmt(gfirst)} -> {_fmt(gmax)} | {row['steps']} | "
            f"{'100%' if legality == 1.0 else _fmt(legality)} |"
        )
    lines += ["", "### why each status", ""]
    for row in summary_rows:
        lines.append(f"- **{row['label']}** ({row['status']}): {row['reason']}")
    text = "\n".join(lines)
    Path("data/qa/summary.md").write_text(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        Path(step_summary).write_text(text)
    print(text)
    return 0


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def all_steps() -> int:
    """The whole pipeline in one step: fetch -> shard -> verify -> override -> forensics -> factorial.

    Kept as one command so the workflow file stays small enough to publish by hand (ADR-0012). The
    forensics run is a DIAGNOSTIC (its contract is a trace, not convergence) and exits 0; the
    factorial is the verdict run and goes RED on an unexpected divergence.
    """
    import subprocess

    for step in (fetch_prefix, shard, verify_shards, override):
        code = step()
        if code != 0:
            return code
    subprocess.run(
        [
            sys.executable,
            "-m",
            "caissa.training.smoke_train_v2",
            "--config",
            "configs/smoke-train-forensics.yaml",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "caissa.training.smoke_train_v2",
            "--config",
            "configs/smoke-train-stability.ci.yaml",
        ],
        check=True,
    )
    return summary()


COMMANDS = {
    "fetch-prefix": fetch_prefix,
    "shard": shard,
    "verify-shards": verify_shards,
    "override": override,
    "guard": guard,
    "summary": summary,
    "all": all_steps,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in COMMANDS:
        print(f"usage: {__name__} {{{'|'.join(COMMANDS)}}}", file=sys.stderr)
        return 2
    return int(COMMANDS[args[0]]())


if __name__ == "__main__":
    raise SystemExit(main())
