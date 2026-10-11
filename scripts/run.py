#!/usr/bin/env python3
"""Caïssa-1 — one source of commands for every environment (ADR-0005).

The local loop (Windows, no `make`, no `git`) and the CI runner must converge on
the SAME commands. This script is that single source: `make X` delegates here, and
`python scripts/run.py X` is the identical thing. Nothing in this file needs a
POSIX shell, so it runs on Windows, macOS and Linux alike.

Usage:
    python scripts/run.py <target> [--config PATH] [-- ...extra args]

Targets: help install lint format test smoke smoke-train smoke-train-v2 data data-sample
         processing label eval train ci clean
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/tiny.yaml"


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> int:
    """Run a command, echoing it first. Returns its exit code."""
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _module(module: str, *args: str) -> list[str]:
    return _py("-m", module, *args)


# --- targets -----------------------------------------------------------------


def t_help(_: argparse.Namespace) -> int:
    print(__doc__)
    return 0


def t_install(_: argparse.Namespace) -> int:
    return _run(_py("-m", "pip", "install", "-e", ".[dev]"))


def t_lint(_: argparse.Namespace) -> int:
    for cmd in (
        _module("ruff", "check", "src", "tests", "tools"),
        _module("black", "--check", "src", "tests", "tools"),
        _module("mypy", "src/caissa"),
    ):
        code = _run(cmd)
        if code != 0:
            return code
    return 0


def t_format(_: argparse.Namespace) -> int:
    for cmd in (
        _module("ruff", "check", "--fix", "src", "tests", "tools"),
        _module("black", "src", "tests", "tools"),
    ):
        code = _run(cmd)
        if code != 0:
            return code
    return 0


def t_test(_: argparse.Namespace) -> int:
    return _run(_module("pytest", "-q"))


def t_smoke(args: argparse.Namespace) -> int:
    return _run(_module("caissa.training.smoke", "--config", args.config))


def t_smoke_train(args: argparse.Namespace) -> int:
    """Deep lane (ADR-0011): the «Vertical» experiment on real shards. Honest scale, no claims."""
    return _run(_module("caissa.training.smoke_train", "--config", args.config))


def t_smoke_train_v2(args: argparse.Namespace) -> int:
    """Deep lane #2 (session 0008): the scaled vertical — history + geometry, with two ablations."""
    return _run(_module("caissa.training.smoke_train_v2", "--config", args.config))


def t_data(_: argparse.Namespace) -> int:
    return _run(_module("caissa.acquisition.manifest", "--status"))


def t_data_sample(args: argparse.Namespace) -> int:
    return _run(_module("caissa.acquisition.sample", "--config", args.config))


def t_processing(args: argparse.Namespace) -> int:
    """Phase 2: stream a PGN -> filter -> dedup -> shard -> QA report."""
    return _run(
        _module(
            "caissa.processing.cli",
            args.pgn,
            "--out",
            args.out,
            "--qa",
            args.qa,
            "--shard-size",
            str(args.shard_size),
            *(["--lenient"] if args.lenient else []),
            *(["--truncated-ok"] if args.truncated_ok else []),
        )
    )


def t_label(args: argparse.Namespace) -> int:
    """Phase 2: blunder labeling over a PGN (Stockfish when present, else MOCK)."""
    return _run(
        _module(
            "caissa.processing.cli",
            "label",
            args.pgn,
            "--limit",
            str(args.limit),
            "--out",
            args.qa,
            "--engine",
            args.engine,
        )
    )


def t_eval(args: argparse.Namespace) -> int:
    return _run(_module("caissa.eval.arena", "--config", args.config))


def t_train(args: argparse.Namespace) -> int:
    return _run(_module("caissa.training.act1", "--config", args.config))


def t_ci(args: argparse.Namespace) -> int:
    for target in (t_lint, t_test):
        code = target(args)
        if code != 0:
            return code
    return 0


def t_clean(_: argparse.Namespace) -> int:
    for name in (".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"):
        shutil.rmtree(REPO_ROOT / name, ignore_errors=True)
    for egg in REPO_ROOT.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    for cache in REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for part in REPO_ROOT.rglob("*.part"):
        part.unlink(missing_ok=True)
    print("[run] cleaned caches, build artifacts and .part files")
    return 0


TARGETS = {
    "help": t_help,
    "install": t_install,
    "lint": t_lint,
    "format": t_format,
    "test": t_test,
    "smoke": t_smoke,
    "smoke-train": t_smoke_train,
    "smoke-train-v2": t_smoke_train_v2,
    "data": t_data,
    "data-sample": t_data_sample,
    "processing": t_processing,
    "label": t_label,
    "eval": t_eval,
    "train": t_train,
    "ci": t_ci,
    "clean": t_clean,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 command runner (ADR-0005)")
    parser.add_argument("target", nargs="?", default="help", choices=sorted(TARGETS))
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config file for the target")
    # The smoke-train lanes ship their own configs, so their defaults are the experiment's, not
    # `tiny`. Each is matched exactly (`smoke-train-v2` must not fall into the `smoke-train` branch).
    if argv and argv[:1] == ["smoke-train-v2"] and "--config" not in argv:
        argv = ["smoke-train-v2", "--config", "configs/smoke-train-v2.yaml", *argv[1:]]
    elif argv and argv[:1] == ["smoke-train"] and "--config" not in argv:
        argv = ["smoke-train", "--config", "configs/smoke-train.yaml", *argv[1:]]
    parser.add_argument("--pgn", default="data/raw/lichess/lichess_day_2013-01.pgn.zst")
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--qa", default="data/qa/processing.json")
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--engine", default="auto", choices=["auto", "mock", "stockfish"])
    parser.add_argument("--lenient", action="store_true")
    parser.add_argument("--truncated-ok", action="store_true")
    args = parser.parse_args(argv)
    return TARGETS[args.target](args)


if __name__ == "__main__":
    raise SystemExit(main())
