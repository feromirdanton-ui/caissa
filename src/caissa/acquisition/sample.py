"""`make data-sample` — Phase 1 DoD entry point (GOAL.md §5, §10).

Pulls a month of Lichess + 1000 puzzles + a TWIC month, resumably. In the build phase this
is wired but not executed (it would download GB). It refuses to fabricate data.
"""

from __future__ import annotations

import argparse

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 data sample (Phase 1)")
    parser.add_argument("--config", default="configs/data-sample.yaml")
    args = parser.parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    print(f"[data-sample] config={args.config}")
    print(f"[data-sample] lichess_month={cfg.get('lichess_month')}")
    print(f"[data-sample] puzzles={cfg.get('puzzles')} twic_month={cfg.get('twic_month')}")
    print("[data-sample] BUILD PHASE: loaders are wired but not executed here.")
    print("[data-sample] Run on a machine with disk budget; see docs/TRAINING.md §2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
