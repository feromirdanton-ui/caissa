"""Act I entry point — the Definition of Awesome (GOAL.md §2).

`make train CONFIG=configs/nano.yaml` lands here. On a real cluster this launches SFT on
move prediction + position reconstruction. In the build phase it validates the config and
the environment, then refuses to pretend: it does NOT fabricate training results.
"""

from __future__ import annotations

import argparse
import sys

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 Act I (language SFT)")
    parser.add_argument("--config", default="configs/nano.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="validate only")
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print(f"[act1] config={args.config}")
    print(f"[act1] scale={cfg.get('scale')} params~{cfg.get('params')}")
    print(f"[act1] data={cfg.get('data')}")
    print(f"[act1] resume={args.resume}")

    try:
        import torch  # noqa: F401

        has_torch = True
    except Exception:
        has_torch = False

    if not has_torch:
        print("[act1] torch not installed — this is the build phase, not a training run.")
        print("[act1] Install with `pip install -e '.[torch]'` on the target cluster.")
        return 0 if args.dry_run else 2

    if args.dry_run:
        print("[act1] dry-run OK — environment and config validated.")
        return 0

    print("[act1] BUILD PHASE: refusing to fabricate a training run.")
    print("[act1] Wire the real loop here on the cluster; see docs/TRAINING.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
