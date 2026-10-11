"""`make data-sample` — Phase 1 DoD entry point (GOAL.md §5, §10).

Pulls the honest minimum the directive fixes: **one day of Lichess bulk** (the head of one
month's archive), **1000 puzzles**, and **one TWIC month** — resumably, with every guarantee of
`lichess_bulk` (`.part` + `Range`, atomic rename, sha256 before the manifest, loud failure on an
unverifiable body). A full Lichess month is fetched only when the environment has the gigabytes;
the disk quota is printed, checked and recorded in the QA report.

Nothing enters git: the archives live under `data/` (path-based `.gitignore`), and only the
manifests and this QA-lite report are produced. Honesty rules (Article 5): a small honest sample
beats a silent truncation, and both beat a fabricated volume.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
import zstandard as zstd

from caissa.acquisition.lichess_bulk import LICHESS_STANDARD, download_prefix
from caissa.acquisition.manifest import ManifestStore
from caissa.acquisition.puzzles import fetch_puzzles
from caissa.acquisition.twic import fetch_twic_month

QA_DIR = Path("data/qa")
QA_REPORT = "qa-lite.json"

_RESULT = re.compile(r'^\[Result "(.+?)"\]')
_WHITE_ELO = re.compile(r'^\[WhiteElo "(\d+)"\]')
_BLACK_ELO = re.compile(r'^\[BlackElo "(\d+)"\]')


def disk_free_gb(path: Path = Path(".")) -> float:
    """Free space of the volume that will hold the data, in GB."""
    target = path if path.exists() else Path(".")
    return shutil.disk_usage(str(target)).free / (1 << 30)


def _lines(path: Path, chunk: int = 1 << 20):
    """Yield decoded lines from a (possibly truncated) zstd file."""
    dctx = zstd.ZstdDecompressor()
    pending = b""
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        while True:
            try:
                block = reader.read(chunk)
            except zstd.ZstdError:
                break
            if not block:
                break
            pending += block
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                yield line.decode("utf-8", "replace")


def scan_month_prefix(path: Path, *, max_games: int = 2000) -> dict:
    """QA-lite over the head of a Lichess month: game count and Elo distribution.

    The head is a *prefix* of a compressed stream, so it ends mid-stream; the decoder is read
    to the last complete block and the tail is dropped. Every number is measured from the bytes
    on disk — none is assumed.
    """
    games = 0
    elos: list[int] = []
    results: dict[str, int] = {}
    for line in _lines(path):
        if line.startswith("[Event "):
            games += 1
        elif line.startswith("[WhiteElo "):
            if games <= max_games:
                m = _WHITE_ELO.match(line)
                if m:
                    elos.append(int(m.group(1)))
        elif line.startswith("[BlackElo "):
            if games <= max_games:
                m = _BLACK_ELO.match(line)
                if m:
                    elos.append(int(m.group(1)))
        elif line.startswith("[Result "):
            m = _RESULT.match(line)
            if m:
                results[m.group(1)] = results.get(m.group(1), 0) + 1

    histogram: dict[str, int] = {}
    for elo in elos:
        if elo <= 0:
            continue
        low = (elo // 200) * 200
        key = f"{low}-{low + 199}"
        histogram[key] = histogram.get(key, 0) + 1

    return {
        "games": games,
        "elo_samples": len(elos),
        "elo_mean": round(sum(elos) / len(elos), 1) if elos else None,
        "elo_min": min(elos) if elos else None,
        "elo_max": max(elos) if elos else None,
        "elo_histogram": dict(sorted(histogram.items())),
        "results": dict(sorted(results.items())),
    }


def run(config_path: str, *, allow_large: bool = False) -> dict:
    """Execute the sample pull and return the QA-lite report (also written to disk)."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    lichess_month = str(cfg["lichess_month"])
    prefix_bytes = int(cfg.get("lichess_prefix_bytes", 50_000_000))
    puzzles_limit = int(cfg.get("puzzles_limit", 1000))
    twic_month = str(cfg["twic_month"])
    twic_max_issues = cfg.get("twic_max_issues")
    quota_gb = float(cfg.get("disk_quota_gb", 1.0))

    free = disk_free_gb()
    print(f"[data-sample] config={config_path} free={free:.2f} GiB quota={quota_gb:.2f} GiB")
    if free < quota_gb:
        message = (
            f"[data-sample] free space {free:.2f} GiB is below the quota {quota_gb:.2f} GiB; "
            "refusing to start (Article 6.6 resource hygiene)."
        )
        if not allow_large:
            raise SystemExit(message)
        print(message + " --allow-large was given, continuing.")

    started = datetime.now(timezone.utc)
    report: dict = {
        "session": "0005",
        "started_at": started.isoformat(),
        "disk_free_gib_at_start": round(free, 3),
        "disk_quota_gib": quota_gb,
        "config": config_path,
    }

    # 1. one day of Lichess bulk — the head of one month's archive.
    day_target = Path("data/raw/lichess") / f"lichess_day_{lichess_month}.pgn.zst"
    result = download_prefix(LICHESS_STANDARD.format(month=lichess_month), day_target, prefix_bytes)
    lichess_qa = scan_month_prefix(result.target)
    lichess_qa.update(
        {
            "month": lichess_month,
            "prefix_bytes": result.prefix_bytes,
            "sha256": result.sha256,
            "resumed_from": result.resumed_from,
            "fetched_bytes_this_run": result.fetched_bytes,
            "remote_total_bytes": result.remote_total,
        }
    )
    report["lichess_day"] = lichess_qa
    print(
        f"[data-sample] lichess {lichess_month}: {result.prefix_bytes} B prefix, "
        f"{lichess_qa['games']} games, fetched={result.fetched_bytes} resumed_from={result.resumed_from}"
    )

    # 2. Lichess puzzles, 1000-row sample.
    puzzles = fetch_puzzles(limit=puzzles_limit)
    report["puzzles"] = {
        "rows": puzzles.rows,
        "sha256": puzzles.sha256,
        "bytes": puzzles.bytes,
        "license": puzzles.license,
        "sample": puzzles.shards[0] if puzzles.shards else None,
    }

    # 3. TWIC — one measured month of weekly issues.
    twic = fetch_twic_month(twic_month, max_issues=twic_max_issues)
    report["twic"] = {
        "month": twic_month,
        "issues": len(twic.shards),
        "games": twic.rows,
        "bytes": twic.bytes,
        "license": twic.license,
        "shards": twic.shards,
    }

    # 4. .part hygiene + manifest status.
    parts = sorted(Path("data").rglob("*.part")) if Path("data").exists() else []
    store = ManifestStore()
    manifest_files = (
        sorted(p.name for p in store.root.glob("*.json")) if store.root.exists() else []
    )
    report["hygiene"] = {
        "part_files_left": [str(p) for p in parts],
        "manifests": manifest_files,
        "disk_free_gib_at_end": round(disk_free_gb(), 3),
    }
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / QA_REPORT).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[data-sample] QA-lite written to {QA_DIR / QA_REPORT}")
    print(
        f"[data-sample] done: lichess {lichess_qa['games']} games, "
        f"puzzles {puzzles.rows} rows, twic {twic.rows} games; "
        f"part files left={len(parts)}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 data sample (Phase 1)")
    parser.add_argument("--config", default="configs/data-sample.yaml")
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="proceed even when free disk is below the configured quota",
    )
    args = parser.parse_args(argv)
    run(args.config, allow_large=args.allow_large)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
