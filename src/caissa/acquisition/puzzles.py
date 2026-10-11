"""Lichess puzzles loader — resumable + sha256 + manifest (GOAL.md §7).

Puzzles are the P0 source of the Memory organ (§4.4) and of the Queens Guard golden
suite (§5 Phase 6): every record carries a FEN, a solution line, a rating and thematic
tags, which is exactly the "signature of a motive" shape the index needs.

The source is ONE static artifact (~300 MB), so there is no per-shard resume to offer;
the guarantees are the ``lichess_bulk`` ones (§3 Article 4, Article 14):

  - ``.part`` + HTTP ``Range`` resume;
  - atomic rename into place — a reader never sees a half-written file;
  - sha256 computed on the written file BEFORE the manifest is written;
  - a ``200``-instead-of-``206`` or a ``416`` is a loud failure, never a silent record;
  - an existing file is re-verified against its manifest (bytes + sha256) and
    re-downloaded when it disagrees (Article 5).

Certification needs a trusted length. The server publishes ``Content-Length`` for this
resource; when a mirror will not, the caller may supply ``expected_bytes`` (a published
volume or an index file). With neither, the body is refused — never recorded (Article 14).

**Sampling.** One streaming decompression pass both counts every row (``rows``) and writes
the first ``limit`` rows to ``data/raw/lichess/puzzles_sample.csv`` with a small rating
histogram. The sample is DATA and never enters git; the manifest is the only trace
(Article 8).
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import zstandard as zstd

from caissa.acquisition.lichess_bulk import download
from caissa.acquisition.manifest import Manifest, ManifestStore, sha256_file, write_manifest

PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
PUZZLES_SCHEMA = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags (CSV, zstd)"
HEADER = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
DEFAULT_DEST = Path("data/raw/lichess")
NAME = "lichess_puzzles"


def _iter_lines(path: Path, chunk: int = 1 << 20):
    """Yield decoded CSV lines from a zstd file, tolerating a truncated tail.

    A truncated prefix is a legitimate input for the day-sample path, so a
    ``ZstdError`` at the tail ends the iteration instead of aborting it: the lines
    decoded so far are complete CSV rows.
    """
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


def scan_and_sample(
    zst_path: Path,
    *,
    sample_path: Path | None = None,
    limit: int = 0,
    buckets: int = 400,
) -> dict:
    """One pass: count every row, and write the first ``limit`` rows to ``sample_path``.

    Returns a QA-lite summary: total rows, sample rows, the sample's rating mean/min/max
    and a coarse rating histogram (buckets of ``buckets`` Elo). Nothing is fabricated:
    every number comes from the bytes on disk.
    """
    rows = 0
    sample_rows = 0
    ratings: list[int] = []
    header_seen = False
    themes_missing = 0
    out = None
    with contextlib.ExitStack() as stack:
        if sample_path is not None and limit > 0:
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            out = stack.enter_context(open(sample_path, "w", encoding="utf-8", newline="\n"))
        for raw in _iter_lines(zst_path):
            if not header_seen:
                header_seen = True
                if out is not None:
                    out.write(raw + "\n")
                continue
            if not raw:
                continue
            rows += 1
            fields = raw.split(",")
            if len(fields) >= 8:
                if not fields[7]:
                    themes_missing += 1
                if out is not None and sample_rows < limit:
                    out.write(raw + "\n")
                    sample_rows += 1
                    with contextlib.suppress(ValueError):
                        ratings.append(int(fields[3]))

    histogram: dict[str, int] = {}
    for rating in ratings:
        low = (rating // buckets) * buckets
        key = f"{low}-{low + buckets - 1}"
        histogram[key] = histogram.get(key, 0) + 1

    summary = {
        "rows": rows,
        "sample_rows": sample_rows,
        "header_ok": header_seen,
        "themes_missing": themes_missing,
        "rating_mean": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "rating_min": min(ratings) if ratings else None,
        "rating_max": max(ratings) if ratings else None,
        "rating_histogram": dict(sorted(histogram.items())),
    }
    return summary


def fetch_puzzles(
    dest_dir: Path = DEFAULT_DEST,
    *,
    limit: int = 1000,
    sample_name: str = "puzzles_sample.csv",
    expected_bytes: int | None = None,
) -> Manifest:
    """Download the puzzles archive, verify it, sample ``limit`` rows and record a manifest.

    A completed, verified archive is skipped: the local file is checked against the
    manifest's bytes and sha256 first, so a restart never re-downloads finished work
    (Article 4). The manifest certifies the *written* file's sha256 (Article 14) and
    carries the QA-lite summary of the single scan pass.
    """
    store = ManifestStore()
    target = dest_dir / "lichess_db_puzzle.csv.zst"
    sample_path = dest_dir / sample_name

    if store.exists(NAME):
        manifest = store.load(NAME)
        if (
            target.exists()
            and target.stat().st_size == manifest.bytes
            and sha256_file(target) == manifest.sha256
        ):
            print(f"[puzzles] {NAME} already complete and verified — skipping (Article 4).")
            _resample(target, sample_path, limit)
            return manifest
        print(
            f"[puzzles] {NAME}: local file missing, truncated or hash-mismatched — re-downloading."
        )
        target.unlink(missing_ok=True)

    download(PUZZLES_URL, target, expected_bytes=expected_bytes)
    summary = scan_and_sample(target, sample_path=sample_path, limit=limit)

    manifest = Manifest(
        source=PUZZLES_URL,
        sha256=sha256_file(target),
        bytes=target.stat().st_size,
        rows=summary["rows"],
        schema=PUZZLES_SCHEMA,
        license="CC0-1.0 (Lichess)",
        notes=(
            "Lichess puzzle database (all themes + ratings). "
            f"QA-lite sample={sample_name} sample_rows={summary['sample_rows']} "
            f"rating_mean={summary['rating_mean']} histogram_buckets={json.dumps(summary['rating_histogram'])}"
        ),
        shards=[sample_name],
    )
    write_manifest(NAME, manifest)
    print(
        f"[puzzles] {NAME}: {manifest.bytes} bytes sha256={manifest.sha256[:12]}… "
        f"rows={manifest.rows} sample={sample_name}({summary['sample_rows']} rows)"
    )
    return manifest


def _resample(target: Path, sample_path: Path, limit: int) -> dict:
    """Re-emit the sample from the already-verified archive (no network)."""
    return scan_and_sample(target, sample_path=sample_path, limit=limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 Lichess puzzles loader")
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--limit", type=int, default=1000, help="rows to keep in the sample")
    parser.add_argument("--expected-bytes", type=int, default=None)
    args = parser.parse_args(argv)
    fetch_puzzles(Path(args.dest), limit=args.limit, expected_bytes=args.expected_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
