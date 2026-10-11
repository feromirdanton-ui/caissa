"""Lichess bulk loader — resumable + sha256 + manifest (GOAL.md §7, §11 task 4).

Real implementation of the resumable download protocol (Article 4):
  - writes to `<target>.part` and resumes with HTTP `Range`;
  - *validates* the range response: a server (or proxy) that ignores `Range` would otherwise
    be appended to the partial file and silently corrupt the shard, so the partial file is
    discarded and the download restarts from zero instead;
  - verifies the final byte count against the advertised length before publishing the file;
  - records sha256 in the manifest and re-verifies it on later runs, so a corrupted local file
    is detected and re-downloaded rather than trusted (Article 5);
  - never re-downloads a completed, verified shard.

This module is NOT run during the build phase (it would download GB). It is exercised by
`tests/unit/test_lichess_bulk.py` against a local HTTP fixture, including the
Range-ignoring-server, stale-part and truncated-body cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from caissa.acquisition.manifest import Manifest, ManifestStore, sha256_file, write_manifest

LICHESS_STANDARD = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"

_RANGE_PREFIX = "bytes "


class IncompleteDownload(RuntimeError):
    """The body did not reach the length the server advertised; the `.part` is kept."""


def _remote_size(url: str) -> int | None:
    """Advertised length of the resource, or None when the server will not say."""
    resp = requests.head(url, allow_redirects=True, timeout=30)
    if resp.status_code != 200:
        return None
    length = resp.headers.get("Content-Length")
    return int(length) if length else None


def _content_range_start(header: str) -> int | None:
    """Parse the first byte offset out of `Content-Range: bytes 100-199/200`."""
    if not header.startswith(_RANGE_PREFIX):
        return None
    spec = header[len(_RANGE_PREFIX) :].split("/")[0].strip()
    start_text = spec.split("-")[0].strip()
    try:
        return int(start_text)
    except ValueError:
        return None


def download(url: str, target: Path, chunk: int = 1 << 20) -> Path:
    """Download `url` to `target`, resuming from a `.part` file if present.

    Resume is honoured only when the server answers `206 Partial Content` for the requested
    offset. Anything else — a `200` carrying the whole body, a missing or mismatched
    `Content-Range`, or a `416` for a stale part — discards the partial file and restarts
    cleanly, because appending would produce a corrupt artifact that still looks complete.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    expected = _remote_size(url)

    # A part larger than the upstream resource means the upstream changed under us.
    if existing and expected is not None:
        if existing > expected:
            part.unlink()
            existing = 0
        elif existing == expected:
            part.replace(target)
            return target

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if existing and resp.status_code == 416:
            resp.close()
            part.unlink()
            return download(url, target, chunk=chunk)
        resp.raise_for_status()

        if existing:
            if resp.status_code != 206:
                resp.close()
                part.unlink()
                return download(url, target, chunk=chunk)
            start = _content_range_start(resp.headers.get("Content-Range", ""))
            if start != existing:
                resp.close()
                part.unlink()
                return download(url, target, chunk=chunk)

        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)

    written = part.stat().st_size
    if expected is not None and written != expected:
        raise IncompleteDownload(
            f"{part.name}: {written} bytes written but the server advertised {expected}; "
            "keeping the .part file so the next run can resume"
        )
    part.replace(target)
    return target


def fetch_month(month: str, dest_dir: Path = Path("data/raw/lichess")) -> Manifest:
    """Fetch one Lichess standard month, resumable, and write its manifest.

    A shard is skipped only when the local file still matches the manifest (bytes + sha256);
    otherwise it is re-downloaded. The manifest is written through `write_manifest`, so
    `fetched_at` is populated (Article 8).
    """
    url = LICHESS_STANDARD.format(month=month)
    name = f"lichess_standard_{month}"
    store = ManifestStore()
    target = dest_dir / f"lichess_db_standard_rated_{month}.pgn.zst"

    if store.exists(name):
        manifest = store.load(name)
        if (
            target.exists()
            and target.stat().st_size == manifest.bytes
            and sha256_file(target) == manifest.sha256
        ):
            print(f"[lichess] {name} already complete and verified — skipping (Article 4).")
            return manifest
        print(
            f"[lichess] {name}: local file missing, truncated or hash-mismatched — re-downloading."
        )
        target.unlink(missing_ok=True)
        target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)

    download(url, target)
    manifest = Manifest(
        source=url,
        sha256=sha256_file(target),
        bytes=target.stat().st_size,
        schema="pgn.zst",
        license="CC0-1.0 (Lichess)",
        notes="Lichess rated standard games, one month.",
    )
    write_manifest(name, manifest)
    print(f"[lichess] {name}: {manifest.bytes} bytes sha256={manifest.sha256[:12]}…")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caïssa-1 Lichess bulk loader")
    parser.add_argument("--month", required=True, help="e.g. 2013-01")
    parser.add_argument("--dest", default="data/raw/lichess")
    args = parser.parse_args(argv)
    fetch_month(args.month, Path(args.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
