"""Lichess bulk loader — resumable + sha256 + manifest (GOAL.md §7, §11 task 4).

Real implementation of the resumable download protocol (Article 4):
  - writes to `<target>.part` and resumes with HTTP `Range`;
  - *validates* the range response: a server (or proxy) that ignores `Range` would otherwise
    be appended to the partial file and silently corrupt the shard, so the partial file is
    discarded and the download restarts from zero instead;
  - **certifies** the received bytes against a trusted reference before publishing — a manifest
    is a certification of truth, not a journal (Article 14). A `200`-instead-of-`206` or a `416`
    is a loud failure, and a body that cannot be confirmed is never recorded;
  - verifies the final byte count against that reference before publishing the file;
  - records sha256 in the manifest and re-verifies it on later runs, so a corrupted local file
    is detected and re-downloaded rather than trusted (Article 5);
  - never re-downloads a completed, verified shard.

**Certification (Article 14).** A body is published only when its length matches a *trusted*
reference. The server's own `Content-Length` (via `HEAD`) is one such reference, but it is not
always available: a `HEAD`-less mirror returns `500`, and then nothing in the `GET` response
contradicts a truncated body — the loader would publish half a shard and write a manifest
certifying it. To close that hole the reference may also come *out of band* as
`expected_bytes` (a published volume, a manifest from a previous run, an index file). When
neither exists, the download is refused with `UnverifiableDownload` rather than recorded.

This module is NOT run during the build phase (it would download GB). It is exercised by
`tests/unit/test_lichess_bulk.py` against a local HTTP fixture, including the
Range-ignoring-server, stale-part, truncated-body and unverifiable-source cases.
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


class UnverifiableDownload(IncompleteDownload):
    """The transfer finished but cannot be *certified* (Article 14).

    Raised when no trusted reference exists to confirm the received bytes, or when the received
    bytes contradict the reference we do have. A manifest is a certification of truth, so it is
    never written for a body we could not confirm — this is a loud failure, not a silent record.
    """


def _remote_size(url: str) -> int | None:
    """Advertised length of the resource, or None when the server will not say."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
    except requests.RequestException:
        return None
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


def _content_range_total(header: str) -> int | None:
    """Parse the total size out of `Content-Range: bytes 100-199/200` or `bytes */200`."""
    if not header.startswith(_RANGE_PREFIX):
        return None
    tail = header[len(_RANGE_PREFIX) :].strip()
    if "/" not in tail:
        return None
    total_text = tail.split("/", 1)[1].strip()
    try:
        return int(total_text)
    except ValueError:
        return None


def _resolve_reference(url: str, target: Path, expected_bytes: int | None) -> int | None:
    """Decide the trusted length for `cache`, or fail loudly when two references disagree.

    `expected_bytes` is out-of-band and authoritative; the server's `HEAD` length is used only
    when no out-of-band reference exists. If both are present and contradict each other, neither
    can be trusted and we refuse (Article 14).
    """
    advertised = _remote_size(url)
    if expected_bytes is not None and advertised is not None and expected_bytes != advertised:
        raise UnverifiableDownload(
            f"{target.name}: the reference size ({expected_bytes} bytes) contradicts the "
            f"server's Content-Length ({advertised} bytes); refusing to certify either."
        )
    return expected_bytes if expected_bytes is not None else advertised


def download(
    url: str,
    target: Path,
    chunk: int = 1 << 20,
    *,
    expected_bytes: int | None = None,
) -> Path:
    """Download `url` to `target`, resuming from a `.part` file if present.

    Resume is honoured only when the server answers `206 Partial Content` for the requested
    offset. Anything else — a `200` carrying the whole body, a missing or mismatched
    `Content-Range`, or a `416` for a stale part — discards the partial file and restarts
    cleanly, because appending would produce a corrupt artifact that still looks complete.

    The finished body is published only when its length matches a trusted reference: the
    caller's `expected_bytes`, or failing that the server's advertised `Content-Length`. With no
    reference the body is unverifiable and the call raises `UnverifiableDownload`.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    trusted = _resolve_reference(url, target, expected_bytes)

    # A part larger than the trusted resource means the resource changed under us (or the part
    # is from a different, larger resource): drop it rather than resume into nonsense.
    if existing and trusted is not None:
        if existing > trusted:
            part.unlink()
            existing = 0
        elif existing == trusted:
            part.replace(target)
            return target

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if existing and resp.status_code == 416:
            # The part is past the end of the resource. A 416 still advertises the true total
            # (`Content-Range: bytes */N`), which is the one trustworthy measure we can get
            # from a server whose HEAD fails — adopt it as the reference and restart.
            total = _content_range_total(resp.headers.get("Content-Range", ""))
            resp.close()
            part.unlink()
            if trusted is None and total is not None:
                return download(url, target, chunk=chunk, expected_bytes=total)
            return download(url, target, chunk=chunk, expected_bytes=trusted)
        resp.raise_for_status()

        if existing:
            if resp.status_code != 206:
                resp.close()
                part.unlink()
                return download(url, target, chunk=chunk, expected_bytes=trusted)
            start = _content_range_start(resp.headers.get("Content-Range", ""))
            if start != existing:
                resp.close()
                part.unlink()
                return download(url, target, chunk=chunk, expected_bytes=trusted)

        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)

    written = part.stat().st_size
    if trusted is None:
        raise UnverifiableDownload(
            f"{part.name}: {written} bytes received but the source advertises no length and no "
            "reference was supplied; refusing to certify an unmeasurable body (Article 14). "
            "Pass expected_bytes (a published volume, index file or prior manifest)."
        )
    if written != trusted:
        raise UnverifiableDownload(
            f"{part.name}: {written} bytes written but the trusted reference is {trusted}; "
            "refusing to publish or record an uncertified shard (Article 14). The .part file "
            "is kept so the next run can resume."
        )
    part.replace(target)
    return target


def fetch_month(
    month: str,
    dest_dir: Path = Path("data/raw/lichess"),
    *,
    expected_bytes: int | None = None,
) -> Manifest:
    """Fetch one Lichess standard month, resumable, and write its manifest.

    A shard is skipped only when the local file still matches the manifest (bytes + sha256);
    otherwise it is re-downloaded. The manifest is written through `write_manifest`, so
    `fetched_at` is populated (Article 8) — and it is written only after the body has been
    certified against a trusted reference (Article 14). `expected_bytes` supplies that
    reference out of band; without it the server's `Content-Length` must be available.
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

    download(url, target, expected_bytes=expected_bytes)
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
    parser.add_argument(
        "--expected-bytes",
        type=int,
        default=None,
        help="trusted size reference (published volume / index); required when HEAD fails",
    )
    args = parser.parse_args(argv)
    fetch_month(args.month, Path(args.dest), expected_bytes=args.expected_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
