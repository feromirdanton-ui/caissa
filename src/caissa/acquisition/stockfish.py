"""Pinned Stockfish acquisition (GOAL.md §7; session-0007).

Stockfish is a *tool*, not a dataset: a fixed version, verified by sha256, recorded in a manifest
**only after** the written file's hash is checked (Article 14, ADR-0009). The binary is cached on
runners by its hash, so a re-run is free and a version bump is an explicit edit here — never a
silent "latest".

Honesty: the version and sha256 below are the ones this project actually fetched and verified.
If upstream republishes, the check fails loudly rather than quietly labelling with a different
engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Fetched and verified in session-0007 from the official-stockfish/Stockfish release `sf_19`.
STOCKFISH_VERSION = "sf_19"
STOCKFISH_TARBALL = "stockfish-linux-x86-64-universal.tar.gz"
STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    f"{STOCKFISH_VERSION}/{STOCKFISH_TARBALL}"
)
# The sha256 of the tarball this project verified (session-0007, official-stockfish/Stockfish sf_19).
# A mismatch is a loud failure, never a silent "different but fine".
STOCKFISH_TARBALL_SHA256 = "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
# The binary path inside the tarball.
STOCKFISH_MEMBER = "stockfish/stockfish-linux-x86-64-universal"

MANIFEST_NAME = "stockfish.json"


@dataclass(frozen=True)
class StockfishManifest:
    """What was actually fetched and checked. ``binary_sha256`` is measured, never declared."""

    version: str
    url: str
    tarball_sha256: str
    tarball_sha256_verified: bool
    binary_sha256: str
    path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Raise unless ``path`` hashes to ``expected``; return the digest on success."""
    actual = _sha256(Path(path))
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def ensure_stockfish(dest: str | Path = "data/tools") -> StockfishManifest:
    """Download, verify, extract and record a pinned Stockfish.

    The manifest is written **after** the binary's sha256 is verified, so it is a certification of
    a fact that was checked — the Article 14 rule (session-0005/0006).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    binary = dest / "stockfish"
    pointer = dest / "stockfish.path"
    tarball = dest / STOCKFISH_TARBALL

    if not tarball.exists():
        print(f"[stockfish] downloading {STOCKFISH_URL}")
        urllib.request.urlretrieve(STOCKFISH_URL, tarball)  # noqa: S310 - fixed https URL

    # Article 14: verify the TARBALL we actually downloaded before trusting its contents.
    tarball_digest = _sha256(tarball)
    tarball_ok = tarball_digest == STOCKFISH_TARBALL_SHA256
    if not tarball_ok:
        raise ValueError(
            "Stockfish tarball sha256 mismatch: "
            f"expected {STOCKFISH_TARBALL_SHA256}, got {tarball_digest}. "
            "Refusing to publish a manifest for an unverified tool (Article 14)."
        )

    if binary.exists() and pointer.exists():
        print(f"[stockfish] reusing {binary} (cached)")
    else:
        # Extract into a scratch dir and move the single binary into place. Extracting straight
        # into ``dest`` would create ``dest/stockfish/`` (the archive's own top-level folder),
        # while ``binary`` is the *file* ``dest/stockfish`` — and moving a file onto the directory
        # of the same name raises ``shutil.Error: Destination path already exists`` every time.
        # Nothing in the local loop caught this because Stockfish was already on PATH; the first
        # clean-runner fetch was the only place the extract+move path actually executed.
        with tempfile.TemporaryDirectory() as scratch:
            with tarfile.open(tarball, "r:gz") as tar:
                member = tar.getmember(STOCKFISH_MEMBER)
                member.mode = 0o755
                tar.extract(member, path=scratch, filter="data")
            extracted = Path(scratch) / STOCKFISH_MEMBER
            if binary.exists():
                # A previous partial run could have left the archive's folder here; clear it so
                # the move below cannot silently nest the binary inside a directory.
                shutil.rmtree(binary) if binary.is_dir() else binary.unlink()
            shutil.move(str(extracted), str(binary))
        binary.chmod(0o755)
        pointer.write_text(str(binary.resolve()), encoding="utf-8")

    version_probe = subprocess.run(  # noqa: S603 - fixed local binary
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    reported = (version_probe.stdout or version_probe.stderr).strip().splitlines()[:1]
    print(f"[stockfish] {binary} → {reported}")

    manifest = StockfishManifest(
        version=STOCKFISH_VERSION,
        url=STOCKFISH_URL,
        tarball_sha256=tarball_digest,
        tarball_sha256_verified=tarball_ok,
        binary_sha256=_sha256(binary),
        path=str(binary.resolve()),
    )
    (manifest_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[stockfish] manifest written: {manifest_dir / MANIFEST_NAME} "
        f"(tarball verified={tarball_ok}, binary sha256={manifest.binary_sha256[:16]}…)"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a pinned Stockfish, sha256-recorded")
    parser.add_argument("--dest", default="data/tools")
    args = parser.parse_args(argv)
    ensure_stockfish(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
