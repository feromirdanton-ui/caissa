"""Pinned-Stockfish acquisition — the extract→move defect, pinned so it fails LOCALLY first.

Session-0007 shipped `acquisition/stockfish.py` and it passed every local check, yet the first
clean-runner dispatch of `smoke-train.yml` failed at "Fetch Stockfish":

    shutil.Error: Destination path 'data/tools/stockfish/stockfish-linux-x86-64-universal' already exists

Root cause — an artefact, not a logic slip: the archive's member is
``stockfish/stockfish-linux-x86-64-universal``, so extracting it *straight into* ``dest`` creates
the directory ``dest/stockfish/``. The module then wants ``dest/stockfish`` to be the *binary file*
of the same name, and moving a file onto a directory of that name raises. Nothing caught it locally
because Stockfish was already on ``PATH``: the extract+move branch had simply never run anywhere
before the runner.

These tests exercise that branch **offline**, with a tiny deterministic tarball whose sha256 the
pinned constant is monkeypatched to, so the defect is now a local failing test rather than a
runner surprise (ADR-0003; the runner's job is to catch what the local loop cannot — here the
local loop can).
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from caissa.acquisition import stockfish as sf

MEMBER = "stockfish/stockfish-linux-x86-64-universal"
FAKE_BINARY = b"#!/bin/sh\necho 'Stockfish 19 by the Stockfish developers (see AUTHORS file)'\n"


def _build_tarball(path: Path) -> str:
    """Write a one-member tarball shaped exactly like the real one; return its sha256."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        info = tarfile.TarInfo(MEMBER)
        info.size = len(FAKE_BINARY)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(FAKE_BINARY))
    path.write_bytes(raw.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """A clean 'repo' in tmp_path with the tarball pre-seeded, so no network is touched."""
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / "data" / "tools"
    dest.mkdir(parents=True)
    tarball = dest / sf.STOCKFISH_TARBALL
    digest = _build_tarball(tarball)
    # Point the pinned constants at the fixture so the Article 14 check passes on real bytes.
    monkeypatch.setattr(sf, "STOCKFISH_TARBALL_SHA256", digest)
    monkeypatch.setattr(sf, "STOCKFISH_MEMBER", MEMBER)
    return dest


def test_fresh_fetch_lands_the_binary_as_a_file(pinned: Path) -> None:
    """The happy path the runner needed: ``dest/stockfish`` is the BINARY, not the archive folder."""
    manifest = sf.ensure_stockfish(pinned)

    binary = pinned / "stockfish"
    assert binary.is_file(), "the extracted binary must end up as a file at dest/stockfish"
    # The exact pre-fix residue must not exist: the archive's own folder is not a valid layout.
    assert not (pinned / "stockfish" / "stockfish-linux-x86-64-universal").exists()
    assert (pinned / "stockfish.path").is_file()
    assert manifest.tarball_sha256_verified is True
    assert Path(manifest.path) == binary.resolve()


def test_second_run_is_idempotent(pinned: Path) -> None:
    """A re-run reuses the cached binary instead of extracting again (the cache-friendly path)."""
    first = sf.ensure_stockfish(pinned)
    second = sf.ensure_stockfish(pinned)
    assert first.binary_sha256 == second.binary_sha256
    assert (pinned / "stockfish").is_file()


def test_recovers_from_a_leftover_archive_directory(pinned: Path) -> None:
    """The regression: a partial run that left ``dest/stockfish/`` as a directory must not wedge.

    This is the exact state the pre-fix code produced (tar.extract created the folder before the
    move failed). The fix must clear it and still place the binary as a file.
    """
    leftover = pinned / "stockfish"
    leftover.mkdir()
    (leftover / "stockfish-linux-x86-64-universal").write_text("stale partial extraction")

    manifest = sf.ensure_stockfish(pinned)

    assert (pinned / "stockfish").is_file(), "the stale directory must be replaced by the binary"
    assert manifest.binary_sha256 == hashlib.sha256(FAKE_BINARY).hexdigest()
