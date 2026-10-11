"""Unit tests for the data manifest (Article 8)."""

from __future__ import annotations

from pathlib import Path

from caissa.acquisition.manifest import Manifest, ManifestStore, sha256_file


def test_manifest_dir_is_anchored_at_repo_root() -> None:
    """P2-1: the manifest anchor is the repo root, not the CWD."""
    from caissa.acquisition.manifest import MANIFEST_DIR, repo_root

    assert (repo_root() / "pyproject.toml").exists()
    assert repo_root() / "data" / "manifests" == MANIFEST_DIR
    assert MANIFEST_DIR.is_absolute()


def test_manifest_roundtrip(tmp_path: Path) -> None:
    store = ManifestStore(root=tmp_path)
    m = Manifest(source="https://example.org/x.zst", sha256="abc", bytes=10, schema="pgn.zst")
    store.save("demo", m)
    assert store.exists("demo")
    loaded = store.load("demo")
    assert loaded.source == m.source
    assert loaded.sha256 == "abc"


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
