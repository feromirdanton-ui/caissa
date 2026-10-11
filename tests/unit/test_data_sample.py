"""`make data-sample` orchestration — the Phase 1 entry point, with the network stubbed.

The orchestration is tested (not skipped) by replacing the three source loaders with
fakes that write real bytes. What this pins down is the part that is easy to get wrong:
the QA-lite report must be *measured* from what the loaders produced, the disk quota must
be honoured, and `.part` hygiene must be reported truthfully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from caissa.acquisition import sample as sample_mod
from caissa.acquisition.lichess_bulk import PrefixResult
from caissa.acquisition.manifest import Manifest


def _write_day(path: Path, games: int) -> None:
    """Write a small PGN archive the QA scan can read (games with Elo + Result headers)."""
    import zstandard as zstd

    blocks = []
    for i in range(games):
        blocks.append(
            f'[Event "Rated Blitz game {i}"]\n[White "a"]\n[Black "b"]\n'
            f'[Result "1-0"]\n[WhiteElo "{1000 + i}"]\n[BlackElo "{1500 + i}"]\n\n1. e4 e5 1-0\n\n'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstd.ZstdCompressor().compress("".join(blocks).encode()))


@pytest.fixture
def config(tmp_path: Path) -> Path:
    cfg = {
        "lichess_month": "2013-01",
        "lichess_prefix_bytes": 8_388_608,
        "puzzles_limit": 1000,
        "twic_month": "2024-01",
        "twic_max_issues": 1,
        "disk_quota_gb": 0.001,
    }
    path = tmp_path / "data-sample.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_run_measures_its_own_output(tmp_path: Path, config: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_prefix(url: str, target: Path, prefix_bytes: int, chunk: int = 0) -> PrefixResult:
        _write_day(target, games=12)
        import hashlib

        data = target.read_bytes()
        return PrefixResult(
            target, prefix_bytes, len(data), hashlib.sha256(data).hexdigest(), 0, 99, False
        )

    def fake_puzzles(limit: int = 1000) -> Manifest:
        return Manifest(
            source="mock://puzzles",
            sha256="ab" * 32,
            bytes=123,
            rows=7,
            license="CC0-1.0 (Lichess)",
            shards=["puzzles_sample.csv"],
        )

    def fake_twic(month: str, max_issues: int | None = None) -> Manifest:
        return Manifest(
            source="mock://twic",
            sha256="cd" * 32,
            bytes=456,
            rows=3,
            license="TWIC",
            shards=["twic1500g.zip"],
        )

    monkeypatch.setattr(sample_mod, "download_prefix", fake_prefix)
    monkeypatch.setattr(sample_mod, "fetch_puzzles", fake_puzzles)
    monkeypatch.setattr(sample_mod, "fetch_twic_month", fake_twic)

    report = sample_mod.run(str(config))

    assert report["lichess_day"]["games"] == 12, "QA-lite must count the games it actually read"
    assert report["lichess_day"]["elo_mean"] is not None, "Elo must be measured, not assumed"
    assert report["lichess_day"]["elo_histogram"], "a histogram must be produced"
    assert report["puzzles"]["rows"] == 7
    assert report["twic"]["issues"] == 1
    assert report["hygiene"]["part_files_left"] == [], "a clean run leaves no .part files"

    written = json.loads((tmp_path / "data" / "qa" / "qa-lite.json").read_text())
    assert written["lichess_day"]["games"] == 12, "the report on disk must match the return value"


def test_run_refuses_below_the_disk_quota(tmp_path: Path, config: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sample_mod, "disk_free_gb", lambda path=Path("."): 0.0001)

    with pytest.raises(SystemExit):
        sample_mod.run(str(config))
