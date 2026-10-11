"""Shared pytest fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Requiem-1 is defined by the constitution, so it is read FROM the constitution instead of
# being duplicated as a literal here: if GOAL.md §8 changes, this fixture changes with it
# (Article 10 — GOAL is the single source of truth). Session-0002 P1-7.
_FEN_BLOCK = re.compile(
    r"```\s*\n\s*([1-8rnbqkpRNBQKP/]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+)\s*\n\s*```"
)


def _requiem_1_fen_from_goal() -> str:
    text = (REPO_ROOT / "GOAL.md").read_text(encoding="utf-8")
    section = text.split("## 8. Requiem-1", 1)[1].split("## 9.", 1)[0]
    match = _FEN_BLOCK.search(section)
    if match is None:
        raise RuntimeError(
            "GOAL.md §8 no longer contains a FEN block — Requiem-1 cannot be located. "
            "The constitution changed shape; fix this fixture before anything else."
        )
    return match.group(1)


REQUIEM_1_FEN = _requiem_1_fen_from_goal()


@pytest.fixture
def requiem_1() -> chess.Board:
    """The eternal regression position, loaded from GOAL.md §8. Never delete."""
    return chess.Board(REQUIEM_1_FEN)


@pytest.fixture
def start_board() -> chess.Board:
    return chess.Board()


@pytest.fixture(autouse=True)
def _isolate_manifest_dir(tmp_path_factory, monkeypatch):
    """No test may write into the repository's data/manifests/ (Article 14).

    The manifest directory is anchored at the repo root (P2-1), so a test that exercises
    the loader would otherwise drop a fake manifest into the real tree — a manifest is a
    certification of truth, not a test artifact. Redirect it to a per-test temp directory.
    """
    from caissa.acquisition import manifest as manifest_module

    isolated = tmp_path_factory.mktemp("manifests")
    monkeypatch.setattr(manifest_module, "MANIFEST_DIR", isolated)
    return isolated
