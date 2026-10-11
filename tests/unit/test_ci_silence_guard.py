"""Silence guard — CI must never hide a failure (GOAL.md Articles 5, 7; ADR-0011 fast lane).

A run that reports success while a step failed is worse than a red run: it converts a known
failure into a believed success. Session-0006 produced exactly that — the labeling step died
with `unrecognized arguments: --label`, `|| true` swallowed it, and the run went green.

This test scans the tree for the idioms that swallow a status code and **fails on any hit** that
is not listed, with a written justification, in ``tests/silence_guard_allowlist.txt``.

Patterns (each has an id used in the allowlist):

  ``shell-or-true``        ``|| true`` in a shell step or Makefile recipe
  ``shell-or-exit0``       ``|| exit 0``
  ``continue-on-error``    a workflow step allowed to fail silently
  ``set-plus-e``           ``set +e`` (errexit disabled)
  ``except-pass``          a Python ``except ...: pass`` that swallows an exception

The allowlist is deliberately a *file* with justifications rather than a code list: adding an
exception requires writing down why, which is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).resolve().parents[1] / "silence_guard_allowlist.txt"

# Directories that are not source and must never be scanned.
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "data",  # data lives outside git (Article 8) and is not code
}

PATTERNS: dict[str, re.Pattern[str]] = {
    "shell-or-true": re.compile(r"\|\|\s*true\b"),
    "shell-or-exit0": re.compile(r"\|\|\s*exit\s+0\b"),
    "continue-on-error": re.compile(r"^\s*continue-on-error\s*:\s*(?!false\b)\S+", re.MULTILINE),
    "set-plus-e": re.compile(r"\bset\s+\+e\b"),
    "except-pass": re.compile(r"except[^\n]*:\s*\n\s*pass\b"),
}

# Files that define the patterns or discuss them: the guard itself and its allowlist must not
# trip the guard. Naming these explicitly (rather than skipping all tests) keeps the scan honest.
SELF_EXEMPT = {
    "tests/unit/test_ci_silence_guard.py",
    "tests/silence_guard_allowlist.txt",
}

SCAN_SUFFIXES = {".yml", ".yaml", ".py", ".mk", ".sh"}
SCAN_NAMES = {"Makefile"}


def _allowlisted() -> list[tuple[str, str, str]]:
    """Parse the allowlist into ``(path, pattern_id, justification)`` triples."""
    entries: list[tuple[str, str, str]] = []
    for raw in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 3 or not all(p.strip() for p in parts):
            raise ValueError(f"allowlist line is not '<path>|<pattern>|<justification>': {line!r}")
        entries.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return entries


def _iter_source_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SCAN_SUFFIXES or path.name in SCAN_NAMES:
            yield path


def _code_only(path: Path, text: str) -> str:
    """Blank out comments and string literals, preserving line numbers.

    A pattern *mentioned* in a comment or a docstring is not a swallow — matching it would
    produce false positives and, worse, train the project to add allowlist entries for prose.
    For Python we blank the COMMENT and STRING tokens via ``tokenize`` (robust against
    multi-line docstrings); for shell/YAML/Makefile we strip ``#`` to end of line.
    """
    if path.suffix == ".py":
        import io
        import tokenize

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return text  # unparsable: scan it as-is rather than skip it silently
        # Collect the exact column span to blank on each line. `None` as an end means
        # "to end of line" (the opening line of a multi-line string).
        blanks: dict[int, list[tuple[int, int | None]]] = {}

        def _blank(row: int, start: int, end: int | None) -> None:
            blanks.setdefault(row, []).append((start, end))

        for tok in tokens:
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            if srow == erow:
                _blank(srow, scol, ecol)
            else:
                _blank(srow, scol, None)
                for row in range(srow + 1, erow):
                    _blank(row, 0, None)
                _blank(erow, 0, ecol)

        lines = text.split("\n")
        for row, spans in blanks.items():
            line = lines[row - 1]
            for start, end in sorted(spans, reverse=True):
                stop = len(line) if end is None else end
                line = line[:start] + " " * max(0, stop - start) + line[stop:]
            lines[row - 1] = line
        return "\n".join(lines)
    # Shell / YAML / Makefile: a `#` comment runs to end of line. (No here-doc handling; a
    # here-doc containing `|| true` would be code and should be flagged anyway.)
    lines = []
    for line in text.splitlines(keepends=True):
        hash_at = line.find("#")
        lines.append(
            line[:hash_at] + ("\n" if line.endswith("\n") else "") if hash_at >= 0 else line
        )
    return "".join(lines)


def _scan() -> list[tuple[str, str, int, str]]:
    """Return every swallow hit as ``(rel_path, pattern_id, line_no, line_text)``."""
    hits: list[tuple[str, str, int, str]] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in SELF_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scannable = _code_only(path, text)
        for pattern_id, pattern in PATTERNS.items():
            for match in pattern.finditer(scannable):
                line_no = scannable.count("\n", 0, match.start()) + 1
                line_text = text.splitlines()[line_no - 1].strip()
                hits.append((rel, pattern_id, line_no, line_text))
    return hits


def test_allowlist_is_well_formed() -> None:
    """Every allowlist entry names a file that exists, a known pattern and a real reason."""
    entries = _allowlisted()
    for path, pattern_id, reason in entries:
        assert (REPO_ROOT / path).exists(), f"allowlist names a missing file: {path}"
        assert pattern_id in PATTERNS, f"allowlist names an unknown pattern: {pattern_id}"
        assert (
            len(reason) >= 20
        ), f"justification too short to be a reason ({path}|{pattern_id}): {reason!r}"


def test_no_unjustified_silence() -> None:
    """No swallow of a failure status anywhere in the tree, unless justified in writing."""
    allowed = {(path, pattern_id) for path, pattern_id, _ in _allowlisted()}
    unexplained = [h for h in _scan() if (h[0], h[1]) not in allowed]
    if unexplained:
        report = "\n".join(
            f"  {path}:{line_no}  [{pattern_id}]  {line_text}"
            for path, pattern_id, line_no, line_text in unexplained
        )
        pytest.fail(
            "CI silence detected — a failure can be swallowed here.\n"
            "Either remove the swallow, or add a line to "
            f"{ALLOWLIST.relative_to(REPO_ROOT).as_posix()}:\n"
            "    <path>|<pattern-id>|<why it is safe>\n\n" + report
        )


def test_guard_can_actually_see_a_swallow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not be vacuous: plant a swallow and prove the scanner flags it.

    Without this self-test a typo in a regex would make the guard silently pass forever —
    the very failure mode the guard exists to prevent.
    """
    target = REPO_ROOT / "tests" / "_silence_probe.tmp.sh"
    try:
        target.write_text("#!/bin/sh\npytest -q || true\n", encoding="utf-8")
        hits = [h for h in _scan() if h[0] == "tests/_silence_probe.tmp.sh"]
        assert hits, "the scanner failed to flag a planted `|| true`"
        assert hits[0][1] == "shell-or-true"
    finally:
        target.unlink(missing_ok=True)
