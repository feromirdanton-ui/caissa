"""Ghost protection (ADR-0007) — the suite must test THIS checkout, not a previous one.

The incident, observed live during session-0005: the work moved to a new checkout while an
editable install of ``caissa`` from the OLD checkout stayed active on ``sys.path``. Then::

    $ python -c "import caissa; print(caissa.__file__)"
    /workspace/caissa-v3/src/caissa/__init__.py     # the previous session's tree

``pytest`` collected the new tests but imported the old package — a green run that proves
nothing about the tree it claims to test. The guard below turns that silent wrong-answer
into a loud failure.
"""

from __future__ import annotations

from pathlib import Path

import caissa

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_caissa_resolves_inside_this_checkout() -> None:
    """``caissa`` must be imported from this checkout — not from site-packages, not from a sibling."""
    resolved = Path(caissa.__file__).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"`caissa` resolved to {resolved}, outside this checkout ({REPO_ROOT}).\n"
        "A previous editable install is still active. Reinstall from THIS checkout:\n"
        "    pip install -e .\n"
        "(ADR-0007 — the ghost of a previous session's install.)"
    )


def test_caissa_resolves_to_the_src_layout() -> None:
    """The resolved path must be ``<checkout>/src/caissa/__init__.py``, not a flat/copied install."""
    resolved = Path(caissa.__file__).resolve()
    assert resolved.parent.name == "caissa", f"unexpected package dir: {resolved.parent}"
    assert resolved.parent.parent.name == "src", (
        f"`caissa` resolved to {resolved}, which is not this checkout's src/ layout. "
        "This is the ghost of another session's install (ADR-0007)."
    )
