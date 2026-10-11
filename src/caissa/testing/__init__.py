"""Caissa-1 testing helpers — fixture generators for realism (ADR-0011, fast lane).

Nothing here is a chess component; these are *fixture builders* that make the defects the
runner found (sessions 0005/0006) reproducible in the local test suite, forever.
"""

from caissa.testing.realism import (
    build_broken_pgn,
    build_valid_pgn,
    to_multiframe_zst,
    truncate_at_frame_boundary,
    truncate_mid_frame,
    write_realism_fixtures,
)

__all__ = [
    "build_broken_pgn",
    "build_valid_pgn",
    "to_multiframe_zst",
    "truncate_at_frame_boundary",
    "truncate_mid_frame",
    "write_realism_fixtures",
]
