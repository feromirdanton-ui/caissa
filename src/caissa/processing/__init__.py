"""Processing organ: streaming PGN parser, filters, sharding, QA (GOAL.md §5 Phase 2)."""

from caissa.processing.pgn import iter_games, stream_pgn

__all__ = ["iter_games", "stream_pgn"]
