"""Motive index schema + a minimal in-memory validator (GOAL.md §4.4).

A motive is a position pattern with a signature (isolated pawn, weak f7, pawn chains,
outposts). The vector index and population are deferred until the model exists — this is the
schema and a validator, honestly labeled as a skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

KNOWN_SIGNATURES = {
    "isolated_pawn",
    "doubled_pawn",
    "weak_square",
    "pawn_chain",
    "outpost",
    "open_file",
    "backward_pawn",
}


@dataclass
class Motive:
    signature: str
    squares: list[str] = field(default_factory=list)
    note: str = ""

    def validate(self) -> None:
        if self.signature not in KNOWN_SIGNATURES:
            raise ValueError(f"unknown motive signature: {self.signature}")
        for sq in self.squares:
            if chess.parse_square(sq) < 0:
                raise ValueError(f"bad square: {sq}")


@dataclass
class MotiveIndex:
    """Skeleton index. Retrieval is a linear scan until a vector index is added (ADR pending)."""

    motives: list[Motive] = field(default_factory=list)

    def add(self, motive: Motive) -> None:
        motive.validate()
        self.motives.append(motive)

    def by_signature(self, signature: str) -> list[Motive]:
        return [m for m in self.motives if m.signature == signature]
