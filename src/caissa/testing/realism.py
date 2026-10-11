"""Realism fixture generator — the lessons of sessions 0005/0006, pinned forever.

Two defects escaped the local suite because every fixture was *unrealistically clean*:

1. **session-0006 — a Lichess archive is a concatenation of zstd frames.** The local fixtures
   were single-frame, so the parser read only the first frame and 73 tests stayed green while
   the runner went red.
2. **session-0005 — a truncated byte prefix.** The day-prefix is a *prefix of a compressed
   stream*: it ends mid-frame. A parser that treats "0 bytes returned" as "archive complete"
   silently reports a short archive as a whole one.

This module builds fixtures that carry those shapes, so a future regression fails **locally**
first. It is a fixture generator, not a component: it produces bytes, never opinions.

Every builder is a pure function of its inputs (deterministic for a given seed), so a fixture
built in CI is byte-identical to one built locally — otherwise the lesson would not transfer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd

# A minimal, valid header block. Lichess writes "Rated <speed> game"; the filter of ADR-0010
# keys on the "Rated" prefix, so the fixtures must carry it or they would test the wrong path.
_HEADER = (
    '[Event "Rated Blitz game"]\n'
    '[Site "https://lichess.org/{site}"]\n'
    '[Date "2024.{month:02d}.{day:02d}"]\n'
    '[White "White{idx}"]\n'
    '[Black "Black{idx}"]\n'
    '[Result "{result}"]\n'
    '[WhiteElo "{welo}"]\n'
    '[BlackElo "{belo}"]\n'
    '[TimeControl "300+0"]\n'
    '[Opening "Sicilian Defence"]\n'
    '[ECO "B20"]\n'
)

# Two short, fully legal lines. Everything before the result token is a real game.
_LINE_SICILIAN = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
_LINE_QGD = "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 O-O"


@dataclass(frozen=True)
class RealismPaths:
    """Where :func:`write_realism_fixtures` put each shape of fixture."""

    valid_multi_frame: Path
    truncated_mid_frame: Path
    truncated_at_boundary: Path
    broken_pgn: Path
    valid_single_frame: Path


def _game_block(index: int, rng: random.Random, *, result: str | None = None) -> str:
    welo = 1200 + rng.randrange(0, 1200)
    belo = 1200 + rng.randrange(0, 1200)
    line = _LINE_SICILIAN if index % 2 == 0 else _LINE_QGD
    if result is None:
        result = ["1-0", "0-1", "1/2-1/2"][index % 3]
    header = _HEADER.format(
        site=f"{index:04d}{rng.randrange(0, 1 << 24):06x}",
        month=1 + (index % 12),
        day=1 + (index % 28),
        idx=index,
        result=result,
        welo=welo,
        belo=belo,
    )
    return f"{header}\n{line} {result}\n\n"


def build_valid_pgn(games: int = 4, *, seed: int = 0) -> bytes:
    """A syntactically sound PGN payload of ``games`` games (no compression)."""
    rng = random.Random(seed)
    return "".join(_game_block(i, rng) for i in range(games)).encode()


def build_broken_pgn(games: int = 4, *, seed: int = 0, broken_index: int = 1) -> bytes:
    """A PGN payload where exactly one game is corrupt and the rest are valid.

    The corrupt game carries an **illegal SAN move** (``2. Ke3`` after ``1. e4 e5``), which is
    the one shape python-chess reliably flags in ``game.errors``. The generator's contract is
    "exactly one bad game", so the accompanying test can assert ``skipped == 1`` exactly rather
    than "at least one" — a looser assertion would hide a second regression.
    """
    rng = random.Random(seed)
    blocks: list[str] = []
    for i in range(games):
        if i == broken_index:
            header = _HEADER.format(
                site=f"broken{i:04d}",
                month=1,
                day=2,
                idx=i,
                result="1-0",
                welo=1500,
                belo=1500,
            )
            # 1. e4 e5 is legal; Ke3 is not (the king is boxed in). python-chess records
            # `illegal san: 'Ke3' ...` in game.errors.
            blocks.append(f"{header}\n1. e4 e5 2. Ke3 1-0\n\n")
        else:
            blocks.append(_game_block(i, rng))
    return "".join(blocks).encode()


def to_multiframe_zst(payload: bytes, frames: int = 4) -> bytes:
    """Compress ``payload`` into ``frames`` **concatenated** zstd frames.

    A Lichess archive has this shape. Splitting on line boundaries keeps every frame a valid
    text fragment, so decompressing the frames in sequence must reproduce ``payload`` exactly.
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    lines = payload.splitlines(keepends=True)
    if frames == 1 or len(lines) < frames:
        return zstd.ZstdCompressor().compress(payload)
    size = len(lines) // frames
    out = bytearray()
    compressor = zstd.ZstdCompressor()
    for i in range(frames):
        chunk = lines[i * size : (i + 1) * size] if i < frames - 1 else lines[i * size :]
        out += compressor.compress(b"".join(chunk))
    return bytes(out)


def truncate_mid_frame(compressed: bytes, fraction: float = 0.5) -> bytes:
    """Cut a compressed stream at ``fraction`` of its length — ending **mid-frame**.

    This is the shape of the Phase 1 day-prefix. A correct parser must *name* the truncation
    rather than return a short archive that looks complete.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    cut = int(len(compressed) * fraction)
    if cut == 0 or cut >= len(compressed):
        raise ValueError("fraction leaves no bytes on one side of the cut")
    return compressed[:cut]


def truncate_at_frame_boundary(compressed_multi_frame: bytes, *, keep_frames: int = 1) -> bytes:
    """Keep only the first ``keep_frames`` frames of a multi-frame stream.

    The result ends **exactly on a frame boundary**, so it is a complete (if shorter) stream and
    must **not** be reported as truncated. This is the control case for
    :func:`truncate_mid_frame`: the two shapes differ only in where the cut falls, and a parser
    that cannot tell them apart would flag every short archive.
    """
    if keep_frames < 1:
        raise ValueError("keep_frames must be >= 1")
    # Deterministic frame walk: `decompressobj` stops at the end of one frame, and its
    # `unused_data` starts exactly at the next frame — so consumed = input - unused.
    pos = 0
    for _ in range(keep_frames):
        if pos >= len(compressed_multi_frame):
            raise ValueError("fewer frames than keep_frames")
        obj = zstd.ZstdDecompressor().decompressobj()
        obj.decompress(compressed_multi_frame[pos:])
        consumed = len(compressed_multi_frame) - pos - len(obj.unused_data)
        if consumed <= 0:
            raise ValueError("could not advance past a frame boundary")
        pos += consumed
    return compressed_multi_frame[:pos]


def write_realism_fixtures(out_dir: str | Path, *, games: int = 6, seed: int = 0) -> RealismPaths:
    """Write every fixture shape into ``out_dir`` and return their paths.

    Files (all deterministic for a given ``seed``):

    - ``realism-valid.pgn.zst``        — a single-frame archive (the old, too-clean shape)
    - ``realism-multiframe.pgn.zst``   — the Lichess shape: concatenated frames
    - ``realism-truncated-mid.pgn.zst``— a byte prefix, cut mid-frame
    - ``realism-truncated-boundary.pgn.zst`` — cut exactly on a frame boundary (complete)
    - ``realism-broken.pgn``           — one game carries an illegal SAN move
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = build_valid_pgn(games, seed=seed)
    single = zstd.ZstdCompressor().compress(payload)
    multi = to_multiframe_zst(payload, frames=3)
    mid = truncate_mid_frame(multi, 0.5)
    boundary = truncate_at_frame_boundary(multi, keep_frames=1)
    broken = build_broken_pgn(games, seed=seed)

    paths = RealismPaths(
        valid_multi_frame=out / "realism-multiframe.pgn.zst",
        truncated_mid_frame=out / "realism-truncated-mid.pgn.zst",
        truncated_at_boundary=out / "realism-truncated-boundary.pgn.zst",
        broken_pgn=out / "realism-broken.pgn",
        valid_single_frame=out / "realism-valid.pgn.zst",
    )
    paths.valid_single_frame.write_bytes(single)
    paths.valid_multi_frame.write_bytes(multi)
    paths.truncated_mid_frame.write_bytes(mid)
    paths.truncated_at_boundary.write_bytes(boundary)
    paths.broken_pgn.write_bytes(broken)
    return paths
