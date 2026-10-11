"""DDP / FSDP stubs — DOCUMENTED, NOT VERIFIED (GOAL.md §4.6, docs/TRAINING.md §5).

These are placeholders with documentation. They are labeled as stubs and are not claimed to
work on a cluster. Do not present them as a working recipe.
"""

from __future__ import annotations

STUB_NOTICE = "STUB — not verified on a cluster in the build phase."


def init_ddp() -> None:
    """Initialise a DDP process group. STUB."""
    raise NotImplementedError(f"init_ddp: {STUB_NOTICE}")


def wrap_fsdp(model: object) -> object:
    """Wrap a model in FSDP. STUB."""
    raise NotImplementedError(f"wrap_fsdp: {STUB_NOTICE}")
