"""Tiny decoder-only transformer with a masked policy head (GOAL.md §4.2).

This is the CPU-smoke model (<10M params). It is a REAL model, not a mock: it trains,
its loss decreases, and its policy head applies the legality mask inside the softmax so
that sampled moves are 100% legal by construction (Article 1).

Requires torch. When torch is unavailable, `caissa.model.numpy_tiny` provides a smaller
numpy fallback used by `make smoke` so CI stays green without a GPU stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is an optional extra; the build phase must not require it
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _HAS_TORCH = False


@dataclass
class TinyConfig:
    vocab_size: int = 64 + 32  # square tokens + meta tokens (padded)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 96
    dropout: float = 0.0


if _HAS_TORCH:

    class TinyModel(nn.Module):
        """Decoder-only transformer with a (from, to) policy head."""

        def __init__(self, cfg: TinyConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.from_head = nn.Linear(cfg.d_model, 64)
            self.to_head = nn.Linear(cfg.d_model, 64)
            self.value_head = nn.Linear(cfg.d_model, 3)  # W / D / L

        def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            b, s = tokens.shape
            pos = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, s)
            x = self.tok(tokens) + self.pos(pos)
            mask = torch.triu(torch.ones(s, s, device=tokens.device), diagonal=1).bool()
            h = self.encoder(x, mask=mask)
            pooled = h[:, -1, :]
            return self.from_head(pooled), self.to_head(pooled)

        def policy_logits(self, tokens: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
            """Return masked logits over (from, to) pairs.

            legal_mask: bool [B, 64, 64]. Masked entries are set to -inf BEFORE the softmax,
            so an illegal move has probability exactly 0 (Article 1).
            """
            from_logits, to_logits = self.forward(tokens)
            pair = from_logits.unsqueeze(2) + to_logits.unsqueeze(1)  # [B, 64, 64]
            return pair.masked_fill(~legal_mask, float("-inf"))

    def build_tiny(cfg: TinyConfig | None = None) -> TinyModel:
        return TinyModel(cfg or TinyConfig())

else:  # pragma: no cover

    class TinyModel:  # type: ignore[no-redef]
        def __init__(self, cfg: TinyConfig) -> None:
            raise ImportError(
                "torch is not installed. Install with `pip install -e '.[torch]'` "
                "or use caissa.model.numpy_tiny for the CPU fallback."
            )

    def build_tiny(cfg: TinyConfig | None = None) -> TinyModel:
        return TinyModel(cfg or TinyConfig())


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numpy reference of the masked softmax — the operation that forbids illegal moves."""
    x = np.array(logits, dtype=float, copy=True)
    x[~mask] = -np.inf
    x = x - np.nanmax(np.where(mask, x, -np.inf))
    e = np.where(mask, np.exp(x), 0.0)
    total = e.sum()
    return e / total if total > 0 else e
