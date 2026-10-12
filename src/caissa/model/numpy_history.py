"""Numpy history model — the scaled vertical of session-0008 (deep lane #2, ADR-0011).

The v1 vertical (`numpy_tiny`) reads **one** board and a linear (from, to) head. It proved the
composition on real data, and its ceiling was always going to be low: a chess move is a function of
where the pieces *came from*, not only where they are. This model adds exactly two organs, each
behind a flag so it can be ablated:

- **history context ``K``** — the last ``K`` boards before the current one are encoded by a SHARED
  board encoder and mixed in order (``W_mix``), so the model can see a threat forming rather than
  only its final frame. ``K=0`` collapses to the v1 information set.
- **geometry bias** — a precomputed 64x64 Chebyshev table added to the (from, to) logits with a
  single learned scale. ``geometry_bias=False`` gives an exactly-zero contribution, which is what
  makes the ablation a one-line config change rather than a code change (`caissa.model.geometry`).

The policy head is a genuine **multi-head bilinear** head: ``n_heads`` heads each produce a
(from, to) outer product from a shared ``d_model`` representation, and the logits are their sum.
The legality mask is applied *inside* the softmax, so an illegal move cannot be sampled or
reinforced (Article 1) — a mask applied after sampling would be a different, weaker guarantee.

Everything here is written out by hand on purpose: this is the training loop the metrics artifact
certifies, so its gradient is not taken on faith — ``tests/unit/test_numpy_history.py`` checks it
against a finite-difference reference on a real position.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from caissa.model.geometry import chebyshev_table
from caissa.model.numpy_tiny import FEATURE_DIM, encode
from caissa.representation.mask import legal_move_mask

_EPS = 1e-12


def _normalized_geometry() -> np.ndarray:
    """The Chebyshev table, centred and scaled so a learned weight of ~1 is meaningful.

    Centring matters: the (from, to) logits are learned freely, so an uncentred bias would just
    add a constant the head already absorbs — and the ablation's "exactly zero" would be a lie.
    """
    table = chebyshev_table()
    centred = table - table.mean()
    std = centred.std()
    return centred / (std if std > 0.0 else 1.0)


GEOMETRY_TABLE = _normalized_geometry()


def encode_history(history: list[chess.Board]) -> np.ndarray:
    """Encode ``history`` (oldest first) into a ``(len(history), FEATURE_DIM)`` matrix."""
    if not history:
        raise ValueError("history must contain at least the current board")
    return np.stack([encode(board) for board in history], axis=0)


@dataclass
class NumpyHistoryModel:
    """A tiny, CPU-only, hand-differentiated policy model with history and geometry.

    Parameters mirror the config keys one-for-one so an ablation is visible at the call site:
    ``history`` (``K``), ``d_model``, ``n_heads`` and ``geometry_bias``.
    """

    history: int = 8
    d_model: int = 256
    n_heads: int = 4
    geometry_bias: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.history < 0:
            raise ValueError("history must be >= 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")

        rng = np.random.default_rng(self.seed)
        slots = self.history + 1

        def _xavier(rows: int, cols: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (rows + cols))
            return rng.uniform(-limit, limit, size=(rows, cols))

        self.w_enc = _xavier(FEATURE_DIM, self.d_model)
        self.b_enc = np.zeros(self.d_model)
        self.w_mix = _xavier(slots * self.d_model, self.d_model)
        self.b_mix = np.zeros(self.d_model)
        # Bilinear from/to heads: logits = sum_h outer(h @ F[h], h @ T[h]).
        self.head_from = _xavier(self.n_heads * self.d_model, 64).reshape(
            self.n_heads, self.d_model, 64
        )
        self.head_to = _xavier(self.n_heads * self.d_model, 64).reshape(
            self.n_heads, self.d_model, 64
        )
        # A single learned scale for the geometric prior. Unused (and exactly zero) when the
        # ablation flag is off, so "with bias" and "without bias" differ in exactly one thing.
        self.geom_scale = np.array(1.0)

    # --- forward -----------------------------------------------------------------

    def _check_input(self, x_hist: np.ndarray) -> None:
        """Reject a malformed history loudly rather than broadcasting into a wrong answer."""
        if x_hist.ndim != 2 or x_hist.shape[1] != FEATURE_DIM:
            raise ValueError(f"x_hist must be (slots, {FEATURE_DIM}), got {x_hist.shape}")
        if x_hist.shape[0] != self.history + 1:
            raise ValueError(
                f"x_hist has {x_hist.shape[0]} slots but the model expects {self.history + 1}"
            )

    def _forward(self, x_hist: np.ndarray):
        """The ONE forward pass both the objective and the gradient use. Geometry included.

        A single forward is deliberate, not a style choice. The session-0008 gradient test found
        the defect this prevents: `train_step` used to inline its own forward that **omitted** the
        geometric bias while `logits()` added it, so the model trained one objective and was
        evaluated on another. Different code paths for "what the loss is" and "what the gradient
        is" is how a subtle, silent wrongness gets in; one path is how it stays out.
        """
        self._check_input(x_hist)
        E = np.tanh(x_hist @ self.w_enc + self.b_enc)  # (slots, d_model)
        c = E.reshape(-1)
        h = np.tanh(c @ self.w_mix + self.b_mix)  # (d_model,)
        a = np.einsum("d,hdf->hf", h, self.head_from)  # (n_heads, 64)
        b = np.einsum("d,hdt->ht", h, self.head_to)  # (n_heads, 64)
        logits = np.einsum("hf,ht->ft", a, b)  # (64, 64)
        if self.geometry_bias:
            logits = logits + self.geom_scale * GEOMETRY_TABLE
        return E, c, h, a, b, logits

    def logits(self, x_hist: np.ndarray) -> np.ndarray:
        """The (from, to) logits with the geometric prior folded in when the flag is on."""
        return self._forward(x_hist)[-1]

    def masked_probs(self, board: chess.Board, x_hist: np.ndarray) -> np.ndarray:
        """Softmax over legal (from, to) pairs only — the Article 1 guarantee."""
        logits = self.logits(x_hist)
        mask = legal_move_mask(board)
        z = np.where(mask, logits, -np.inf)
        z = z - np.max(z)
        e = np.where(mask, np.exp(z), 0.0)
        total = e.sum()
        return e / total if total > 0 else e

    def loss(self, board: chess.Board, x_hist: np.ndarray, target: chess.Move) -> float:
        """Masked cross-entropy for one position, without touching the weights."""
        probs = self.masked_probs(board, x_hist)
        return float(-np.log(max(probs[target.from_square, target.to_square], _EPS)))

    def argmax_move(self, board: chess.Board, x_hist: np.ndarray) -> chess.Move:
        """The policy's top-1 legal move — deterministic, and what the model actually plays.

        The Article 1 gate uses this rather than ``sample_move``: a multinomial draw from a peaked
        softmax returns the argmax almost always, so a "100% legal" claim from sampling would be
        true but weak. Top-1 is the exact move the policy would play out of distribution.
        """
        probs = self.masked_probs(board, x_hist)
        idx = int(np.argmax(probs))
        frm, to = divmod(idx, 64)
        move = chess.Move(frm, to)
        if move not in board.legal_moves:  # promotion: the index carries no promotion piece
            move = chess.Move(frm, to, promotion=chess.QUEEN)
        return move

    def sample_move(
        self, board: chess.Board, x_hist: np.ndarray, rng: np.random.Generator
    ) -> chess.Move:
        """Sample a move from the masked policy. Legal by construction (Article 1)."""
        probs = self.masked_probs(board, x_hist)
        flat = probs.ravel()
        total = flat.sum()
        if total <= 0:
            raise RuntimeError("masked policy has no mass — refusing to sample")
        idx = int(rng.choice(flat.size, p=flat / total))
        frm, to = divmod(idx, 64)
        move = chess.Move(frm, to)
        if move not in board.legal_moves:  # promotion: the index carries no promotion piece
            move = chess.Move(frm, to, promotion=chess.QUEEN)
        return move

    # --- backward ----------------------------------------------------------------

    def gradients(
        self, board: chess.Board, x_hist: np.ndarray, target: chess.Move
    ) -> tuple[float, dict[str, np.ndarray]]:
        """The loss and the gradient of every learned tensor, WITHOUT touching the weights.

        Split out of `train_step` so an optimizer other than plain SGD can consume the same
        gradient object. The arithmetic is byte-for-byte the one `train_step` always used -- this
        is a refactor, not a new path -- so the finite-difference test that has guarded this
        backprop since session 0008 keeps guarding it.
        """
        # The SAME forward the objective uses (geometry included) — see `_forward`'s docstring.
        E, c, h, a, b, logits = self._forward(x_hist)

        mask = legal_move_mask(board)
        z = np.where(mask, logits, -np.inf)
        z = z - np.max(z)
        e = np.where(mask, np.exp(z), 0.0)
        probs = e / e.sum()

        loss = -np.log(max(probs[target.from_square, target.to_square], _EPS))

        dlogits = probs.copy()
        dlogits[target.from_square, target.to_square] -= 1.0
        dlogits[~mask] = 0.0

        # --- policy head ---
        da = np.einsum("ft,ht->hf", dlogits, b)  # (n_heads, 64)
        db = np.einsum("ft,hf->ht", dlogits, a)  # (n_heads, 64)
        d_head_from = np.einsum("hf,d->hdf", da, h)
        d_head_to = np.einsum("ht,d->hdt", db, h)
        dh = np.einsum("hf,hdf->d", da, self.head_from) + np.einsum("ht,hdt->d", db, self.head_to)

        grads: dict[str, np.ndarray] = {}
        if self.geometry_bias:
            grads["geom_scale"] = np.array(float(np.sum(dlogits * GEOMETRY_TABLE)))

        # --- mixer ---
        draw_h = dh * (1.0 - h * h)
        grads["w_mix"] = np.outer(c, draw_h)
        grads["b_mix"] = draw_h
        dc = draw_h @ self.w_mix.T

        # --- encoder ---
        dE = dc.reshape(E.shape) * (1.0 - E * E)
        grads["w_enc"] = x_hist.T @ dE
        grads["b_enc"] = dE.sum(axis=0)
        grads["head_from"] = d_head_from
        grads["head_to"] = d_head_to
        return loss, grads

    def train_step(
        self, board: chess.Board, x_hist: np.ndarray, target: chess.Move, lr: float
    ) -> float:
        """One SGD step. Returns the loss BEFORE the update (so a curve is comparable).

        Kept as the plain-SGD entry point (``lr=0.0`` is a pure measurement) so the v1 family's
        historical behaviour, and every test that pins it, are unchanged by the Adam refactor.
        """
        if lr == 0.0:
            return self.loss(board, x_hist, target)

        loss, grads = self.gradients(board, x_hist, target)
        if self.geometry_bias:
            self.geom_scale -= lr * float(grads["geom_scale"])
        self.head_from -= lr * grads["head_from"]
        self.head_to -= lr * grads["head_to"]
        self.w_mix -= lr * grads["w_mix"]
        self.b_mix -= lr * grads["b_mix"]
        self.w_enc -= lr * grads["w_enc"]
        self.b_enc -= lr * grads["b_enc"]
        return loss

    def adam_step(
        self, board: chess.Board, x_hist: np.ndarray, target: chess.Move, adam, lr: float
    ) -> tuple[float, float]:
        """One Adam step. Returns ``(loss_before_update, grad_norm)``.

        ``grad_norm`` is returned rather than logged internally: the caller decides how often a
        norm is printed, and a training loop holding the number can put it in the artifact.
        """
        from caissa.training.adam import grad_norm

        loss, grads = self.gradients(board, x_hist, target)
        norm = grad_norm(grads)
        adam.step(self, grads, lr)
        return loss, norm
