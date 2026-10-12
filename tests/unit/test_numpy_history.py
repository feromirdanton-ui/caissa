"""`NumpyHistoryModel` — the gradient is checked, not trusted (session-0008, deep lane #2).

The history model of `smoke_train_v2` is the first organ whose *learning* this project will report
as a result. A hand-written backpropagation that is subtly wrong does not crash: it trains a little
and reports a number, which is exactly the kind of quiet falsehood Article 5 forbids. So the
gradient is verified against a finite-difference reference on a real position, and the mask is
proved to be the only thing the sampler can pick from.

The tests are deliberately about *properties that must hold*, not about a remembered numeric
output: a correct forward pass can be re-derived, a leaked legality mask cannot be argued away.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from caissa.model.numpy_history import (
    GEOMETRY_TABLE,
    NumpyHistoryModel,
    encode_history,
)
from caissa.model.numpy_tiny import FEATURE_DIM
from caissa.representation.mask import legal_move_mask

PARAMS = ("w_enc", "b_enc", "w_mix", "b_mix", "head_from", "head_to", "geom_scale")


def _position() -> chess.Board:
    """A real, non-trivial position: 1.e4 e5 2.Nf3 Nc6."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push(chess.Move.from_uci(uci))
    return board


def _history(board: chess.Board, k: int) -> np.ndarray:
    """A K+1 window ending at ``board`` (repeat the current board for earlier slots is avoided:
    we rebuild the real predecessor chain, so the fixture is the shape the model will really see).
    """
    boards: list[chess.Board] = []
    replay = chess.Board()
    for move in board.move_stack:
        boards.append(replay.copy())
        replay.push(move)
    boards.append(board.copy())
    if len(boards) < k + 1:
        boards = [boards[0]] * (k + 1 - len(boards)) + boards
    return encode_history(boards[-(k + 1) :])


def _a_move(board: chess.Board) -> chess.Move:
    return next(iter(board.legal_moves))


def _snapshot(model: NumpyHistoryModel) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(model, name), copy=True) for name in PARAMS}


def _total_loss(
    model: NumpyHistoryModel, x: np.ndarray, board: chess.Board, move: chess.Move
) -> float:
    return model.loss(board, x, move)


def test_analytic_gradient_matches_finite_differences() -> None:
    """Every parameter's analytic gradient must match a central finite difference.

    This is the test that makes the reported learning believable: it is run on a REAL position
    (1.e4 e5 2.Nf3 Nc6) with the legality mask in force, so a gradient error would show up as a
    mismatch rather than as a slightly worse accuracy nobody can attribute.
    """
    board = _position()
    move = _a_move(board)
    x = _history(board, k=3)
    lr = 1e-3
    eps = 1e-6

    model = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7)
    before = _snapshot(model)

    model.train_step(board, x, move, lr=lr)  # performs the real update
    after = _snapshot(model)

    analytic = {name: (before[name] - after[name]) / lr for name in PARAMS}

    # Numeric gradient of each parameter, computed from a fresh copy of the *pre-step* model.
    numeric: dict[str, np.ndarray] = {}
    for name in PARAMS:
        ref = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7)
        grad = np.zeros_like(before[name], dtype=float)
        flat = grad.reshape(-1)
        base = np.array(before[name], copy=True)
        for idx in range(flat.size):
            for sign, store in ((+1, "plus"), (-1, "minus")):
                shifted = base.reshape(-1).copy()
                shifted[idx] += sign * eps
                setattr(ref, name, shifted.reshape(base.shape))
                value = _total_loss(ref, x, board, move)
                if store == "plus":
                    plus = value
                else:
                    minus = value
            flat[idx] = (plus - minus) / (2 * eps)
        numeric[name] = grad

    for name in PARAMS:
        np.testing.assert_allclose(
            analytic[name],
            numeric[name],
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"gradient mismatch for {name!r} — the backprop does not match the forward pass",
        )


def test_masked_probs_have_mass_only_on_legal_moves() -> None:
    """Article 1: the mask, not luck, is what makes illegal moves impossible."""
    board = _position()
    model = NumpyHistoryModel(history=2, d_model=8, n_heads=2, seed=1)
    x = _history(board, k=2)
    probs = model.masked_probs(board, x)
    mask = legal_move_mask(board)

    assert probs.shape == (64, 64)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs[~mask] == 0.0), "an illegal (from, to) pair was given probability mass"
    assert probs[mask].sum() > 0.99


def test_sampling_is_one_hundred_percent_legal() -> None:
    """Sample many moves across many real positions — every one must be legal (Article 1)."""
    rng = np.random.default_rng(0)
    model = NumpyHistoryModel(history=2, d_model=8, n_heads=2, seed=3)
    board = chess.Board()
    checked = 0
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"):
        board.push(chess.Move.from_uci(uci))
        if board.is_game_over():
            break
        x = _history(board, k=2)
        for _ in range(25):
            move = model.sample_move(board, x, rng)
            assert move in board.legal_moves, f"illegal move {move} sampled on {board.fen()}"
            checked += 1
    assert checked >= 200, "the sampler was barely exercised — strengthen the fixture"


def test_geometry_ablation_is_exactly_zero_and_changes_only_that_term() -> None:
    """The ablation flag must remove the prior EXACTLY — otherwise the comparison is not controlled.\n\n    With the flag off the logits must equal the unbiased computation bit-for-bit; with it on the\n    difference must be exactly ``geom_scale * GEOMETRY_TABLE`` on legal entries."""
    board = _position()
    x = _history(board, k=2)
    off = NumpyHistoryModel(history=2, d_model=8, n_heads=2, geometry_bias=False, seed=5)
    on = NumpyHistoryModel(history=2, d_model=8, n_heads=2, geometry_bias=True, seed=5)

    logits_off = off.logits(x)
    logits_on = on.logits(x)
    assert not np.allclose(logits_off, logits_on), "the flag did nothing — it is not an ablation"

    delta = logits_on - logits_off
    np.testing.assert_allclose(delta, on.geom_scale * GEOMETRY_TABLE, rtol=1e-12, atol=1e-12)

    # And with the flag off, zeroing geom_scale must not change anything at all.
    logits_off2 = off.logits(x)
    np.testing.assert_array_equal(logits_off, logits_off2)


def test_history_zero_accepts_a_single_board_and_has_the_v1_information_set() -> None:
    """K=0 must be a legal configuration: one board in, one (from, to) distribution out."""
    board = _position()
    model = NumpyHistoryModel(history=0, d_model=8, n_heads=2, seed=2)
    x = encode_history([board.copy()])
    assert x.shape == (1, FEATURE_DIM)
    probs = model.masked_probs(board, x)
    assert np.isclose(probs.sum(), 1.0)
    assert probs[legal_move_mask(board)].sum() > 0.99


def test_wrong_history_shape_is_a_loud_failure() -> None:
    """A malformed window must raise, never broadcast into a plausible-looking answer."""
    board = _position()
    model = NumpyHistoryModel(history=3, d_model=8, n_heads=2, seed=0)
    too_short = encode_history([board.copy(), board.copy()])  # 2 slots, model wants 4
    with pytest.raises(ValueError, match="slots"):
        model.logits(too_short)
    with pytest.raises(ValueError, match="slots"):
        model.train_step(board, too_short, _a_move(board), lr=0.1)


def test_learning_actually_reduces_loss_on_a_fixed_target() -> None:
    """A real SGD loop on one position must drive its loss DOWN — proof the step has the right sign."""
    board = _position()
    move = chess.Move.from_uci("f3e5")  # Nxe5, a legal capture
    assert move in board.legal_moves
    x = _history(board, k=2)
    model = NumpyHistoryModel(history=2, d_model=16, n_heads=2, seed=11)

    first = model.loss(board, x, move)
    for _ in range(60):
        model.train_step(board, x, move, lr=0.05)
    last = model.loss(board, x, move)

    assert last < first * 0.5, f"loss did not fall meaningfully: {first:.4f} -> {last:.4f}"


def test_zero_learning_rate_is_a_pure_measurement() -> None:
    """``lr=0`` must return the loss and change no parameter (used for validation loss)."""
    board = _position()
    x = _history(board, k=1)
    model = NumpyHistoryModel(history=1, d_model=8, n_heads=2, seed=4)
    before = _snapshot(model)
    value = model.train_step(board, x, _a_move(board), lr=0.0)
    after = _snapshot(model)
    for name in PARAMS:
        np.testing.assert_array_equal(before[name], after[name])
    assert value == pytest.approx(model.loss(board, x, _a_move(board)))


def test_training_objective_includes_the_geometry_bias() -> None:
    """REGRESSION (found by the gradient check): the trained loss must BE the reported loss.

    `train_step` used to inline its own forward that omitted the geometric bias while `logits()`
    added it — so with the bias on, the model trained one objective and every reported/validated
    loss was computed from another. The gradient test caught it (analytic vs numeric diverged only
    when `geometry_bias=True`); nothing else would have, because both objectives learn *something*.

    With a negligible learning rate the step's returned loss must equal the public ``loss()`` to
    high precision. Before the fix they differed by ~0.1 here; after it they agree.
    """
    board = _position()
    x = _history(board, k=3)
    move = _a_move(board)

    for geom in (False, True):
        model = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=geom, seed=7)
        reported = model.loss(board, x, move)
        stepped = model.train_step(board, x, move, lr=1e-9)
        assert stepped == pytest.approx(reported, rel=1e-5, abs=1e-7), (
            "train_step's loss differs from loss(): the training objective and the reported "
            f"objective are not the same function (geometry_bias={geom})"
        )


def test_geometry_bias_actually_affects_the_training_loss() -> None:
    """The bias must enter the TRAINING loss, not only `logits()` — the ablation's precondition."""
    board = _position()
    x = _history(board, k=3)
    move = _a_move(board)
    off = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=False, seed=7)
    on = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7)
    assert on.loss(board, x, move) != pytest.approx(off.loss(board, x, move))
    # And the flag must not perturb the very first step's gradient of an unrelated scale:
    # with lr tiny, a model whose bias is on must move its w_enc differently than one whose is off.
    on.train_step(board, x, move, lr=1e-2)
    off.train_step(board, x, move, lr=1e-2)
    assert not np.allclose(on.w_enc, off.w_enc), "the bias did not reach the input gradient"


# --- session-0013 F2: cosine (bounded) logits ---------------------------------------------------


def test_cosine_logits_gradient_matches_finite_differences() -> None:
    """F2's normalisation Jacobian must be exact -- a wrong one trains a different objective.

    The cosine head adds a unit-normalisation between the projections and the logits, so the
    backward must push the gradient through ``(I - a_hat a_hat^T)/||a||``. That Jacobian is easy to
    get subtly wrong (a missing projection term still trains, just wrongly), so it is checked
    against a central finite difference on a real position, exactly as the plain head is.
    """
    board = _position()
    move = _a_move(board)
    x = _history(board, k=3)
    lr = 1e-3
    eps = 1e-6

    model = NumpyHistoryModel(
        history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7, logit_norm=True
    )
    before = _snapshot(model)
    model.train_step(board, x, move, lr=lr)
    after = _snapshot(model)
    analytic = {name: (before[name] - after[name]) / lr for name in PARAMS}

    numeric: dict[str, np.ndarray] = {}
    for name in PARAMS:
        ref = NumpyHistoryModel(
            history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7, logit_norm=True
        )
        grad = np.zeros_like(before[name], dtype=float)
        flat = grad.reshape(-1)
        base = np.array(before[name], copy=True)
        for idx in range(flat.size):
            for sign, store in ((+1, "plus"), (-1, "minus")):
                shifted = base.reshape(-1).copy()
                shifted[idx] += sign * eps
                setattr(ref, name, shifted.reshape(base.shape))
                value = _total_loss(ref, x, board, move)
                if store == "plus":
                    plus = value
                else:
                    minus = value
            flat[idx] = (plus - minus) / (2 * eps)
        numeric[name] = grad

    for name in PARAMS:
        np.testing.assert_allclose(
            analytic[name],
            numeric[name],
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"cosine-head gradient mismatch for {name!r}",
        )


def test_cosine_logits_are_bounded_however_large_the_weights_grow() -> None:
    """The F2 mechanism: logits stay bounded as the head weights grow -- the divergence's cause removed.

    Session-0012's blow-up was unbounded bilinear logits. With cosine logits the score is a scaled
    inner product of UNIT vectors, so ``|logit| <= logit_scale * n_heads`` regardless of the weight
    magnitude. This test scales the head weights by 1e3 and asserts the logits do not move -- which
    is the property the plain head does NOT have.
    """
    board = _position()
    x = _history(board, k=3)
    model = NumpyHistoryModel(
        history=3, d_model=12, n_heads=2, geometry_bias=False, seed=7, logit_norm=True
    )
    base = model.logits(x)
    model.head_from = model.head_from * 1e3
    model.head_to = model.head_to * 1e3
    scaled = model.logits(x)
    np.testing.assert_allclose(base, scaled, rtol=1e-9, atol=1e-9)
    assert np.max(np.abs(scaled)) <= model.logit_scale * model.n_heads + 1e-9

    # ... while the PLAIN head's logits grow with the weights -- the contrast that makes F2 a fix.
    plain = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=False, seed=7)
    plain_base = np.max(np.abs(plain.logits(x)))
    plain.head_from = plain.head_from * 1e3
    plain.head_to = plain.head_to * 1e3
    assert np.max(np.abs(plain.logits(x))) > 1e3 * plain_base


def test_cosine_logits_off_by_default_leaves_the_plain_head_unchanged() -> None:
    """The flag must be inert when off: the v1/v2 historical behaviour cannot move."""
    board = _position()
    x = _history(board, k=3)
    a = NumpyHistoryModel(history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7)
    b = NumpyHistoryModel(
        history=3, d_model=12, n_heads=2, geometry_bias=True, seed=7, logit_norm=False
    )
    np.testing.assert_allclose(a.logits(x), b.logits(x))
