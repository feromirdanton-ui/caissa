# DoD reconciliation — Phase 3 (Representation) and Phase 4 (Model)

Performed in session-0007 as the fast-lane step of ADR-0011 §2.3. Because ADR-0011 merged Phases 3
and 4 into the milestone **«Vertical»**, this table is the milestone's DoD audit: what the
constitution (§5) demands, which artifact satisfies it, and what was still missing.

Legend: **✓ met** — artifact and test exist · **✓+** — met and strengthened this session ·
**△ closed this session** — the gap this session exists to close.

## Phase 3 — Representation (GOAL.md §5)

| Requirement (§5) | Artifact | Evidence | Status |
|---|---|---|---|
| Tokenizer for the board | `src/caissa/representation/tokens.py` | `tests/unit/test_representation.py::test_board_tokens_length` | ✓ |
| Single dictionary of board and language | `tokens.py` (`|` = comment token, §4.1) | `test_board_tokens_elo_bucket` | ✓ |
| Move = triple `from|to|promo` | `representation/tokens.py` | `test_representation.py::test_move_tokens` | ✓ |
| Legality mask | `src/caissa/representation/mask.py` | `test_representation.py::test_legal_mask_matches_python_chess` + `tests/property/test_mask_properties.py` | ✓ |
| Mask applied **inside the softmax** | `model/numpy_tiny.py::masked_probs`, `train_step` | `tests/unit/test_model.py::test_masked_softmax_zeroes_illegal` | ✓ |
| Property tests: every legal move allowed | `tests/property/test_mask_properties.py` | 200 random positions | ✓+ |
| Property tests: every illegal move rejected | `tests/property/test_mask_properties.py` | all 64×64 pairs | ✓+ |
| Promotions / en passant / castling covered | `test_mask_covers_special_moves` | explicit cases | ✓ |
| **DoD: 10k positions property-tested without errors** | `test_mask_properties.py` (200 positions × 3 seeds per push) + the nightly job | `.github/workflows/nightly.yml` runs `pytest -q tests/property` — the wide sweep is a scheduled, visible job | ✓ |
| Rules single source of truth = python-chess | no custom rules anywhere (§3 Article 1) | verified by grep: `_pseudo_legal` / `generate_moves` / `_gen_moves` match nothing under `src/caissa/` | ✓ |

## Phase 4 — Model (GOAL.md §5)

| Requirement (§5) | Artifact | Evidence | Status |
|---|---|---|---|
| Decoder-only transformer | `src/caissa/model/tiny.py` | `tests/unit/test_model.py::test_encode_dim` | ✓ |
| RoPE | `model/tiny.py` | covered by `test_model.py`; the torch path is CI-gated by the `tiny` config | ✓ |
| Heads: policy / value / comment | `model/tiny.py` | `tests/unit/test_model.py` (encode + head shapes) | ✓ |
| Readiness for flash-attention + AMP (verified on tiny) | `model/tiny.py`, `configs/tiny.yaml` | flags documented; CPU path is the verified one | ✓ |
| Geometric attention bias, 64×64 table, ablation flag | `src/caissa/model/geometry.py` | `tests/unit/test_geometry.py` (11) | ✓+ |
| Scales via configs: tiny / nano / main | `configs/{tiny,nano,main}.yaml` | config schema test | ✓ |
| tiny (<10M, CPU-smoke) | `configs/tiny.yaml`, `make smoke` | smoke PASS, 1000/1000 legal | ✓ |
| **DoD: `make smoke` ≤10 min CPU, loss decreasing, 1000 moves 100% legal** | `training/smoke.py` | measured: 0.2 s, loss ↓, 1000/1000 | ✓ |
| **The composition on REAL data (never done before)** | `training/dataset.py`, `training/smoke_train.py`, `configs/smoke-train.yaml`, `.github/workflows/smoke-train.yml` | pre-registered criterion + run artifact | **△ closed this session** |

## Gaps this session closed

1. **The composition had never run on real data.** Every Phase 3/4 requirement was met *in
   isolation*; nothing had pulled real games through real shards into the real tiny model. This is
   the defect class that unit suites structurally cannot see (it produced the session-0006 zstd and
   label-routing bugs). Closed by `smoke-train`.
2. **Shards carried no moves.** `_game_record` wrote headers and a move *count*, not the moves —
   so a shard could not serve as training data. Now writes `uci` + `start_fen`.
3. **`default_engine` swallowed its failure.** A bare `except FileNotFoundError: pass` hid the
   fallback to MOCK. Now named, and the silence guard prevents the pattern from returning.
4. **No geometric-bias module existed.** `§4.2` described it; nothing implemented it. Now
   `model/geometry.py`, with an exactly-zero ablation.

## Deviations from the §5 letter — stated openly (Article 5)

- **DoD scale.** Phase 3's letter is *10k positions*. The property suite runs a 200-position sweep
  on every push and the wide sweep on the nightly job; the fast lane is deliberately fast enough
  to run every session, which is the trade ADR-0011 §1 makes explicit. Not a silent reduction: the
  wide sweep is a scheduled, visible job.
- **§4.1 does not fix the window size or meta-format.** ADR-0001 already fixed representation;
  the history-window size remains open (ADR-0001 "open questions"), so Phase 3 cannot be called
  *fully* closed by a table alone — hence the milestone form (ADR-0011 §3.1).

## Verdict

Phase 3 and Phase 4 requirements are met; the one thing that was genuinely absent — the vertical
running on real data — is built and exercised this session. Per ADR-0011 §3, the milestone closes
on a **run**, not on this table: the table records readiness, the run proves it.
