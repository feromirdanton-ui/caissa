# Caïssa-1

> Strength **and** voice. Memory **and** the ability to teach.

Caïssa-1 is a chess being that combines four organs — **Intuition** (System 1 policy/value),
**Word** (System 2 verbal reasoning), **Memory** (motive retrieval), and the **Queens Guard**
(move-hygiene module) — on a single value axis that is *not* Elo but **understanding**.

This repository is the **build phase** of the project. It prepares everything an arriving GPU
needs to start training with one command. It does **not** train large models, and it never
pretends to.

**The constitution is [`GOAL.md`](GOAL.md).** Read it before touching anything. It is not changed
without an ADR.

## Definition of Awesome

```bash
git clone <repo> && cd caissa
pip install -e ".[torch]"             # on the GPU box, this is what turns Act I on
make train CONFIG=configs/nano.yaml   # Act I starts (and refuses to fake a run without torch)
```

## Quick start (CPU, no GPU needed)

```bash
make install        # editable install + dev deps
make lint test      # Phase 0 gate — must be green
make smoke          # tiny model, CPU, <=10 min, 1000 sampled moves — all legal
```

## Layout

See [`docs/MANIFEST.md`](docs/MANIFEST.md) for the concept manifest and
[`DECISIONS/`](DECISIONS/) for the ADR log. The repo skeleton is fixed by
[ADR-0002](DECISIONS/adr-0002-repo-layout.md).

## Honesty rules (non-negotiable)

- A **mock** is labeled as a mock. A simulation is never passed off as a result.
- **No data or weights in git** — only manifests (source, sha256, volume, schema, rows).
- **No fabricated training results.** The build phase prepares; it does not train.
- CI is always green. Red CI is a stop signal for everything else.

## Status

Phase 0 — Foundation. **Not yet accepted**: session-0002 reproduced the delivered artifact in a clean room and found P0 defects ([docs/ISSUES-0002.md](docs/ISSUES-0002.md)); Phase 1 is gated on closing them. Raw evidence: [LOG/session-0002.md](LOG/session-0002.md).
