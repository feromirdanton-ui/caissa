# session-0007 follow-up — the first clean-runner fetch of the pinned Stockfish failed

**Status:** fixed, verified locally AND on the runner. Commit `d9f59ee6` (branch `main`).
**Trigger:** `smoke-train` run [38176585929](https://github.com/feromirdanton-ui/caissa/actions/runs/38176585929),
job «real Stockfish blunder labeling», step «Fetch Stockfish (pinned, sha256 verified)».

## What happened

The first dispatch of `smoke-train.yml` split cleanly in two:

- **`vertical (data → shards → train → metrics)` — SUCCESS.** The deep-lane composition ran on
  the runner: real Lichess prefix → 6 shards → 30 000 train / 1 664 val positions → the tiny
  model → metrics artifact, verdict **ALIVE** (val loss 3.2603 → 3.1326; top-1 0.1839 vs baseline
  0.0132; legality 1000/1000). This is the run the pre-registered criterion was written for, and
  it passed.
- **`real Stockfish blunder labeling` — FAILED** at the fetch step:

  ```
  shutil.Error: Destination path
  'data/tools/stockfish/stockfish-linux-x86-64-universal' already exists
  ```

## Root cause — an artefact, not a logic slip

`acquisition/stockfish.py` extracts the single archive member
`stockfish/stockfish-linux-x86-64-universal` with `tar.extract(member, path=dest)`. Tar
recreates the member's own top-level folder, so that call produces the **directory**
`dest/stockfish/`. The module then wants `dest/stockfish` to be the **binary file** of the same
name — and `shutil.move(file, existing_directory_of_the_same_name)` raises, every time.

**Why 115 green local tests never saw it.** Stockfish was already on `PATH` in the local loop, so
`default_engine()` never fell through to the fetch branch; the extract+move path had never
executed *anywhere* before the runner. This is precisely the defect class ADR-0003 says the runner
exists to catch — and it did.

## The fix

Extract into a `tempfile.TemporaryDirectory()` and move the binary into place from there, clearing
a stale `dest/stockfish/` directory first so a partial run cannot silently nest the binary. The
`tar.extract` destination is no longer `dest` itself, so the archive's own folder can never
collide with the binary path.

## Evidence (reproduce, don't hope)

1. **Reproduced the exact failure** locally against the real pinned tarball
   (`tools-cache/stockfish.tar.gz`, sha256 `9defc0d4…`) with the pre-fix code — identical
   `shutil.Error`.
2. **Fixed path, end-to-end:** fresh fetch lands `dest/stockfish` as a **file**; a second run is
   **idempotent**; `stockfish --version` reports *Stockfish 19* with binary sha
   `0f83d24cc46d2c66…` — the same measured value committed in session 0007.
3. **Regression test** — `tests/unit/test_acquisition_stockfish.py` (3 tests, offline,
   deterministic one-member tarball with the pinned sha monkeypatched): the fresh fetch, the
   idempotent re-run, and the stale-directory recovery. The third fails against the pre-fix code.
4. **Local gate:** `make lint test` → **118 passed** (was 115); `mypy` success, 43 source files.
5. **Runner gate:** CI run [38177092062](https://github.com/feromirdanton-ui/caissa/actions/runs/38177092062)
   — **success** on `d9f59ee6`. Re-dispatched `smoke-train` run
   [38177107043](https://github.com/feromirdanton-ui/caissa/actions/runs/38177107043): the fetch
   step that used to die is **green** and the labeling job now proceeds into real-engine labeling.

## Open note (not part of this fix)

The real-Stockfish labeling job is long-running by nature (engine search over the day prefix).
Its wall-clock is a performance question separate from this defect; the defect it blocked is
closed.
