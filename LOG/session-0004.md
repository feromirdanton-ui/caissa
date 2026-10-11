# LOG — session-0004 «Certify before you record»

- **Date:** 2026-10-11
- **Phase:** 1 — Acquisition (first work session after the Phase 0 acceptance request)
- **Issue worked:** GitHub **#3** — `lichess_bulk loader: resumable + sha256 + manifest` (P0, phase-1;
  listed as task 4 in GOAL.md §11)
- **Branch:** `feat/issue-3-certify-downloads`
- **Status:** fix implemented, locally green, pushed for CI, merged to `main`.

---

## What

One session, one issue (Article 9). The loader was hardened so that **a manifest is written only
after the downloaded bytes have been certified against a trusted reference** — the constitution's
Article 14 («the manifest is a certification of truth, not a journal»).

The change is in two files:

- `src/caissa/acquisition/lichess_bulk.py`
- `tests/unit/test_lichess_bulk.py`

## Why — a defect, proven before it was fixed

The audit that closed session-0003 (P0-2) taught the loader to reject a `200`-instead-of-`206`
range response and to verify the sha256 of the written file. It did **not** cover the case where
the server *cannot be measured at all*.

`_remote_size()` reads the length from a `HEAD` request. When that request fails — a mirror that
does not answer `HEAD` (a `500`), a proxy that strips it — `expected` becomes `None`, and both
size checks in `download()` are skipped: the `written != expected` guard is gated on
`expected is not None`, and the same `None` propagates into `fetch_month()`, which then writes a
manifest regardless. Nothing in the `GET` response contradicts a truncated body, so the loader
published it and **recorded a manifest certifying a falsehood**.

The defect was reproduced before any fix, against the module's own local HTTP fixture (a server
whose `HEAD` returns `500` and whose `GET` sends half the resource):

```
real resource size : 307200 bytes   sha256=a7d8881521cbb1e4...
published file size: 153600 bytes
manifest bytes     : 153600
manifest sha256    : 41dae6b28c30ab57...
RESULT: the loader PUBLISHED a truncated file and WROTE A MANIFEST for it.
```

That is exactly the failure class Article 14 was written for: a record that certifies something
that is not true. It is not a crash — it is worse, a silent false credential.

## Change

- `download(..., expected_bytes=None)`: the trusted length is resolved **before** the transfer.
  `expected_bytes` (out of band — a published volume, an index file, a prior manifest) is
  authoritative; failing that, the server's `Content-Length`. If a reference exists and the
  written bytes do not match it, `UnverifiableDownload` is raised — the `.part` is kept for the
  next run, but nothing is published and nothing is recorded.
- If **no** reference exists, the body is unverifiable and the call raises `UnverifiableDownload`
  with an actionable message. The module now refuses to guess rather than certify.
- If the out-of-band reference and the server's `Content-Length` **disagree**, neither is trusted
  and the download is refused — no silent pick of a convenient number.
- A `416` (a stale `.part` past the end of the resource) now adopts `Content-Range: bytes */N` as
  the reference. That header is the one trustworthy measurement a `HEAD`-less server still emits,
  so such a mirror becomes certifiable instead of unusable.
- `fetch_month()` and the `caissa-download` CLI gained `expected_bytes`; the manifest write in
  `fetch_month()` is now reachable only after a certified download.
- `UnverifiableDownload` subclasses `IncompleteDownload`, so callers that already treated an
  incomplete download as retryable keep working unchanged.

## Tests — evidence

Two tests were added to `tests/unit/test_lichess_bulk.py` (the file predates the session; the
two new cases are appended and the shared helper was reused unchanged):

- `test_truncated_body_without_trusted_length_is_not_certified` — a short, unmeasurable body is
  refused, the file is not published, and **no manifest is written** (the manifest writer is
  wrapped to observe it).
- `test_trusted_reference_certifies_a_body_the_server_will_not_measure` — three outcomes of the
  reference: absent → refuse; contradicts the body → refuse; agrees → accept and record.

Both failed before the fix (`AttributeError: ... has no attribute 'UnverifiableDownload'`, i.e.
the guard did not exist) and pass after it. The reproducer above, re-run against the fixed module,
now raises loudly:

```
UnverifiableDownload: lichess_db_standard_rated_2013-01.pgn.zst.part: 153600 bytes received but
the source advertises no length and no reference was supplied; refusing to certify an
unmeasurable body (Article 14). Pass expected_bytes (a published volume, index file or prior manifest).
```

**Local gate (dev-proof, Article 12):**

| check | result |
|---|---|
| `ruff check` | All checks passed |
| `black --check` | 41 files unchanged |
| `mypy src/caissa` | no issues, 32 source files |
| `pytest -q` (full suite) | **39 passed** (37 before + 2 new) |
| `tests/unit/test_lichess_bulk.py` | 9 passed (7 before + 2 new) |
| clone guard (`git ls-files -i -c --exclude-standard`) | empty — no tracked ignored file |

No data and no weights entered git (Article 8): the session's only manifest artifact was created
by the throwaway reproducer, caught during the local gate, and deleted — its content is quoted
above as evidence, not committed.

## Risks

- None open. The stricter rule can make a real download fail where it previously "succeeded" —
  that is the point: a body you cannot certify is not a shard, it is an unknown. The
  `expected_bytes` path is the documented remedy, and the `416` path keeps `HEAD`-less mirrors
  usable.
- The build phase does not exercise the loader against the real network (§2); the field behaviour
  of Lichess's actual `HEAD`/`Range` responses is Phase 1 work (`make data-sample`).

## Next stone

GitHub **#4** — streaming PGN parser: replace the eager `iter_games` path with a true generator
over `jsonl`-style streaming, fix the `is_rated_standard` heuristic that accepts any game with
two numeric Elos, and add a memory-bound test (parse a large PGN with bounded RSS).
