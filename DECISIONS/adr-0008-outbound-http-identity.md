# ADR-0008 — One outbound HTTP identity: the 406 that only the runner could see

- **Status:** Accepted (session-0005)
- **Date:** 2026-10-11
- **Trigger:** the FIRST run of `data-sample.yml` on a clean runner failed (run 38118099543).
- **Related:** GOAL.md §3 Article 14, §6.6; ADR-0004 (a green run on a clone is the proof), ADR-0006.

## Context — what failed, and why the fixture could not see it

The first clean-runner run of `data-sample.yml` (2026-10-11) pulled the day of Lichess bulk
(8 388 608 B, 56 962 games) and the puzzles archive (307 234 795 B, 6 157 341 rows) correctly,
then failed on TWIC:

    RuntimeError: [twic] no issues dated 2024-01 were found within the probe window
    (anchor 1500->2023-08-08, measured 2026-10-11).

The probe was not broken. Reproduced from the sandbox:

    HEAD https://theweekinchess.com/zips/twic1524g.zip
      default python-requests User-Agent  -> 406 Not Acceptable
      curl / a browser User-Agent         -> 200 OK, Last-Modified: Tue, 23 Jan 2024 ...

The site filters on the User-Agent string. `requests` sends `python-requests/…` by default and
every loader sent no User-Agent at all, so **every** TWIC request — the HEAD probe *and* the
GET download — was answered 406. The loader read a 406 as "issue does not exist", so the window
came back empty and the month was refused with a message that blamed the window, not the 406.

The local test fixture cannot see this class of failure: `http.server` — like any static file
server — answers every client regardless of User-Agent. The tests were green, and *correctly* so
for what they asserted; they simply could not assert the one thing that mattered on the real
site. This is exactly the value of ADR-0004's rule: a green run on a clean runner is the proof,
and the *first* such run paid for itself by finding this.

## Options

1. **Widen the probe window.** Rejected: it does not help — a 406 is not "wrong number", it is
   "no answer to this client". Widening would spend hundreds of polite requests to learn nothing.
2. **Set the User-Agent only in the TWIC loader.** Works, but a server-side UA filter is not
   TWIC-specific and could bite any source; the identity should be one decision, not four.
3. **One shared outbound identity, sent by every loader.** `src/caissa/acquisition/http.py`
   holds `USER_AGENT`; `lichess_bulk` (HEAD, GET, prefix GET), `puzzles` (via `download`) and
   `twic` (the probe) all send it. **Chosen.**

## Decision

- `caissa.acquisition.http.USER_AGENT` = `Caissa-1/<version> (+<repo url>)` — a client that says
  who it is (polite, per §6.6), not one that disguises itself as a browser.
- Every request carries it explicitly (`headers={"User-Agent": USER_AGENT}`), so a caller's bare
  `requests.Session()` cannot silently drop it.
- `twic`'s refusal message now names the 406 as a possible cause, so the next failure is
  diagnosable from the log alone.
- Regression test `test_every_outbound_request_identifies_itself` records the User-Agent at the
  fixture and fails if any loader's request arrives without it — the fixture cannot reproduce the
  406, but it can prove the identity is always sent.
- `test_twic_anchor_is_still_honest` (marked `network`, deselected by default) re-measures the
  live anchor so a drifted constant is caught deliberately.

## Consequences

- **Positive:** TWIC works on a clean runner: `fetch_twic_month("2024-01")` found issues
  1523 (2024-01-15) and 1524 (2024-01-23) and downloaded one (6 627 games) — verified live from
  the sandbox after the fix.
- **Positive:** the identity lives in one place; a future site that filters on it is a one-line
  change.
- **Cost / honest gap:** the fixture still cannot reproduce a server-side UA filter. The
  regression test proves we always *send* an identity, not that a given site *accepts* it; only
  the runner (or a live network test) can prove acceptance. Recorded, not hidden.
- **Follow-up:** if a second source starts filtering, promote `USER_AGENT` handling into a small
  `session()` helper so retry/backoff and identity travel together.
