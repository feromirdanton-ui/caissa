# ADR-0003 — Phase acceptance protocol

- **Status:** accepted (session 0002)
- **Date:** 2026-10-11
- **Supersedes:** nothing
- **Related:** GOAL.md §3 Articles 5 and 7, §5 (phase DoD), §6.2 (session cycle), GOAL.md §9

## Context

Phase 0 was delivered together with a report stating that the phase was complete: repo skeleton,
CI, green `make lint test`, Requiem-1 in place. The report was accurate about *the files* but the
phase had never been reproduced by anyone other than its author, and the delivered artifact is a
zip, not a repository. Two facts made the difference between a report and a result concrete:

- a completion report can be written more easily than a working build, so a report is a *claim*
  (Article 5 forbids passing a claim off as a result);
- the delivered tree passed `pytest` from the zip while failing 3 tests from a `git clone` of the
  same tree, because one test fixture was matched by `.gitignore`. Nothing in the report was
  untrue, and nothing in the report could have revealed it.

The Definition of Awesome in GOAL.md §2 ("an engineer does `git clone && make train`") is a claim
about a *clone*, not about a zip. A phase that is only green as a zip is not green.

## Options considered

1. **Accept on report.** The author runs the commands and reports the outcome; the architect reads
   the report and closes the phase. *Pro:* costs nothing, keeps the session cycle fast. *Con:* the
   check with the highest information content — that the artifact is self-sufficient on a machine
   that did not produce it — is exactly the one that is skipped. Every honest mistake found in
   session 0002 (gitignored fixture, empty `tools/`, inflated smoke percentage) survived this
   protocol.
2. **Accept on pasted logs.** The author attaches a log of the runs. *Pro:* cheap, better than
   narration, closes the "no evidence at all" hole. *Con:* a log is still text produced by the
   author; it does not say which tree was run, and it cannot show what a *fresh* checkout contains.
   It also silently omits what was not run.
3. **Accept on clean-room reproduction.** The artifact is identified by hash, unpacked into an
   empty directory by the *acceptor*, the DoD commands are executed there, and the raw
   command → output pairs are recorded. *Pro:* it directly tests self-sufficiency, it is repeatable,
   and it composes with the fast session cycle. *Con:* costs an extra session, needs a stated
   toolchain, and can only approximate "clean machine" (see consequences).
4. **Accept on CI alone.** Trust a green GitHub Actions run. *Pro:* automation, no human step.
   *Con:* CI runs on the *author's* checkout; it cannot see files that git refuses to track (the
   very P0 of session 0002), and in this delivery no workflow has ever run because the artifact
   contains no `.git`.

## Decision

Adopt **option 3**, extended with an explicit clone simulation, as the mandatory protocol. A phase
is complete only when all of the following hold:

1. **The artifact is identified** by sha256 of the delivered file, plus the file count.
2. **The clean room is fresh:** the artifact is unpacked into a new empty directory, and the DoD
   commands are run from that directory — never from the working copy that produced it.
3. **A clone simulation is run:** a copy with every `.gitignore`-matched file and every empty
   directory removed (what git would actually give a reader) must also pass `make lint test`.
   This step exists because it is the only one that catches "the file exists but is not in the
   repository".
4. **Raw evidence is recorded:** each DoD item is entered in `LOG/session-NNNN.md` as
   command → observed output (or exit code), including the failures. A DoD item with no captured
   output is not accepted.
5. **The checklist is answered independently** of the author's own tests where the claim is
   load-bearing: e.g. the Requiem-1 FEN is parsed out of GOAL.md §8 rather than taken from the
   repo's own constant, and the legality mask is re-checked against `board.legal_moves` on random
   games by audit-owned code.
6. **What could not be verified is named**, with the reason (no git, no Actions runner, no network)
   and the strongest available substitute evidence.
7. **Deviations become issues** with a `P0`–`P3` label. A P0 is fixed in the same session when it
   fits in ≤3 attempts; otherwise it is `parked` with a documented workaround (GOAL.md §6.5).
8. **Only then** is the phase marked done, `LOG/session-NNNN.md` written, and — once a real
   repository exists — the tag created. A report without reproduction is not a phase completion.

Corollary: a report is always welcome as *navigation* — it tells the acceptor where to look — but
it is never the evidence.

## Consequences

- Sessions get longer, and the acceptance step needs its own artifacts (clean-room directory, logs).
  This is the intended cost: it is bought back the first time a phase is *not* rebuilt twice.
- "Clean machine" must be stated precisely, because it is usually an approximation: the record must
  name the OS, the interpreter and the tool versions, and must say which parts were simulated
  (session 0002 ran on Windows without `make`, without `git` and without a CI runner, so the
  Makefile steps were executed as their underlying commands and the clone was simulated by
  applying `.gitignore` semantics by hand).
- The protocol makes the artifact's *packaging* part of the deliverable. A zip that is not a git
  repository cannot satisfy the Definition of Awesome, no matter how green its tests are.
- Requirements gain teeth: "everything resumable" (Article 4) and "mocks are labelled" (Article 5)
  are now checked by reproduction, not by reading the module docstring — a docstring claiming a test
  exists is not a test.
- Slight risk of ritual: the checklist must keep targeting load-bearing claims, or it decays into
  box-ticking. The list in §Decision is deliberately short.

## Open questions

- Should the clean room be a container image (`docker run` on a pinned base) so "clean machine"
  stops being an approximation and stops depending on the acceptor's OS? This would also give
  Windows users a working `make`.
- How should the protocol treat a phase whose DoD depends on network resources (a real Lichess
  month, TWIC)? Session 0002 tested the downloader against a local HTTP fixture instead; that is
  honest but weaker, and the difference should be recorded as a known gap rather than as a pass.
- Who is allowed to accept? Self-acceptance (author = acceptor) should probably require a second
  independent reproduction, otherwise the protocol degrades back to option 1.
