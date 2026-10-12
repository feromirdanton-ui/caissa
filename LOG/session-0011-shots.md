# Shot registry — session 0011 «Сила-2» (APPEND-ONLY)

Every dispatch attempt stays in this file forever, including one that never starts. A silent retry
does not exist. The chain «pre-registration → shot → verdict» must have no gaps.

---

## Registry

| # | timestamp (UTC) | run URL | head_sha | dispatched | outcome |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

**No entries.** As of the writing of this file, session 0011 has **fired no shot** — the run was not
dispatched, because the session's file tree could not be published byte-exactly to `main` (see
`LOG/session-0011.md` §5: two transcribed characters in the base64 payload of the pre-registration).
The registry is empty because nothing was fired, not because a record was omitted.

When the shot is fired, this table gains a row **before** any verdict is read, and the pre-registered
`head_sha` is checked after dispatch so that "the run I read" and "the commit I published" are known
to be the same commit.

## Rules this file enforces

1. **Append-only.** Rows are never edited or removed. A cancelled run stays, marked VOID.
2. **URL at the moment of dispatch.** The run URL is written when the run is dispatched, so the
   provenance chain cannot be reconstructed after the fact from memory.
3. **head_sha verified.** A run whose `head_sha` is not the expected commit is VOID regardless of its
   outcome — a run against the wrong tree is not a measurement.
4. **INVALID is recorded, not retried silently.** A run that fails the pre-registered validity gate
   (non-finite metric, or legality < 100 %) is recorded as INVALID; the fix is a NEW pre-registration
   and a NEW shot, never an edit to the threshold.
