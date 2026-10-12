# Shot registry — append-only

Every dispatch attempt stays here forever. A run that did not start, or started on the wrong
commit, is recorded as what it is. There are no silent retries.

| # | session | workflow | run | head_sha | outcome | note |
|---|---|---|---|---|---|---|
| 0009-1 | 0009 | smoke-train-v2 | [38186826762](https://github.com/feromirdanton-ui/caissa/actions/runs/38186826762) | `ff7e2813` | **VOID** | started on the wrong commit (ref/dispatch race); cancelled |
| 0009-2 | 0009 | smoke-train-v2 | [38186843291](https://github.com/feromirdanton-ui/caissa/actions/runs/38186843291) | `584517fd` | success | verdict INCONCLUSIVE; model destroyed (NaN) |
| 0010-1 | 0010 | smoke-train-v2 | [38213524325](https://github.com/feromirdanton-ui/caissa/actions/runs/38213524325) | `fd16b091` | success | verdict INCONCLUSIVE; val loss finite |
| 0011-0 | 0011 | — | — | — | **build without a shot** | the instrument was built and the local gate was green, but the publish transport lost two base64 characters, so the tree never reached the repository and no run was dispatched. Recorded as a fact, not a gap. |
| 0012-1 | 0012 | materialize | [38250168401](https://github.com/feromirdanton-ui/caissa/actions/runs/38250168401) | `266bf566` | **VOID** | payload verified (152 blobs) and commit created, but the push was rejected: the Actions `GITHUB_TOKEN` cannot write `.github/workflows/*`. Platform limit, not a transport bug. |
| 0012-2 | 0012 | materialize | [38251100097](https://github.com/feromirdanton-ui/caissa/actions/runs/38251100097) | `266bf566` | success | published 151 files; `main` → `63dac0a0` |
| 0012-3 | 0012 | smoke-train-v2 | [38251372886](https://github.com/feromirdanton-ui/caissa/actions/runs/38251372886) | `bf379d66` | success | **the shot.** Artifact `smoke-train-v2-metrics` id `11748652041`. Machine printed SUCCESS; corrected verdict **DEAD** (see `LOG/session-0012.md`). |
| 0012-4 | 0012 | materialize | [38253349593](https://github.com/feromirdanton-ui/caissa/actions/runs/38253349593) | `bf379d66` | success | published the session-0012 files (LOG, ADR-0012, the polarity fix and its test); `main` → `5d8229d6` |

## Notes

- **0012-3** is the shot this session was for. Its `head_sha` was verified against the polled
  `refs/heads/main` before the run was read, so there is no race.
- The machine's `duel_verdict` for 0012-3 was `SUCCESS`; that was a **defect in the rule's
  implementation** (McNemar polarity), not a result. The pre-registered rule, correctly applied,
  gives **DEAD**. The raw artifact is preserved unmodified; the correction is in the LOG.
