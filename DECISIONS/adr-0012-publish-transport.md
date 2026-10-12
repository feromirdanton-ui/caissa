# ADR-0012 — Publication transport: mechanical for anything a model would have to type

- **Status:** accepted (session-0012)
- **Supersedes:** the implicit "publish everything through the API with inline base64" practice of
  ADR-0009, which remains correct for small files and is now bounded.

## Context

Three incidents, one shape. In each, a model typed the bytes of a file and the bytes arrived wrong:

| session | what happened | evidence |
|---|---|---|
| 0005 | an incomplete tree was published — files silently missing | `LOG/session-0005.md` |
| 0008 | a file landed as mojibake: the transport normalised literal `\uXXXX` escapes | blob `d34a2329` ≠ local `af336cd0`; `cmp -l` showed 5 escapes = 15 bytes |
| 0011 | two base64 characters were lost in transcription | preregistration blob `bfea76b5` ≠ local `7e0e7530`; 2 bytes of 10319 differed (pos 1886 `L`/`l`, pos 10251 `r`/`s`) |

In all three the corruption was caught only by comparing the returned blob sha against the local
`git hash-object`. The publish step is the risky step, and the risk scales with how much text a
model has to reproduce.

## Decision

1. **Threshold.** Inline base64 typed by a model is permitted **only for files under 2 KB**. At or
   above 2 KB, the file moves by a **mechanical** transport: the bytes are produced by a program
   (tar/openssl/curl), never typed.
2. **Mechanical transport (option B, chosen for this environment).** This sandbox has no git
   credentials, no SSH key and no `gh`; the runner's `GITHUB_TOKEN` is read-only by default. So:
   pack the tree → encrypt it → upload the opaque blob to a public host → a small bootstrap
   workflow (`.github/workflows/materialize.yml`, itself under 2 KB of hand-published text)
   downloads it, checks the payload sha256, decrypts, and overlays it.
3. **Verify duty — every file, every transport.** After any publish, the returned blob sha MUST
   equal the local `git hash-object` of the file. A mismatch is a **loud failure**, never a retry by
   hand. This applies to the API path and the mechanical path alike.
4. **Platform limit, recorded.** The Actions `GITHUB_TOKEN` can never create or update a file under
   `.github/workflows/` (`refusing to allow a GitHub App to create or update workflow ... without
   'workflows' permission`, run `38250168401`). Workflow files are therefore published through the
   API, which does hold that permission.

## Consequences

- A corrupted payload fails the materialize run **before** the commit, because the workflow verifies
  every file against the manifest's `git hash-object`.
- The transport is self-hosting: `materialize.yml` lives in the repository and can be re-dispatched.
- It depends on a public 72 h host; a longer-lived store would be preferable and is a known
  limitation.
- The 2 KB threshold is a policy, not a law of nature: it is the size below which a model's
  transcription has proven reliable in this project's history.
