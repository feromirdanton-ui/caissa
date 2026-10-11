# ADR-0011 — Two-speed mode: fast lane and deep lane

- **Status:** Accepted (session-0007)
- **Date:** 2026-10-12
- **Milestone:** Vertical (Phases 3+4 merged)
- **Related:** GOAL.md §3 Articles 5, 7, 9; §5 Phases 3–4; §6.2, §6.4, §6.7; ADR-0003, ADR-0004.

## Context

The build has been running as a sequence of milestones, each closed by a human signature. Two
failures of *rhythm* have become visible:

1. **The fast lane and the deep lane were not distinguished.** A cheap, honest check (a lint
   rule, a property test, a guard trap) and an expensive experiment (a real training run on
   real shards) competed for the same "session outcome" slot. The cheap check was therefore
   often postponed to make room for the expensive one — and the cheap checks are exactly the
   ones that protect truth.
2. **The vertical was never exercised end to end.** The pipeline had never pulled real data
   through real shards into a real (tiny) model. Every part was green in isolation; the
   *composition* was untested. That is the one class of defect the existing unit suite
   structurally cannot see, and it is the class that produced the zstd-frame and label-routing
   defects (session-0006).

There is also a temptation to grow *ceremony*: new phases, new ADRs, new declarations. The
constitution already answers this (§6.4: a brilliant idea ≠ a task; new features enter only via
ADR + priority). This ADR makes the rhythm explicit and forbids the ceremony that does not
answer a question.

## Options

1. **Keep the current single-lane rhythm.** Rejected: it keeps postponing the cheap truth-checks
   and never forces the composition to be exercised.
2. **Run two parallel tracks with two signatures per milestone.** Rejected: signature inflation.
   A signature attests to a *milestone*, not to a lane. The human's attention is the scarcest
   resource in the project — spending it twice per milestone devalues it.
3. **Two lanes inside one milestone; one signature per milestone.** **Chosen.**

## Decision

### 1. Two speeds

- **Fast lane — minutes, every session, no download, no torch, no GPU.**
  What belongs here: lint/type/test gates, property tests, new fixtures for realism, the golden
  suite, guard traps, docs, the silence guard (§2), DoD reconciliation tables, ADRs that answer a
  question. The fast lane **never** depends on the network or on a large download; it is runnable
  in a session of minutes. Its job is to make mistakes *visible*.
- **Deep lane — one real experiment per crossroad, ≤ 60 min CPU on the runner.**
  What belongs here: an end-to-end run that exercises the composition on real data — data-sample →
  shards → train → metrics. Its output is a **run URL + an artifact**, never a claim. Weights
  produced by a deep-lane run are **disposable run artifacts, never git objects** (Article 8).

The deep lane is not "harder fast lane" — it answers a *different* question: not "is each part
correct?" but "does the whole hold together?".

### 2. The sacred — never to be simplified or skipped

These protect *truth*, which is the project's entire value axis (§1). They are expensive on purpose
and cannot be waived for schedule:

1. The constitution (GOAL.md) — re-read, not paraphrased (§3 Article 10).
2. The legality mask (Article 1) and Requiem-1 (§8) — never deleted, never weakened.
3. Honesty of artifacts (Article 5) — a mock is labeled a mock.
4. Manifests and sha256 (Article 8, Article 14) — a record only after verifying the written file.
5. The clone guard and a green CI from a fresh clone (Article 7, ADR-0003).
6. Publish → verify → CI (ADR-0009).

### 3. The abolished — ritual without a question

Removed by this ADR, and not to be reintroduced without a fresh ADR that answers a question:

1. **Phase 3 and Phase 4 as two separate milestones.** They are the two halves of one thing: the
   representation that the model consumes and the model that consumes it. Split, each could close
   "green" while the composition was never run. Merged into the milestone **«Vertical»**.
2. **A phase boundary without an end-to-end run.** A milestone closes on a *deep-lane run of the
   composition*, not on a set of green unit suites.
3. **Declarations without a run URL** (already law since session-0006; restated here as abolished).
4. **A new ADR that does not answer a question.** "Documenting" a decision nobody contested is
   ceremony. Every ADR must name the alternatives it kills.
5. **Creating an artifact "for completeness"** — a workflow, a module, a config with no consuming
   question, and no test. Such a thing is dead weight that will be mistaken for readiness.

### 4. Roadmap after this ADR

```
Phase 0 Foundation ── closed, signed
Phase 1 Acquisition ─ closed, signed (v0.1.0)
Phase 2 Processing ── closed, signed
──────────────────────────────────────────────────────────
MILESTONE «Vertical»  (Phases 3 + 4 merged, + smoke-train)
    representation + legality mask  →  tiny model  →  REAL data  →  metrics
    closes on: one deep-lane run (ci + smoke-train) + LOG. ONE human signature.
──────────────────────────────────────────────────────────
MILESTONE «Recipes»  (next: nano/main configs, runbook walked on tiny, DDP/FSDP stubs)
──────────────────────────────────────────────────────────
```

**One signature per milestone** (ADR-0003 unchanged in mechanism; this ADR changes only the
*count* — one, not one per lane and not one per sub-phase).

### 5. Pre-registration of the falsification criterion

Any deep-lane experiment must have its **falsification criterion written down before it runs**, in
the LOG, in the same session. Retouching a criterion after seeing the numbers is forbidden (§9,
Article 5): a negative result is recorded and signed with the same care as a positive one, and is
worth more than a retouched positive one. The first application is the smoke-train of session-0007.

## Consequences

- **Positive:** the cheap truth-checks can no longer be starved by an expensive run — they are a
  different lane, not a competing outcome.
- **Positive:** the composition is exercised at every milestone close, which is the only place the
  zstd-frame/label-routing class of defect is visible.
- **Positive:** the human signs once per milestone; attention is spent on verdicts, not on ceremony.
- **Cost:** merging Phases 3 and 4 makes the milestone larger. Mitigated by the deep lane being a
  *single* ≤60 min run, so the milestone closes on one artifact rather than a long campaign.
- **Cost:** "fast lane" invites scope creep — everything cheap looks like it belongs. Bounded by the
  rule that a fast-lane item must be a *check* (a test, a fixture, a report), never a feature
  without an issue (§3 Article 9).

## Open questions

- Does every *crossroad* need its own deep-lane run, or is one per milestone enough? Current
  answer: one per milestone, plus one per question the fast lane cannot settle.
- The 60-minute budget is a CPU-runner constraint, not a law. A deep-lane experiment that honestly
  needs longer should say so in its LOG entry rather than silently truncating its scale (§Article 5).
