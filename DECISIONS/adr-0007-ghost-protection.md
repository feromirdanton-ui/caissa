# ADR-0007 — Ghost protection: the suite must test *this* checkout

- **Status:** Accepted (session-0005)
- **Date:** 2026-10-11
- **Trigger:** an incident observed live at the start of session-0005.
- **Related:** GOAL.md §3 Articles 11–12 (repository is the source of truth; green CI is the proof), ADR-0004, ADR-0005.

## Context — the incident

The session began by adopting the previous state into a **new checkout** (`caissa-v4`), as the
continuation rules require. The previous checkout (`caissa-v3`) had been installed editable with
`pip install -e .`, which had written an editable pointer into `site-packages`. Because the new
checkout was *also* on `sys.path` only when run from its own directory, the old editable install
won the import:

```
$ python -c "import caissa; print(caissa.__file__)"
/workspace/caissa-v3/src/caissa/__init__.py     # the PREVIOUS session's tree
```

`pytest` collected the new tests but imported the **old** package. Every "green" run in this
situation is a lie about the tree it claims to test: it can pass because the *previous* code is
correct, while the new change is broken — or fail for the reverse reason. This is the same class of
failure as session-0002's P0-1 (green locally, red for a clone): the environment, not the code,
decides the verdict, and it decides it silently.

## Options

1. **Do nothing; trust the developer to reinstall.** Rejected: the failure is silent and produces a
   confidently wrong "green". Article 12 makes a green run the acceptance proof, so a green run that
   does not name its tree proves nothing.
2. **Force `sys.path` order in `conftest.py`.** Works, but it is a runtime patch that hides the
   problem instead of reporting it, and it does nothing for a plain `python -c "import caissa"` or a
   console script.
3. **Pin `pythonpath = ["src"]` in pytest configuration and add a test that asserts the resolved
   provenance of the package.** The import path is deterministic, and the wrong-tree condition
   becomes a **loud, named failure** at the top of the suite. **Chosen.**

## Decision

Three things, all cheap and permanent:

- **`[tool.pytest.ini_options] pythonpath = ["src"]`** — pytest imports `caissa` from *this*
  checkout's `src/` regardless of what a stale editable install left on `sys.path`.
- **`tests/unit/test_provenance.py`** — asserts, in the suite, that `caissa.__file__` resolves
  inside the repository root and to a `src/caissa/` layout. If a future ghost (a sibling checkout,
  a copied install, a `pip install` of a different tree) satisfies the import, the suite names it
  with the exact remediation (`pip install -e .`) instead of passing.
- **README → "Локальный запуск"** — documents the trap for humans: switching checkouts while an
  editable install is active leaves a ghost; reinstall after moving, and use a virtualenv per
  checkout.

## Consequences

- **Positive:** "green" now carries provenance. A wrong-tree import fails loudly and early, before
  any assertion can be trusted.
- **Positive:** the guard is version-agnostic and tool-agnostic — it protects pytest today and any
  other runner that sets the same `pythonpath`.
- **Cost:** one extra test and one config line; near zero.
- **Known limit:** the guard protects the *test* import. A console script (`caissa-download`) still
  resolves through the entry point that installed it, so a stale install can still shadow a CLI
  invocation; the README says so. A virtualenv per checkout is the general fix.
- **Follow-up:** when CI gains a job that runs on a *second* checkout of the same repo, add a
  provenance assertion there too.
