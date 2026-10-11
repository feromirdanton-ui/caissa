# tools/

This directory holds the command-line entry points of the project (GOAL.md §4.6).

The three CLIs are **packaged console scripts**, declared in `pyproject.toml` and implemented
next to the code they drive, so they are installed with the package rather than pasted here:

| command | entry point | module |
|---|---|---|
| `caissa-download` | `caissa.acquisition.cli:main` | `src/caissa/acquisition/` |
| `caissa-shard` | `caissa.processing.cli:main` | `src/caissa/processing/` |
| `caissa-eval` | `caissa.eval.cli:main` | `src/caissa/eval/` |

`make lint` and CI run `ruff`/`black` over `tools` as well as `src` and `tests`, so this file
also exists to keep the path present: git does not track empty directories, and a missing path
made the lint job fail on a fresh clone (session-0002 P1-1).

Ad-hoc scripts that are *not* part of the package belong here as real files.
