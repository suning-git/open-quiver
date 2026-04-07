# `common/` — Shared Cross-Project Layer

This directory holds artifacts that are shared across personal workspaces
(currently `ning/` and `ZhK/`). Everything here is treated as a **stable
interface** that multiple projects depend on.

## Collaboration philosophy: lego blocks

`common/` provides **building blocks**. Personal projects assemble them.

- Personal projects (`ning/`, `ZhK/`) own their **glue code, UI, experiments,
  and orchestration logic**.
- `common/` provides **stable, opinion-free pieces** that personal projects
  can opt into.
- A personal project may use any subset of `common/` — or none of it.
- Changes inside personal folders need no coordination. Changes inside
  `common/` do.

## What belongs in `common/`

A piece belongs here only if it satisfies **all** of these:

1. **Stable** — its interface is unlikely to churn.
2. **Opinion-free** — no UI choice, no API key, no prompt experiment.
3. **Already shared in practice** — at least one consumer needs it now,
   and a second one is likely.
4. **Testable in isolation** — high test coverage is required.

Good fits:

- Pure mathematical functions (e.g. quiver mutation primitives).
- Data formats and the data itself (e.g. game definitions in `games/`).
- Schemas and conventions documented in markdown.

## What does NOT belong in `common/`

Things that stay in personal folders:

- Game loops, agent loops, retry policies — opinionated control flow.
- LLM prompts — these evolve with experiments.
- Provider configs, API keys — personal.
- Streamlit / CLI / Tkinter UIs — personal style.
- Visualization choices.
- Stateful classes whose schema is still being explored.

If you find yourself wanting to share an opinionated piece, prefer to copy
it into the other personal project first. Only promote to `common/` after
two consumers have lived with the same shape long enough to trust it.

## Modification rules

- **Adding new files** — generally safe, no coordination required.
- **Editing existing files** — requires coordination with other collaborators.
  Treat existing APIs and data files as load-bearing.
- **Renaming or deleting** — always requires coordination.
- **Breaking changes** — avoid; if unavoidable, announce first.

When in doubt: prefer additive changes. Add a new function, a new file, or
a new field — don't change the meaning of an existing one.

## Discipline

> Resist the urge to share early. It is much easier to promote a piece into
> `common/` later than to pull it back out once others depend on it.

## Current contents

- [`games/`](games/) — Shared game catalog (JSON files defining mutation games).
  See [`games/README.md`](games/README.md) for the file format.
