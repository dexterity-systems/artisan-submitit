# AGENTS.md

Project conventions for contributors (human and AI).

## Committed content

Committed code, tests, documentation, and generated artifacts must make sense
from a fresh checkout and references available to the repository's intended
audience. Do not cite private or uncommitted planning documents by path,
title, identifier, or section number. State the relevant behavior, constraint,
or rationale directly, or reference documentation available to that audience.

Use portable paths and synthetic example identities, and omit incidental
personal or organizational metadata. Preserve required attribution, copyright
and license notices, intentionally published project metadata, and the meaning
and validation of evidence. Do not rewrite immutable evidence or weaken
identity checks to satisfy this rule.

## Environment

| Setting | Value |
| --- | --- |
| Environment | Pixi (`~/.pixi/bin/pixi`) |
| Formatting | Ruff |
| Type checking | mypy |
| Testing | Pytest |

Always use the full Pixi path when running commands. Use Pixi 0.83.0. CI and
agent verification use `--locked` for work that does not change dependencies.
Local development uses ordinary `pixi install` and `pixi run`; source edits,
commits, comments, tasks, and version-only edits do not require lock updates.
After changing dependencies, run `pixi lock` deliberately and review the diff.
Each worktree must own its `.pixi/envs`; do not share installed environments
through symlinks, mounts, or detached-environment configuration. Package caches
may be shared. The Docker cluster deliberately uses its own detached environment
volume; use a separate Compose project and volume for each checkout.

## Commands

```bash
~/.pixi/bin/pixi install --locked
~/.pixi/bin/pixi run --locked -e dev setup
~/.pixi/bin/pixi run --locked -e dev fmt
~/.pixi/bin/pixi run --locked -e dev typecheck
~/.pixi/bin/pixi run --locked -e dev test
~/.pixi/bin/pixi run --locked -e dev pre-commit run --all-files
~/.pixi/bin/pixi run --locked -e dev build-dist
~/.pixi/bin/pixi run --locked -e dev check-dist
~/.pixi/bin/pixi run --locked -e docs docs-build
~/.pixi/bin/pixi run --locked -e dev test-sbatch
~/.pixi/bin/pixi run --locked -e dev test-srun
~/.pixi/bin/pixi run --locked slurm-build
~/.pixi/bin/pixi run --locked slurm-up
~/.pixi/bin/pixi run --locked slurm-down
```

The development environment expects a real sibling Artisan checkout at
`../artisan`; a symlink is not Pixi-lock portable. Set `ARTISAN_WORKTREE` to any
Artisan checkout or nested worktree for Docker commands.

The literal `[project].version` in `pyproject.toml` defines the package version.
The public `__version__` reads installed distribution metadata. A version-only
edit refreshes editable metadata on the next ordinary or locked Pixi command,
without rewriting the lock. No Git metadata or generated version file is needed.
Before a release, update the literal version deliberately; the release workflow
checks both built artifacts against the intended tag before publication.

## Code Style

- Keep changes DRY, simple, and scoped.
- Do not add backward-compatibility shims.
- Validate configuration early and raise specific exceptions.
- Add type hints to every function signature.
- Write Google-style docstrings and comments that explain why.
- Prefer functions shorter than 30 lines.

## Testing

- Tests live in `tests/` and mirror `src/artisan_submitit/`.
- Unit tests must not require SLURM.
- Scheduler-backed tests live in `tests/integration/` and use
  `@pytest.mark.slurm`.
- Cover happy paths, ordering, failures, cancellation isolation, and invalid
  configuration.

The focused guides form a small Jupyter Book 2 site. Keep `docs/myst.yml`
navigation and examples synchronized when behavior changes.

## Git

Branches use `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, or `chore/`.
Commits use `type: Brief description`. Never push without explicit approval.

## Architecture

```text
src/artisan_submitit/
├── slurm.py         # Submitit/AutoExecutor runner and lifecycle router
├── srun.py          # Intra-allocation runner and process lifecycle router
├── srun_worker.py   # Serialized callable/result-file worker protocol
├── worker.py        # Public native-runner composition and ownership evidence
├── cancellation.py  # Shared cancellation budget and exact scheduler evidence
└── utils.py         # Scheduler limits, time parsing, and log helpers
```

Artisan owns batching and unit execution. This package owns only transport to
SLURM and implements the public `artisan.orchestration.runner_api` contract.
Compute-node ownership composes the public `artisan.orchestration.Runner.LOCAL`
facade. Do not import core's private execution or supervision implementations.
