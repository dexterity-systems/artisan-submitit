# AGENTS.md

Project conventions for contributors (human and AI).

## Environment

| Setting | Value |
| --- | --- |
| Environment | Pixi (`~/.pixi/bin/pixi`) |
| Formatting | Ruff |
| Type checking | mypy |
| Testing | Pytest |

Always use the full Pixi path when running commands. Use `--locked` for routine
commands so the lock file is not rewritten unexpectedly.

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
~/.pixi/bin/pixi run --locked -e dev test-sbatch
~/.pixi/bin/pixi run --locked -e dev test-srun
~/.pixi/bin/pixi run --locked slurm-build
~/.pixi/bin/pixi run --locked slurm-up
~/.pixi/bin/pixi run --locked slurm-down
```

The development environment expects a real sibling Artisan checkout at
`../artisan`; a symlink is not Pixi-lock portable. Set `ARTISAN_WORKTREE` to any
Artisan checkout or nested worktree for Docker commands.

Versions come from Git tags through hatch-vcs. Do not edit the generated,
gitignored `src/artisan_submitit/_version.py`.

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

The focused guides remain plain Markdown because this small provider has no
Jupyter Book navigation or executable documentation surface. Review their
links and examples directly when behavior changes.

## Git

Branches use `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, or `chore/`.
Commits use `type: Brief description`. Never push without explicit approval.

## Architecture

```text
src/artisan_submitit/
├── slurm.py         # Submitit/AutoExecutor runner and lifecycle router
├── srun.py          # Intra-allocation runner and process lifecycle router
├── srun_worker.py   # Serialized callable/result-file worker protocol
└── utils.py         # Scheduler limits, time parsing, and log helpers
```

Artisan owns batching and unit execution. This package owns only transport to
SLURM and implements the public `artisan.orchestration.runner_api` contract.
