# Repository Guidelines

## Project Structure & Module Organization
- `mjepa_cifar10/` contains the core library code (data pipeline in `data.py`, training loop in `pretrain.py`).
- `scripts/pretrain.py` is the CLI entrypoint used by the Makefile targets.
- `config/pretrain/` holds YAML experiment configs (e.g., `vit-small.yaml`).
- `logs/` is the default training output directory.
- `tests/` is reserved for pytest tests (currently empty).
- `Makefile`, `pyproject.toml`, and `uv.lock` define tooling and dependencies.

## Build, Test, and Development Commands
- `make init` — initialize the `uv` environment and install all dependency groups.
- `make deploy` — install runtime dependencies from the lockfile (no dev tools).
- `make update` — sync dependencies after `pyproject.toml` changes.
- `make train` — run training; uses `Makefile.config` and selects DDP when `NUM_TRAINERS > 1`.
- `make train-single` — force single‑GPU training.
- `make check` — run formatting, linting, and type checks.
- `make style` — auto‑fix formatting and lint issues via Ruff.
- `make quality` — Ruff lint + formatting checks (no fixes).
- `make types` — run basedpyright type checks.

## Coding Style & Naming Conventions
- Python 3.11–3.13, 4‑space indentation, max line length 120 (Ruff).
- Use Ruff for linting/formatting; keep imports sorted (Ruff isort rules).
- Run `make style` before pushing.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, module files in `snake_case.py`.
- Configs in `config/pretrain/` should follow `vit-<size>.yaml` naming when adding variants.

## Testing Guidelines
- Tests are expected under `tests/` and should be named `test_*.py`.
- Run tests with `uv run pytest` (or `pytest` if your env is active).
- Mark long GPU‑bound tests with `@pytest.mark.ci_skip` to keep CI fast.
- Coverage is tracked by Codecov, but no strict threshold is enforced.

## Commit & Pull Request Guidelines
- Commits use short, imperative summaries; include PR/issue references when relevant (e.g., "Add warmup schedule (#12)").
- PRs should include: a brief summary, the exact command used to reproduce (e.g., `make train`), config file path, and any W&B run link or metrics.

## Configuration & Secrets
- Create a local `Makefile.config` from `Makefile.config.template`; it is git‑ignored and stores dataset paths, device IDs, and experiment names.
- Log in to Weights & Biases before training; do not commit API keys or dataset paths.
