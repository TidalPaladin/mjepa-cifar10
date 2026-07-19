# Repository Guidelines

## Project Structure & Module Organization

- `mjepa_cifar10/` contains the core library code (data pipeline in `data.py`, training loop in `pretrain.py`).
- `scripts/pretrain.py` is the CLI entrypoint used by the Makefile targets.
- `config/pretrain/` holds YAML experiment configs (e.g., `vit-small.yaml`).
- `logs/` is the default training output directory.
- `research/studies/` contains committed study specifications; `research/LOG.md` is the append-only result record.
- `.agents/skills/run-jepa-research/` defines the goal-mode research workflow and retention protocol.
- `tests/` contains unit, configuration-migration, and optional multi-GPU tests.
- `Makefile`, `pyproject.toml`, and `uv.lock` define tooling and dependencies.

## Build, Test, and Development Commands

- `make init`: initialize the `uv` environment and install all dependency groups.
- `make deploy`: install runtime dependencies from the lockfile (no dev tools).
- `make update`: upgrade the lockfile and synchronize all dependency groups after `pyproject.toml` changes.
- `make train`: run training; uses `Makefile.config` and selects DDP when `NUM_TRAINERS > 1`.
- `make train-single`: force single‑GPU training.
- `make check`: run non-mutating formatting, linting, type, and CI-safe test checks.
- `make style`: auto-fix formatting and lint issues via Ruff.
- `make quality`: run Ruff lint and formatting checks without fixes.
- `make types`: run basedpyright type checks.
- `make test`: run tests that do not carry the `ci_skip` marker.
- `make test-ci`: run CI-safe tests and write `coverage.xml`.
- `uv run python scripts/research.py preflight <study.yaml>`: verify code, environment, GPUs, data, and storage.
- `uv run python scripts/research.py launch <study.yaml>`: launch pending managed jobs on physical GPUs 1 and 2.
- `uv run python scripts/research.py monitor <study.yaml>`: recover terminal state and launch the next eligible jobs.
- `uv run python scripts/research.py summarize <study.yaml>`: compute convergence and promotion results.
- `uv run python scripts/research.py inventory`: index legacy and managed local run artifacts without changing them.

## Coding Style & Naming Conventions

- Python 3.11–3.14, 4‑space indentation, max line length 120 (Ruff).
- Use Ruff for linting/formatting; keep imports sorted (Ruff isort rules).
- Run `make style` before pushing.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, module files in `snake_case.py`.
- Configs in `config/pretrain/` should follow `vit-<size>.yaml` naming when adding variants.

## Testing Guidelines

- Tests are expected under `tests/` and should be named `test_*.py`.
- Run the standard local gate with `make check`.
- Run multi-GPU tests separately with `uv run pytest -m ci_skip` on a host with at least two CUDA devices.
- Mark long GPU‑bound tests with `@pytest.mark.ci_skip` to keep CI fast.
- Coverage is tracked by Codecov, but no strict threshold is enforced.

## Dependency Updates

- Pin direct registry dependencies exactly in `pyproject.toml` and commit `uv.lock`.
- Pin Git dependencies by full commit SHA under `[tool.uv.sources]`.
- Use `make update` for planned dependency refreshes, then run `make check`, `make test-ci`, and the dependency audit.
- The default PyTorch wheels use CUDA 13 and require a Turing-generation or newer NVIDIA GPU.

## Commit & Pull Request Guidelines

- Commits use short, imperative summaries; include PR/issue references when relevant (e.g., "Add warmup schedule (#12)").
- PRs should include: a brief summary, the exact command used to reproduce (e.g., `make train`), config file path, and any W&B run link or metrics.

## Configuration & Secrets

- Create a local `Makefile.config` from `Makefile.config.template`; it is git‑ignored and stores dataset paths, device IDs, and experiment names.
- Log in to Weights & Biases before training; do not commit API keys or dataset paths.

## Managed Research Safety

- Invoke `$run-jepa-research` for autonomous JEPA studies and read its protocol before launch.
- Keep experiment adapters here and reusable JEPA primitives in the sibling `mjepa` repository.
- Preserve the online probe invariant: classifier-head gradients are allowed, while teacher features remain detached under `torch.inference_mode()`.
- Use the fixed 45,000/5,000 stratified training split. Reserve the official test set for the confirmed baseline and winner.
- Do not exceed eight pretraining trials, two concurrent jobs, physical GPUs 1 and 2, or 24 hours per job.
- Require `50 GiB + 2 * concurrent_jobs * estimated_checkpoint_size` free before every launch.
- Never delete legacy weights. Managed retention may delete only terminal rejected runs under the exact study run directory, and deleted weights are not recoverable.
- Do not launch from dirty, stale, unpushed, protected, editable, or SHA-mismatched study environments.
- Poll at 10 and 20 minutes after launch, then every 30 minutes. Luna 5.6 medium may perform read-only polling; the primary goal agent owns state transitions and mutations.
