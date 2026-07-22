# Repository Guidelines

## Project Structure & Module Organization

- `mjepa_cifar10/` contains the core library code (data pipeline in `data.py`, training loop in `pretrain.py`).
- `scripts/pretrain.py` is the CLI entrypoint used by the Makefile targets.
- `config/pretrain/` holds YAML experiment configs (e.g., `vit-small.yaml`).
- `logs/` is the default training output directory.
- `research/studies/` contains committed study specifications; `research/baselines/` contains immutable, hashed metric curves approved for fixed-reference follow-ups; `research/LOG.md` is the append-only result record.
- `.agents/skills/autoresearch/` defines the generic empirical-research safety contract vendored from the template.
- `.agents/skills/run-jepa-research/` adds the JEPA and CIFAR-10 adapter protocol.
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
- `uv run python scripts/research.py monitor <study.yaml> --no-launch`: strictly read-only monitoring for delegated agents.
- `uv run python scripts/research.py notify <study.yaml> <run-id> [--requeue]`: reconstruct or explicitly requeue one terminal notification.
- `uv run python scripts/research.py register-root --root logs/research`: register or migrate one exact notification root.
- `uv run python scripts/research.py notify-worker --once --root logs/research --study-id <study-id>`: deliver due notifications for one study directly through the running local Codex app-server daemon socket.
- `uv run python scripts/research.py event-controller --root logs/research --study-id <study-id>`: watch trainer, supervisor, and terminal lifecycle events without model polling while isolating delivery from unrelated historical notification failures.
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
- A Muon-only follow-up may reuse a completed seed-0 baseline only through `baseline_reference`: commit the exact metric curve under `research/baselines/`, record its SHA-256 in the study YAML, run all configured candidates at seed 0, and label a qualifying result `reference-promotion`. This does not authorize paired confirmation or supervised evaluation.
- Require `50 GiB + 2 * concurrent_jobs * estimated_checkpoint_size` free before every launch.
- Never delete legacy weights. Managed retention may delete only terminal rejected runs under the exact study run directory, and deleted weights are not recoverable.
- Do not launch from dirty, stale, unpushed, protected, editable, or SHA-mismatched study environments.
- Apply the repository-scoped `$autoresearch` skill before `$run-jepa-research`.
- Declare and gate W&B emissions per operation. Launch emits `metrics`, `configs`, and `provenance`; summary emits `metrics` and `provenance`. Record requested and effective modes locally before an external write.
- Use app-server lifecycle notifications as the primary wake path and sparse monitoring only as a fallback. Never keep a Codex turn open to sleep or poll. Check at 10 and 20 minutes after launch, then every 30 minutes. Pin read-only scheduled checks to GPT-5.6 Luna with medium reasoning; the primary goal agent owns state transitions and mutations.
- Run the non-model event controller for new managed launches with the active `--study-id`. Direct Unix-socket delivery is the default, and the CLI discovers the running daemon socket. It watches durable state with inotify, supervisor exits with pidfds, and progress deadlines with local timers. Study-scoped delivery prevents a stale notification from another study from disarming the active wake path. Wake once after the first train-validation-checkpoint cycle, on exceptional safety events, and on terminal state. Never wake for routine progress writes or notification retry writes.
- When a research report is already being produced, sample current Codex rate-limit telemetry once if available and include a compact usage snapshot. Never schedule, wake, wait, or poll solely for usage reporting, and do not advance research monitoring counters for it.
- Advance a run's routine-check count only when its recorded `next_check_at` is due. A wake for another run must preserve its schedule.
- Never let training wait for Codex or let notification failure change terminal run status. Test app-server integration only with fake servers.
- Run notification sweeps only against an exact root registered by the research launcher or `register-root`; reject missing, mismatched, symlinked, repository, home, or broad roots before scanning.
- When an authorized pull request contains terminal comparative research results, update its body after the result commit is pushed with a `## Findings` table generated from the committed structured summary. Include every evaluated variant, key hyperparameters, primary and convergence metrics, elapsed wall time, decision, total study span, and summed run time or compute cost; mark censored values and distinguish active from wall time. Omit the section for protocol-only changes and active studies.
