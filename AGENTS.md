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
- Do not exceed eight pretraining trials per managed study, two concurrent jobs, physical GPUs 1 and 2, or 24 hours per job.
- The user-authorized SReLU MLP program may use at most 16 scientific pretraining runs across linked study specifications, with no more than eight runs in any one study. Its one-epoch mechanical smoke run is excluded. Reuse a committed, hash-verified seed-0 baseline only for the width and directional-tuning screens; paired confirmation must launch fresh baseline and winner runs at seeds 0, 1, and 2.
- A Muon-only follow-up may reuse a completed seed-0 baseline only through `baseline_reference`: commit the exact metric curve under `research/baselines/`, record its SHA-256 in the study YAML, run all configured candidates at seed 0, and label a qualifying result `reference-promotion`. This does not authorize paired confirmation or supervised evaluation.
- Require `50 GiB + 2 * concurrent_jobs * estimated_checkpoint_size` free before every launch.
- Never delete legacy weights. Managed retention may delete only terminal rejected runs under the exact study run directory, and deleted weights are not recoverable.
- Keep the primary study repository clean, pushed, non-protected, and matched to its recorded SHA. Tandem repositories such as `mjepa` and `vit` may remain unpushed when they are clean, locally committed on a study branch, pinned by exact SHA, and imported from that commit.
- Treat non-destructive Git operations in this repository as standing-authorized. Commit and push study branches without another permission request. Create and commit tandem-repository branches locally, but never push them without explicit permission.
- Apply the repository-scoped `$autoresearch` skill before `$run-jepa-research`.
- Treat online W&B operations as standing-authorized. Track scientific experiments online, declare and gate emissions per operation, and fail preflight instead of silently launching offline when an online study lacks its destination or manifest. Launch emits `metrics`, `configs`, and `provenance`; summary emits `metrics` and `provenance`. Record requested and effective modes locally before every external write.
- Use app-server lifecycle notifications as the primary wake path and sparse monitoring only as a fallback. Never keep a Codex turn open to sleep or poll. Check at 10 and 20 minutes after launch, then every 30 minutes. Pin read-only scheduled checks to GPT-5.6 Luna with medium reasoning; the primary goal agent owns state transitions and mutations.
- Run the non-model event controller for new managed launches with the active `--study-id`. Direct Unix-socket delivery is the default, and the CLI discovers the running daemon socket. It watches durable state with inotify, supervisor exits with pidfds, and progress deadlines with local timers. Study-scoped delivery prevents a stale notification from another study from disarming the active wake path. Wake once after the first train-validation-checkpoint cycle, on exceptional safety events, and on terminal state. Never wake for routine progress writes or notification retry writes.
- Before managed child spawn, capture the live originating thread's effective permission-profile identity and approval policy from app-server and persist them in the run's immutable `wake-context.json`. When `CODEX_PERMISSION_PROFILE` is unset, omit the override and persist the non-null profile ID resolved by app-server, including an implicit built-in ID. Fail before dispatch if app-server does not report a selectable profile. Never hardcode a permission profile or replace an existing run context.
- After verifying supervisor identities and durable startup state for a dispatched round, immediately return the persistent goal to its event-wait state when the goal API permits it. Do the same after a nonterminal lifecycle event when no immediate mutation remains. The controller reactivates a blocked goal on the next event; do not spend automatic continuations rediscovering that training is still active.
- When a research report is already being produced, sample current Codex rate-limit telemetry once if available and include a compact usage snapshot. Never schedule, wake, wait, or poll solely for usage reporting, and do not advance research monitoring counters for it.
- Treat token-use limits as monitoring-only limits. Count only intervals spent polling or inspecting live experiment state. Exclude initial study setup, implementation, tests, launch preparation and execution, result analysis, and all code or configuration changes during a study. Never use aggregate goal or task token usage to block research. If monitoring-only usage cannot be isolated, report it as unavailable.
- Advance a run's routine-check count only when its recorded `next_check_at` is due. A wake for another run must preserve its schedule.
- Never let training wait for Codex or let notification failure change terminal run status. Test app-server integration only with fake servers.
- Lifecycle delivery must resume with the run's exact captured permission profile and approval policy, verify the returned effective context, and only then query the originating thread goal. A legacy context with a null profile requires explicit recovery; never map it to the current default. An absent field, null effective profile, or any mismatch is a permanent delivery failure. Transition only `blocked` goals back to `active`; respect explicit `paused`, `complete`, `usageLimited`, and `budgetLimited` states.
- Run notification sweeps only against an exact root registered by the research launcher or `register-root`; reject missing, mismatched, symlinked, repository, home, or broad roots before scanning.
- When an authorized pull request contains terminal comparative research results, update its body after the result commit is pushed with a `## Findings` table generated from the committed structured summary. Include every evaluated variant, key hyperparameters, primary and convergence metrics, elapsed wall time, decision, total study span, and summed run time or compute cost; mark censored values and distinguish active from wall time. Omit the section for protocol-only changes and active studies.
