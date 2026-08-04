# Repository Guidelines

## Project Structure & Module Organization

- `mjepa_cifar10/` contains the core library code (data pipeline in `data.py`, training loop in `pretrain.py`).
- `scripts/pretrain.py` is the CLI entrypoint used by the Makefile targets.
- `config/pretrain/` holds YAML experiment configs (e.g., `vit-small.yaml`).
- The default ViT-S/4 layout is one CLS token, seven register tokens, and four independent CLS predictor partitions. Use the explicit `vit-small-four-cls-legacy.yaml` configs only to reproduce studies completed with the former four-CLS baseline.
- I-JEPA token-specialization studies group CLS and register tokens as the global prefix. Separate pre-attention and pre-MLP normalization plus LayerScale in every encoder block, split QKV only in the configured leading blocks, and clone every added visual branch from its global branch so the shared and specialized backbones are identical at initialization. Keep attention, output and MLP projections, final normalization, predictor, teacher, and objective unchanged.
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
- `uv run python scripts/research.py register-root --root logs/research`: register one exact version-2 notification root.
- `uv run python scripts/research.py notify-worker --once --root logs/research --study-id <study-id>`: deliver due notifications for one study directly through the running local Codex app-server daemon socket.
- `uv run python scripts/research.py event-controller --root logs/research --study-id <study-id>`: watch trainer, supervisor, terminal, and exact notification-retry deadlines without model polling; durable controller output is written under `logs/research/.event-controller/`.
- `uv run python scripts/research.py start-controller --root logs/research --study-id <study-id>`: start or reuse one detached, study-scoped event controller and return its verified PID and Linux start ticks.
- `uv run python scripts/research.py notify-wait --root logs/research --controller-pid <pid> --controller-start-ticks <ticks> --study-id <study-id>`: bind the active goal wait to one verified direct event-controller process and its exact study scope.
- `uv run python scripts/research.py summarize <study.yaml>`: compute convergence and promotion results.
- `uv run python scripts/evaluate_checkpoint_probe.py <config> <checkpoint> <data> <output> --study-id <study-id> --run-id <run-id> --expected-step <step> --append-metrics`: evaluate a retained online probe on the fixed validation holdout and append an idempotent terminal metric.
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
- For a clean unpushed tandem commit, locate the existing `uv` Git cache clone that contains the prior pinned SHA, fetch the new exact commit from the local tandem checkout into that cache, then run `uv lock --offline` and `uv sync --all-groups --offline`. Verify the installed source SHA; do not hardcode a cache key or edit `uv.lock` by hand.
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
- Preserve the online probe invariant: classifier-head gradients are allowed, while full-view target features are detached at the classifier boundary. EMA targets also remain under `torch.inference_mode()`; shared-student targets must retain gradients for the self-supervised objective.
- For LeJEPA multiview experiments, keep the first global view as the masked-prediction anchor. Additional global/local views feed only same-image invariance and SigREG, retain shared-student gradients, and never replace the fixed single-view validation/probe path. Enable `multi_crop` only with a positive `invariance_loss_weight`.
- Use the fixed 45,000/5,000 stratified training split. Reserve the official test set for the confirmed baseline and winner.
- Do not exceed eight pretraining trials per managed study, two concurrent jobs, physical GPUs 1 and 2, or 24 hours per job.
- The user-authorized SReLU MLP program may use at most 16 scientific pretraining runs across linked study specifications, with no more than eight runs in any one study. Its one-epoch mechanical smoke run is excluded. Reuse a committed, hash-verified seed-0 baseline only for the width and directional-tuning screens; paired confirmation must launch fresh baseline and winner runs at seeds 0, 1, and 2.
- A Muon-only follow-up may reuse a completed seed-0 baseline only through `baseline_reference`: commit the exact metric curve under `research/baselines/`, record its SHA-256 in the study YAML, run all configured candidates at seed 0, and label a qualifying result `reference-promotion`. This does not authorize paired confirmation or supervised evaluation.
- A study may opt into the cost promotion route with `promotion.cost_gain`. Require the configured common-step active-time gain, lower isolated path latency, and no more than 0.005 peak-accuracy loss. Confirmation requires the same three-seed mean gate and at least two paired seeds with both lower active time and lower isolated latency.
- A study may opt into the equivalence route by configuring both `promotion.equivalence_convergence_ratio` and `promotion.equivalence_auc_loss`. Require peak and final accuracy within `maximum_accuracy_loss`, step and active-time convergence within the ratio, and step and active-time AUC within the absolute loss. Confirmation requires the same three-seed mean gate and at least two paired seeds that pass every threshold.
- A study may require an additional seed-0 control with `promotion.screening_control_variant` and `promotion.screening_control_accuracy_gain`. Configure both fields together, name a non-baseline variant, exclude that control from selection, and require the recorded peak-accuracy gain in addition to the standard promotion route.
- Require `50 GiB + 2 * concurrent_jobs * estimated_checkpoint_size` free before every launch.
- Never delete legacy weights. Managed retention may delete only terminal rejected runs under the exact study run directory, and deleted weights are not recoverable.
- Keep the primary study repository clean, pushed, non-protected, and matched to its recorded SHA. Tandem repositories such as `mjepa` and `vit` may remain unpushed when they are clean, locally committed on a study branch, pinned by exact SHA, and imported from that commit.
- Treat non-destructive Git operations in this repository as standing-authorized. Commit and push study branches without another permission request. Create and commit tandem-repository branches locally, but never push them without explicit permission.
- Apply the repository-scoped `$autoresearch` skill before `$run-jepa-research`.
- Treat online W&B operations as standing-authorized. Track scientific experiments online, declare and gate emissions per operation, and fail preflight instead of silently launching offline when an online study lacks its destination or manifest. Launch emits `metrics`, `configs`, and `provenance`; summary emits `metrics` and `provenance`. Record requested and effective modes locally before every external write.
- Apply `$notify-wake` for app-server delivery, authority capture, reconciliation, response validation, and owned goal waits. Keep only research event production, trusted prompts, registered roots, controller behavior, and retry timing in this repository. Pin `notify-wake-runtime==1.0.0` by exact Git SHA and require the Codex 0.146.0 schema.
- Use app-server lifecycle notifications as the primary wake path and sparse monitoring only as a fallback. Never keep a Codex turn open to sleep or poll. Check at 10 and 20 minutes after launch, then every 30 minutes. Pin read-only scheduled checks to GPT-5.6 Luna with medium reasoning. Luna may also run dedicated relay tasks or model-selectable subagents for low-value event validation and summaries. Never change the active root conversation's model; the root model owns goal changes, launches, recovery, scientific decisions, and mutations.
- Start the non-model event controller with `research.py start-controller --study-id <study-id>` for new managed launches. This detaches the direct Python controller from the interactive terminal, reuses an exact matching live controller, and returns the verified identity required by `notify-wait`. Do not wrap it in a generic notify-wake process watch because the generic watch and research notifier use different goal-wait source identities. The controller limits reconciliation and inotify watches to the selected studies, bounds each delivery sweep, and retries a controller-level timeout locally. Direct Unix-socket delivery is the default. Wake once after the first train-validation-checkpoint cycle, on exceptional safety events, and on terminal state. Never wake for routine progress writes or notification retry writes.
- Before managed child spawn, capture the live originating thread's effective permission-profile identity and approval policy from app-server and persist them in the run's immutable `wake-context.json`. When `CODEX_PERMISSION_PROFILE` is unset, omit the override and persist the non-null profile ID resolved by app-server, including an implicit built-in ID. Fail before dispatch if app-server does not report a selectable profile. Never hardcode a permission profile or replace an existing run context.
- After verifying supervisor identities and the controller's durable startup record, use `research.py notify-wait` with its exact PID, Linux start ticks, and study IDs. Enter this owned wait when the goal is active, no immediate work remains, and the goal API permits blocking. Do the same after a nonterminal lifecycle event when no immediate mutation remains. Reactivate only a blocked goal whose exact identity and `updatedAt` match an acknowledged owned lease. Treat unmatched, changed, or uncertain blocked goals as manually blocked.
- When a research report is already being produced, sample current Codex rate-limit telemetry once if available and include a compact usage snapshot. Never schedule, wake, wait, or poll solely for usage reporting, and do not advance research monitoring counters for it.
- Treat token-use limits as monitoring-only limits. Count only intervals spent polling or inspecting live experiment state. Exclude initial study setup, implementation, tests, launch preparation and execution, result analysis, and all code or configuration changes during a study. Never use aggregate goal or task token usage to block research. If monitoring-only usage cannot be isolated, report it as unavailable.
- Advance a run's routine-check count only when its recorded `next_check_at` is due. A wake for another run must preserve its schedule.
- Never let training wait for Codex or let notification failure change terminal run status. Test app-server integration only with fake servers.
- Keep `status`, `monitor --no-launch`, `notify`, `summarize`, and `storage-report` independent of `CIFAR10_DATA`. Require the dataset only for preflight or commands that can launch training, and report unresolved environment variables by name.
- Store new research queues only under each registered root's `.notify-wake/v2/`. Version-1 contexts, notifications, ledgers, and response shapes are inert audit history and must not be parsed, migrated, requeued, or conditionally supported. Reject a mismatch with `unsupported notify-wake contract; cutover required`.
- Codex 0.146.0 goal transitions and idle-turn starts are not atomic. The owned lease and exact `updatedAt` checks detect changes outside the read-write window but cannot prevent a concurrent user or client update inside it.
- Run notification sweeps only against an exact root registered by the research launcher or `register-root`; reject missing, mismatched, symlinked, repository, home, or broad roots before scanning.
- On checkpoint resume, preserve the launch-time isolated-path benchmark and its immutable W&B config; do not recompute or overwrite it.
- When a primary endpoint falls after the last scheduled validation epoch, evaluate the terminal checkpoint with `evaluate_checkpoint_probe.py --expected-step <step> --append-metrics` before the final summary. Use only the fixed validation holdout, keep the model in evaluation and inference modes, and record the derived metric and evaluator hashes.
- When an authorized pull request contains terminal comparative research results, update its body after the result commit is pushed with a `## Findings` table generated from the committed structured summary. Include every evaluated variant, key hyperparameters, primary and convergence metrics, elapsed wall time, decision, total study span, and summed run time or compute cost; mark censored values and distinguish active from wall time. Omit the section for protocol-only changes and active studies.
