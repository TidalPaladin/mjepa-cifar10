# MJEPA CIFAR-10

Scripts for training a ViT model on CIFAR-10 using [MJEPA](https://github.com/TidalPaladin/mjepa).

## Requirements

- Python 3.11 through 3.14
- uv 0.11.28
- An NVIDIA Turing-generation or newer GPU for the default CUDA 13 PyTorch wheels
- NVIDIA driver 580.65.06 or newer on Linux
- A Weights & Biases account for training logs

## Setup

1. Run `make init` to create the virtual environment from `uv.lock`.
2. Run `wandb login`.

Use `make update` after changing a dependency pin. This upgrades `uv.lock` and synchronizes every dependency group.

## Development checks

- `make check` runs Ruff checks, basedpyright, and the CI-safe test suite without rewriting files.
- `make style` applies Ruff lint and formatting fixes.
- `make test-ci` runs the CI-safe tests and writes `coverage.xml`.
- `uv run pytest -m ci_skip` runs the multi-GPU tests when at least two CUDA devices are available.

## Training

To run model training:

1. Create your training configuration:

   ```bash
   cp Makefile.config.template Makefile.config
   ```

2. Edit `Makefile.config` with your training parameters (data path, device, etc.)
3. Run pretraining:

   ```bash
   make train          # runs distributed or single GPU based on NUM_TRAINERS
   make train-single   # forces single GPU training
   ```

4. Run finetuning from a backbone checkpoint:

   ```bash
   make finetune        # requires CONFIG to point at config/finetune/* and CHECKPOINT to a backbone.safetensors file
   make finetune-single # forces single GPU finetuning
   ```

## Goal-mode research studies

Invoke the repository-scoped `$autoresearch` skill followed by
`$run-jepa-research` when a Codex goal should own a bounded JEPA ablation from
hypothesis through evaluation and retention. The generic skill supplies the
research safety contract; the JEPA skill supplies this repository's commands,
metrics, data protocol, and retention rules. The harness stores recoverable
local state under `logs/research/<study-id>` and can use W&B for authorized
metrics and provenance.

Create a committed YAML file under `research/studies/`, then use:

```bash
uv run python scripts/research.py preflight research/studies/<study-id>.yaml
uv run python scripts/research.py launch research/studies/<study-id>.yaml
uv run python scripts/research.py status research/studies/<study-id>.yaml
uv run python scripts/research.py monitor research/studies/<study-id>.yaml
uv run python scripts/research.py notify research/studies/<study-id>.yaml <run-id>
uv run python scripts/research.py register-root --root logs/research
uv run python scripts/research.py notify-worker --once --root logs/research
uv run python scripts/research.py summarize research/studies/<study-id>.yaml --record
uv run python scripts/research.py storage-report research/studies/<study-id>.yaml
uv run python scripts/research.py inventory --wandb-entity <entity>
```

`launch --dry-run` creates the atomic study state without starting training. A real launch uses physical GPUs 1 and 2, exposes one GPU to each process, runs at most two jobs, and enforces a 24-hour job timeout. Before each launch, the harness checks for at least `50 GiB + 2 * concurrent_jobs * estimated_checkpoint_size` free.

Terminal workers write `terminal.json` before queueing `notification.json`.
Launch registers the exact notification root with `.mjepa-research-root.json`.
Use `register-root` once for an existing root; it is idempotent and migrates the
legacy marker. The marker binds its canonical `root_path`, so the one-shot
worker rejects missing, mismatched, symlinked, repository, home, and broad roots
before scanning. It connects to an existing local Codex app-server daemon,
resumes the originating task, and uses `turn/start` for an idle task or
`turn/steer` for its sole active turn. It retries with bounded jitter, serializes
delivery per task, and records acceptance only after app-server accepts the RPC.
Training never waits for Codex, and delivery failure cannot alter terminal run
status.

Use a same-task scheduled follow-up to run `notify-worker --once`. Keep sparse read-only monitoring as a fallback: check at 10 and 20 minutes to catch startup failures, then every 30 minutes, with no more than five routine checks. A Luna 5.6 medium follow-up may perform read-only monitoring. The primary goal agent keeps responsibility for launches, promotion decisions, code and Git changes, and checkpoint deletion. Automated tests use fake app-server transports and must never wake a real task.

W&B consent is checked independently for every operation. Launch emits
`metrics`, `configs`, and `provenance`; summary emits `metrics` and `provenance`.
Each operation can run online only with a named entity, explicit authorization,
an explicit matching manifest, and approval for all of its emitted classes.
Otherwise it stays local-only. Local provenance records the requested mode,
effective mode, destination, manifest, approvals, and gate decision before the
external write.

The fixed CIFAR-10 evaluation protocol uses 45,000 training examples and a stratified 5,000-example validation set with 500 examples per class. The official test set is reserved for the confirmed baseline and winner. The online probe applies the classifier head to teacher features computed under `torch.inference_mode()`, so isolated probe loss updates only the head.

See [.agents/skills/run-jepa-research/SKILL.md](.agents/skills/run-jepa-research/SKILL.md) for the workflow and [research/LOG.md](research/LOG.md) for the append-only result record. Existing weights under `logs/` are legacy artifacts and are not eligible for automatic retention.
