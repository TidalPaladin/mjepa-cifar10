---
name: run-jepa-research
description: Run bounded, recoverable JEPA ablation studies in this repository. Use when a goal-mode Codex agent must form a hypothesis, modify this repository or the sibling mjepa library, launch GPU training, monitor long runs, compare convergence and downstream performance, record results, and apply checkpoint retention.
---

# Run JEPA Research

Apply the repository-scoped `$autoresearch` skill from `.agents/skills/autoresearch`
first. It supplies the recoverability, provenance, monitoring, notification,
logging, and retention rules. This document supplies only the JEPA/CIFAR-10
adapter details.

Use `scripts/research.py` as the persistent interface for JEPA studies. W&B stores metrics, configs, and provenance. `logs/research/<study-id>/state.json` lets a later goal turn recover the run without transcript context.

## Preconditions

1. Read [references/protocol.md](references/protocol.md) in full.
2. Confirm that a persistent goal is active. Use the goal status tool when available. If no goal exists, ask the user to start or authorize one before launching a study.
3. Recover existing work with `status`, `inventory`, and the study's local state before proposing a new run.
4. Keep the online probe unchanged unless the hypothesis targets probing. The classifier consumes teacher features produced under `torch.inference_mode()`. Probe loss may update the classifier head, but not the teacher, predictor, or non-head encoder parameters.

## Study workflow

1. State one falsifiable question, one mechanism, and the result that would reject the hypothesis.
2. Create `research/studies/<study-id>.yaml`. Use baseline seed 0 and no more than three seed-0 candidates in the screening phase.
3. Put CIFAR-10 adapters and experiment wiring here. Put reusable JEPA primitives in the sibling `../mjepa` repository.
4. Use matching `codex/research/<study-id>` branches in each changed repository. Commit tandem `mjepa` or `vit` changes locally; do not push those repositories without explicit permission.
5. Add regression tests before the implementation. Preserve the fixed 45,000/5,000 train/validation split and reserve the official test set for the confirmed baseline and winner.
6. Run `preflight`. Do not launch until the primary repository is clean and pushed and every tandem repository is clean, locally committed, tested, pinned by exact SHA, and installed from that recorded commit. A tandem push is not required.
7. Launch with `scripts/research.py launch`. The harness exposes one physical GPU to each process, uses only GPUs 1 and 2, permits two jobs, and stops each job after 24 hours. Run `start-controller` for the exact study, verify the returned durable startup record, then run `notify-wait` with its PID and Linux start ticks. Use this repository-owned wait only when the goal is active, the goal API permits blocking, and no immediate work remains.
8. Use the detached controller returned by `start-controller --study-id <study-id>` as the primary wake path. The starter reuses an exact matching process, while the controller scopes inotify and reconciliation to the selected study and bounds each delivery sweep. Do not wrap it in a generic notify-wake process watch because that creates a different goal-wait source identity. The v2 queue is under `logs/research/.notify-wake/v2/`; version-1 state is inert and unsupported. The controller waits on persisted notification retry deadlines without model polling, keeps fresh events deliverable while another event backs off, and records durable JSONL output under `logs/research/.event-controller/`. It may wake once after the first train-validation-checkpoint cycle, on a supervisor loss or progress stall, and on terminal state. Never keep a Codex turn open to sleep or poll. Keep the host and Codex app running.
9. Pin read-only scheduled fallback checks to GPT-5.6 Luna with medium reasoning instead of inheriting the chat default. Luna may also run a dedicated relay task or model-selectable subagent for low-value event validation and summarization. Never change the active root conversation's model. The root model retains goal changes, launches, recovery, summaries, promotion decisions, code and Git changes, and checkpoint deletion.
10. Add one opportunistic Codex rate-limit snapshot to research reports that are already being produced when live telemetry is available. Track only tokens spent polling or inspecting live experiment state. Exclude setup, implementation, tests, launch preparation and execution, analysis, and later code or configuration work. Never use aggregate goal or task token usage for an excess-use decision. Do not schedule, wake, wait, or poll for usage alone.
11. Promote, replicate, fine-tune, and record results only through the thresholds in the study protocol. Do not exceed eight pretraining trials per managed study. Apply a larger goal-wide budget only when the user explicitly authorizes it and the protocol records its linked-study allocation.
12. Append the result to `research/LOG.md`, commit and push the primary-repository result, then apply eligible retention. Keep tandem commits local unless their push was separately authorized. Never delete legacy weights.

## Commands

```bash
uv run python scripts/research.py preflight research/studies/<study-id>.yaml
uv run python scripts/research.py launch research/studies/<study-id>.yaml
uv run python scripts/research.py status research/studies/<study-id>.yaml
uv run python scripts/research.py monitor research/studies/<study-id>.yaml
uv run python scripts/research.py monitor research/studies/<study-id>.yaml --no-launch  # read-only monitor
uv run python scripts/research.py notify research/studies/<study-id>.yaml <run-id> [--requeue]
uv run python scripts/research.py register-root --root logs/research
uv run python scripts/research.py notify-worker --once --root logs/research --study-id <study-id>
uv run python scripts/research.py event-controller --root logs/research --study-id <study-id>
uv run python scripts/research.py start-controller --root logs/research --study-id <study-id>
uv run python scripts/research.py notify-wait --root logs/research --controller-pid <pid> --controller-start-ticks <ticks> --study-id <study-id>
uv run python scripts/research.py summarize research/studies/<study-id>.yaml --record
uv run python scripts/research.py inventory --wandb-entity <entity>
uv run python scripts/research.py storage-report research/studies/<study-id>.yaml
```

Use `launch --dry-run` to validate state creation without starting a process. `status`, `monitor --no-launch`, `notify`, `summarize`, and `storage-report` do not require the CIFAR-10 mount. `preflight`, a real `launch`, and launch-capable `monitor` require it and report an unresolved `CIFAR10_DATA` variable explicitly. Use `summarize --apply-retention` only after metrics, provenance, decisions, and the result log are committed and pushed.

Use `launch --retry-failed` only after inspecting the terminal log and fixing the recorded failure. It preserves the W&B run ID and any retryable checkpoint.

## Handoff

Report the study ID, phase, active run IDs, W&B URLs, metric summary, checkpoint disposition, branches and SHAs, and the next scheduled follow-up. State censored convergence targets explicitly. For three-seed results, report mean, standard deviation, and paired differences without a statistical-significance claim.
