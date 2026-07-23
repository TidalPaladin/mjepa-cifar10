# JEPA research protocol

## Research record

Each committed study YAML must record:

- the question, hypothesis, proposed mechanism, baseline, variants, configs, and seeds;
- W&B entity, project, and group;
- parent, `mjepa`, and `vit` code references;
- the data split, evaluation protocol, resource limits, and promotion rules;
- expected checkpoint size or the documented 3 GiB fallback;
- the rejection condition and planned checkpoint disposition.

The run directory must retain the copied config, `provenance.json`, `metadata.json`, `metrics.jsonl`, `state.json` reference, terminal result, W&B identity, checkpoint state, and smaller backbone until its retention stage.

## Code ownership and Git state

Keep experiment-specific code in `mjepa-cifar10`. Move a primitive into `mjepa` when another dataset or JEPA application could use it without importing CIFAR-10 code.

When `mjepa` changes:

1. Create `codex/research/<study-id>` in both repositories.
2. Develop with the editable sibling checkout.
3. Run each repository's `make check` and `make test-ci` targets.
4. Commit and push the `mjepa` branch first.
5. Pin that full SHA in the parent `pyproject.toml`, update `uv.lock`, and build a frozen parent environment.
6. Verify the imported `mjepa` and `vit` sources match the recorded commits.
7. Commit and push the parent branch.

Never push `main` or `master`, rewrite published history, or open a pull request without a separate request. Refuse a launch when either research repository is dirty, unpushed, on a protected branch, stale relative to `uv.lock`, or installed from a mismatched source.

## Dataset and probes

Use the deterministic CIFAR-10 split implemented in `mjepa_cifar10.data`:

- 45,000 official-training examples for optimization;
- 5,000 official-training examples for validation;
- exactly 500 validation examples per class;
- the official test set only after a candidate is confirmed.

Select 10-shot and 100-shot subsets with seeds 0, 1, and 2. Each subset contains exactly 10 or 100 examples per class, comes only from the 45,000-example training split, and remains disjoint from validation. A subset smaller than the configured batch size must still produce a batch.

The online linear probe applies the student classifier head to teacher features. `forward_teacher` runs under `torch.inference_mode()`, which makes the teacher path pseudo-frozen. An isolated probe backward pass must produce gradients only for the classifier head.

## Pretraining trial policy

Permit at most eight pretraining trials.

1. Run the baseline at seed 0 and up to three seed-0 variants.
2. Promote at most one candidate if it satisfies at least one condition:
   - peak validation probe accuracy improves by at least 0.01;
   - active time to the fixed 95% target improves by at least 15%, while peak accuracy loses no more than 0.005;
   - common-horizon active-time AUC improves by at least 10%, with the same accuracy constraint.
3. Rank qualifying candidates by active-time AUC, peak accuracy, then time to the 95% target.
4. Confirm the chosen candidate and baseline with seeds 1 and 2.
5. If no initial candidate qualifies, spend no more than the four remaining trials on additional seed-0 variants. Do not replicate an unqualified result.

Confirmation requires the three-seed mean to meet the same promotion threshold and at least two paired seeds to move in the required direction. Report mean, sample standard deviation, per-seed paired differences, and censored runs. Three pairs do not support a statistical-significance claim.

An explicitly authorized candidate-only follow-up may reuse one completed
seed-0 baseline instead of launching a new baseline. Commit the exact validation
metric curve under `research/baselines/`, record its source study/run and
SHA-256 in `baseline_reference`, and preserve the original metric names and
active-time clock. The harness must verify the hash, schedule every configured
variant at seed 0, and count only the new candidate jobs against the eight-trial
limit. Use the reference curve to derive targets and common horizons. If a
candidate qualifies, record `reference-promotion` and retain its weights, but do
not call it confirmed, launch supervised evaluation, or report paired effects.
Those steps require separately authorized baseline seeds 1 and 2 and the normal
confirmation rule.

## Convergence metrics

Derive fixed targets at 90% and 95% of the baseline seed-0 peak online-probe validation accuracy. For every run, report:

- optimizer step and cumulative active GPU seconds to each target;
- a censored result when the target is not reached;
- peak and final validation probe accuracy;
- trapezoidal validation-accuracy AUC over the common optimizer-step horizon;
- trapezoidal validation-accuracy AUC over the common active-time horizon.

Do not reset active time after resume. The checkpoint restores the student, predictor, teacher, optimizer, scheduler, completed epoch, optimizer step, W&B run ID, image size, and cumulative active seconds. Checkpoint writes use a temporary file in the run directory followed by atomic replacement.

## Supervised evaluation

After confirmation, run full-data, 10-shot, and 100-shot supervised fine-tuning for the baseline and winner. Pair pretraining, subset, initialization, and training seeds 0, 1, and 2. Report validation convergence and official-test accuracy as three-seed mean and sample standard deviation.

Use W&B namespaces consistently:

- `probe/*` for online-probe metrics;
- `pretrain/*` for self-supervised optimization metrics;
- `sft/*` for supervised fine-tuning metrics;
- `convergence/*` for steps, active time, targets, and AUC;
- `provenance/*` for code, environment, data, command, and weight disposition.

Declare the emitted-data manifest for every W&B operation in the study YAML. A
launch emits `metrics`, `configs`, and `provenance`; a summary emits `metrics`
and `provenance`. Gate each operation independently. Online W&B requires an
explicit entity, `authorized: true`, an explicit matching manifest, and approval
for every class that operation emits. Otherwise that complete operation stays
local-only. Record its requested mode, effective mode, destination, manifest,
approvals, and decision in local provenance before writing externally.

## Monitoring and recovery

Training runs in a detached supervisor. The supervisor owns the training process
group until every child exits. On timeout, cancellation, heartbeat failure, or
another exceptional exit, it terminates and reaps that group before releasing
the GPU lock. It writes `worker.json` heartbeats while active, atomically writes
`terminal.json` on completion, failure, or timeout, and then creates a pending
`notification.json` terminal event. A nonzero child exit also records a concise,
structured cause in terminal state. Notification failure cannot change terminal
status. `monitor` merges terminal and notification files into `state.json` and
launches eligible pending work only after the same launch checks pass.
`monitor --no-launch` is strictly read-only for delegated monitoring.

For supervisor-bound managed runs, the rank-zero trainer atomically replaces
`progress.json` at training start, normal metric intervals, validation start,
and checkpoint boundaries. After the first train-validation cycle has produced
both `checkpoint.pt` and `backbone.safetensors`, it writes the one-shot
`first-cycle.json` milestone. This milestone proves that training, validation,
and recovery checkpointing work together. It is stable and idempotent for the
run attempt.

If a run is marked `retryable`, inspect its terminal log, fix and push the cause, then use `launch --retry-failed`. The retry keeps the W&B ID and checkpoint. Detached workers must remove the inherited `WANDB_SERVICE` token so each job starts its own W&B service instead of using the launcher's short-lived socket.

Run `event-controller --root logs/research --study-id <study-id>` as a
persistent local non-model process for new launches. Direct Unix-socket
delivery is the default, and the CLI discovers the running daemon socket.
Study-scoped delivery prevents a
pending or retrying notification from another study from disarming the active
wake path. Launch and dry-run operations register the exact
managed root with `.mjepa-research-root.json`. Register a pre-existing root once
with `register-root --root logs/research`; this also migrates the legacy marker.
The marker binds its canonical `root_path`, and the notification worker rejects
missing, mismatched, symlinked, repository, home, and broad roots before any
recursive scan. The worker connects directly to the discovered Codex app-server
daemon Unix socket. It validates the terminal event, serializes delivery per
task, starts an idle task or steers its newest in-progress turn with an
expected-turn guard, and records acceptance only after the RPC succeeds. Failed
deliveries use bounded full-jitter exponential backoff and
require explicit `notify --requeue` after the eighth failure or a permanent
validation error. Training never starts or waits for app-server.

The event controller uses Linux inotify for durable source files, pidfds for
supervisor exits, and local deadline timers for trainer progress. It creates a
one-shot `supervisor-lost.json` event when a recorded live supervisor disappears
without terminal state and a one-shot `progress-stalled.json` event when a live
supervisor's trainer-owned progress exceeds the configured deadline. It queues
and delivers first-cycle, safety, and terminal notifications with stable event
identifiers. Routine `progress.json`, heartbeat, notification, acceptance, and
retry writes never wake Codex or retrigger delivery.

If app-server delivery fails, keep queued events durable and disarm further
delivery attempts until the daemon control socket is replaced or a due sparse
recovery check explicitly runs the one-shot worker. Use
`--defer-until-socket-replaced` when starting against a transport already known
to be unavailable. Never hold a Codex turn open to sleep, wait on a shell
process, or poll terminal files. Runs launched before the trainer instrumentation
cannot emit progress or first-cycle events; retain terminal, supervisor, and
sparse recovery coverage for them.

Keep sparse routine monitoring only as a fallback:

- check 10 minutes after launch;
- check again 20 minutes after launch;
- if both checks show normal progress, poll every 30 minutes for the rest of the phase;
- return to 10-minute checks for the first two checks after a retry or a newly launched phase.

Advance the routine-check count only when that run's `next_check_at` is due and the check is performed. A terminal wake for one run must clear only that run's schedule and preserve schedules and budgets for other active runs.

Pin scheduled read-only checks to GPT-5.6 Luna with medium reasoning when model
selection is available; select it once in the scheduled-task configuration
instead of relying on the chat default. App-server `turn/start` can also select
`model: gpt-5.6-luna` and `effort: medium`; the notifier sends those overrides
when starting an idle task. `turn/steer` inherits the active turn's model. Record
the effective model and any fallback. The monitor is limited to `status`,
`monitor --no-launch`, terminal-log inspection, and concise failure reporting.
It must not launch pending work, change code or Git state, call `summarize`,
select a winner, or delete checkpoints. The primary goal agent performs those
state transitions.

Each monitoring pass should:

1. read the active goal;
2. run `status` or `monitor --no-launch` for the study ID;
3. report a terminal phase to the primary goal agent;
4. stop scheduling when the study is complete, not confirmed, or needs user authority.

While producing an already-due monitoring, terminal, promotion, or handoff
report, sample current Codex rate-limit telemetry once when it is available.
Include the UTC observation time, used and remaining percentages, window and
reset time, and the change from the prior reported snapshot when known. This
observation must not advance a routine-check count or change `next_check_at`.
Never create a scheduled task, start or wake a turn, wait, or poll solely for
usage reporting.

The primary goal agent calls `summarize`, commits and pushes result or schedule changes, and launches the next phase.

App-server delivery is at least once and deduplicated by lifecycle-event ID. The wake prompt contains only validated identifiers, status, and the absolute event-state path, never raw logs or stack traces. A host or scheduler failure can still miss a wake. When app-server or the controller is unavailable, the sparse monitor and atomic state let the primary task recover with `status`.

## Storage and retention

Before each launch, require at least:

```text
50 GiB + 2 * concurrent_jobs * estimated_checkpoint_size
```

Estimate checkpoint size from the largest of the 20 most recently modified local checkpoints. Use 3 GiB if none exist. The factor of two reserves room for the existing checkpoint and its atomic replacement.

Retention applies only to new managed runs under `logs/research/<study-id>/runs/<run-id>`:

- rejected run: delete `checkpoint.pt` after metrics, provenance, rejection decision, and result commit are pushed;
- rejected run at study close: also delete `backbone.safetensors`;
- baseline, confirmed winner, or fixed-reference promoted candidate: retain full and backbone weights;
- retryable failure: retain weights until retry or study close.

Before deleting, verify that the run is terminal, the state decision permits deletion, and the resolved path is the exact managed run directory. Log each path and byte count in `retention.jsonl`. Weight deletion is not recoverable. Do not inspect for deletion or alter the existing legacy checkpoint collection unless the user makes a separate cleanup request.

## Completion checks

Before publishing a result:

1. Validate the skill with `quick_validate.py`.
2. Exercise `launch --dry-run` for the study.
3. Run `make check` and `make test-ci` in both repositories.
4. Run the one-epoch W&B-offline GPU smoke study on physical GPU 1 or 2, including progress, first-cycle notification, checkpoint, resume, status, summary, and retention behavior.
5. Confirm the study ID recovers its copied config, local metrics, provenance, state, W&B identity, research-log entry, and retained checkpoint.
6. Commit and push the result update. Do not open a pull request unless requested.
7. If pull-request publication is authorized and the branch contains terminal comparative results, create or refresh a `## Findings` section after the result commit is pushed. Generate its table from `summary.json`; include every evaluated variant, key optimizer hyperparameters, peak and final validation accuracy, convergence metrics, per-run wall time, decision, total study span, and summed run time. Mark censored values and distinguish active from wall time and nominal from effective Muon learning rates. Omit the section for smoke-only, protocol-only, and active studies.
