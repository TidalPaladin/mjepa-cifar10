# Study specifications

Copy `example.yaml` to a study-specific filename. Replace every placeholder before launch. Keep the question and hypothesis falsifiable, record the full code references, and use at most eight pretraining trials per managed study. A larger user-authorized goal budget must be allocated across linked study specifications and recorded in the repository protocol.

Set `CIFAR10_DATA` in the launch environment so the committed specification does not expose a machine-specific dataset path.

Use the normal inline `baseline` run for studies that can perform paired
confirmation. For an explicitly candidate-only follow-up, add a
`baseline_reference` mapping with `study_id`, `run_id`, `metrics`, and
`metrics_sha256`. Commit the exact validation curve under `research/baselines/`;
the harness verifies its SHA-256, does not schedule the baseline, and runs every
configured variant at seed 0. A qualifying result ends in
`reference-promotion`, which is provisional because no paired baseline seeds
were run. Do not use it to claim confirmation or start supervised evaluation.

External W&B publication is opt-in and gated per operation. Keep the explicit
`wandb.emitted_data_classes` manifests from `example.yaml`: launch emits
`metrics`, `configs`, and `provenance`, while summary emits `metrics` and
`provenance`. When a study includes a W&B entity, set `wandb.authorized: true`
and approve every class needed by the operation. Leave it unauthorized for
local-only tracking. Local provenance records the requested and effective modes
before any external write.

`monitor --no-launch` is strictly read-only and is suitable for a delegated
monitor. The primary coordinator uses `monitor` to reconcile terminal state and
schedule the next eligible jobs. Terminal workers create a pending
`notification.json` after durable terminal state. Use `notify-worker --once` to
deliver due events through an existing local Codex app-server daemon, and pass
`--study-id <study-id>` to isolate the active study from unrelated historical
delivery failures. Direct Unix-socket delivery is the default; it discovers
and connects to the running daemon socket. Launch
registers the exact root with `.mjepa-research-root.json`. Use `register-root
--root logs/research` for a pre-existing root; the command is idempotent and
migrates the legacy marker. Sweeps validate the marker's exact canonical root
before scanning. Use
`notify <study> <run> --requeue` only to reset a permanently failed delivery.
If the originating goal is blocked by a prior bounded wait, delivery resumes it
through `thread/goal/set` before sending the lifecycle prompt. User-paused and
terminal goal states are never overridden.
