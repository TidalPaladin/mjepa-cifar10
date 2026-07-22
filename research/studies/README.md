# Study specifications

Copy `example.yaml` to a study-specific filename. Replace every placeholder before launch. Keep the question and hypothesis falsifiable, record the full code references, and use at most eight pretraining trials.

Set `CIFAR10_DATA` in the launch environment so the committed specification does not expose a machine-specific dataset path.

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
deliver due events through an existing local Codex app-server daemon. Launch
registers the exact root with `.mjepa-research-root.json`. Use `register-root
--root logs/research` for a pre-existing root; the command is idempotent and
migrates the legacy marker. Sweeps validate the marker's exact canonical root
before scanning. Use
`notify <study> <run> --requeue` only to reset a permanently failed delivery.
