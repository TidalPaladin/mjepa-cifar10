# JEPA Research Log

Append one entry per completed or terminated study. Record the hypothesis, mechanism, exact code changes and SHAs, W&B URLs, convergence and downstream metrics, conclusion, follow-up, and checkpoint disposition. Do not rewrite historical entries.

<!-- study:harness-smoke:phase:no-promotion -->
## harness-smoke

- Question: Can a managed one-epoch baseline recover its config, metrics, provenance, checkpoint, resume state, and summary from the study ID?
- Hypothesis: The research harness will complete one offline baseline run and recover every required artifact after process exit.
- Mechanisms and exact changes:
  - `baseline`: Mechanism: Run the standard JEPA path for one epoch through the managed supervisor. Changes: Use the committed small smoke model and W&B offline mode.
- Launch code provenance:
  - `pretrain-baseline-seed0`: parent=`6b1dc433ac45d40bc95fd3e7fce434646fa3bd22` (`codex/research/harness-smoke`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/harness-smoke`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
- Phase: no-promotion
- Winner: none
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis after metric review.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-baseline-seed0`: status=completed; decision=baseline; W&B=offline/unlinked (`6a727eca`); checkpoint=retained; metrics=peak_accuracy=0.161200, final_accuracy=0.161200, step_to_90=87, step_to_95=87, active_seconds_to_90=50.932, active_seconds_to_95=50.932, step_auc=0.161200, active_time_auc=0.161200

<!-- study:vit-small-baseline-v1:phase:no-promotion -->
## vit-small-baseline-v1

- Question: What convergence and online-probe performance does the current ViT-S/4 MJEPA configuration achieve on the fixed CIFAR-10 validation split?
- Hypothesis: The current configuration will complete 400 epochs within 24 active GPU hours and produce a valid online-probe trajectory from which fixed 90% and 95% convergence targets can be derived.
- Mechanisms and exact changes:
  - `baseline`: Mechanism: Predict teacher targets from masked student context with Gram anchoring and SigReg enabled by the current configuration. Changes: none beyond the committed baseline configuration.
- Launch code provenance:
  - `pretrain-baseline-seed0`: parent=`de591f950ec9ff1163c1cafe32849fb3dee07bd3` (`codex/research/vit-small-baseline-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
- Phase: no-promotion
- Winner: none
- Conclusion: The 400-epoch baseline completed successfully with a 90.58% peak validation probe accuracy and 90.00% final accuracy; no candidates were configured for promotion.
- Follow-up: use the baseline peak to define fixed convergence targets for any future ablation, then launch only a falsifiable seed-0 variant under the eight-trial limit.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-baseline-seed0`: status=completed; decision=baseline; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/29291984); checkpoint=retained; metrics=peak_accuracy=0.905800, final_accuracy=0.900000, step_to_90=6090, step_to_95=8265, active_seconds_to_90=3922.749, active_seconds_to_95=5307.830, step_auc=0.795948, active_time_auc=0.794192
<!-- autoresearch-operation:{"content_sha256":"394262ddee9a63b561332f18d9f8ff0c3b27c4dd59107c4bb567934ed2b8e69d","operation_id":"22d88455298fc953e230c7f01d3f7859"} -->

<!-- study:muon-optimizer-v1-smoke:phase:no-promotion -->
## muon-optimizer-v1-smoke

- Question: Can the managed harness train, checkpoint, recover, and summarize one hybrid Muon epoch on GPU?
- Hypothesis: The hybrid Muon smoke run will complete with valid local metrics and recoverable optimizer and scheduler state.
- Mechanisms and exact changes:
  - `muon-smoke`: Mechanism: Route eligible 2-D hidden weights to Muon and all other trainable parameters to AdamW. Changes: Use the small one-epoch smoke model with hybrid Muon and local-only W&B.
- Launch code provenance:
  - `pretrain-muon-smoke-seed0`: parent=`7b23e21c0d4cd722fa313699b57bd4dea7838648` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=False; approved_data_classes=none
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1-smoke/summary.json`; external_detail=False
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis after metric review.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-muon-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-22T01:41:04.663319+00:00; finished=2026-07-22T01:41:29.698237+00:00; terminal_event=a8b04c98-b7d4-4f83-8581-f6f04b8d39be; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1-smoke/runs/pretrain-muon-smoke-seed0`; W&B=offline/unlinked (`2d1ab94f`); checkpoint=retained; metrics=peak_accuracy=0.100000, final_accuracy=0.100000, step_to_90=87, step_to_95=87, active_seconds_to_90=14.283, active_seconds_to_95=14.283, step_auc=0.100000, active_time_auc=0.100000; error=none
