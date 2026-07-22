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
<!-- autoresearch-operation:{"content_sha256":"09f3f74fe119e0b63a377994a290f17d76948cb2ba134f9cc41a2ba6e17c63b5","operation_id":"dacda8260add93832e3333cf942839e4"} -->

<!-- study:muon-optimizer-v1:phase:exploration -->
## muon-optimizer-v1

- Question: Does hybrid Muon improve ViT-S/4 MJEPA pretraining convergence or online-probe validation accuracy relative to AdamW on the fixed CIFAR-10 split?
- Hypothesis: A Muon configuration will meet a preregistered promotion threshold and rank above the AdamW configurations by common-horizon active-time AUC, peak validation accuracy, then active time to the fixed 95% target.
- Mechanisms and exact changes:
  - `adamw-baseline`: Mechanism: Apply AdamW to all trainable student, predictor, and classifier-head parameters. Changes: not recorded.
  - `muon-matched`: Mechanism: Apply Newton-Schulz-orthogonalized momentum updates to 2-D hidden weights, with AdamW-RMS learning-rate adjustment; retain AdamW for the probe head, token-like parameters, and non-2-D parameters. Changes: Replace AdamW with hybrid Muon for eligible 2-D hidden weights.; Use match_rms_adamw scaling with Muon learning rate 0.002, momentum 0.95, and weight decay 0.2.; Keep the auxiliary AdamW learning rate at 0.002.
  - `muon-lr-half`: Mechanism: Use the same hybrid routing and RMS adjustment as muon-matched while reducing only the Muon branch learning rate. Changes: Set the Muon learning rate to 0.001.; Keep the auxiliary AdamW learning rate at 0.002 and weight decay at 0.2.
  - `adamw-lr-half`: Mechanism: Preserve the formal AdamW optimizer and reduce its global learning rate. Changes: Set the AdamW learning rate to 0.001.; Keep AdamW weight decay at 0.2 and betas at [0.85, 0.95].
  - `muon-wd-half`: Mechanism: Preserve matched-rate hybrid Muon and reduce optimizer weight decay. Changes: Set the Muon and auxiliary AdamW weight decay to 0.1.; Keep the Muon and auxiliary AdamW learning rates at 0.002.
  - `adamw-wd-half`: Mechanism: Preserve the formal AdamW optimizer and reduce its global weight decay. Changes: Set the AdamW weight decay to 0.1.; Keep the AdamW learning rate at 0.002.
  - `muon-lr-wd-half`: Mechanism: Use hybrid Muon with the lower preregistered learning rate and weight decay. Changes: Set the Muon learning rate to 0.001 and optimizer weight decay to 0.1.; Keep the auxiliary AdamW learning rate at 0.002.
  - `adamw-lr-wd-half`: Mechanism: Preserve AdamW while using the lower preregistered learning rate and weight decay. Changes: Set the AdamW learning rate to 0.001 and weight decay to 0.1.
- Launch code provenance:
  - `pretrain-adamw-baseline-seed0`: parent=`0e4cb08d0e765af18029186eede61d0bd0bab2cc` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-adamw-lr-half-seed0`: parent=`b7eeb4ad40b546cd7a089c949bc0d76a2c47760c` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-lr-half-seed0`: parent=`b7eeb4ad40b546cd7a089c949bc0d76a2c47760c` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-matched-seed0`: parent=`0e4cb08d0e765af18029186eede61d0bd0bab2cc` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
- Phase: exploration
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/summary.json`; external_detail=True
- Conclusion: No initial seed-0 candidate met a promotion threshold; bounded seed-0 exploration is still running.
- Follow-up: run the preregistered exploration trials.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-adamw-baseline-seed0`: attempt=2; status=completed; decision=baseline; started=2026-07-22T01:55:03.116159+00:00; finished=2026-07-22T04:59:39.527157+00:00; terminal_event=fe452302-8a7a-405b-b987-02e7af197fb1; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-adamw-baseline-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/deccdb23); checkpoint=retained; metrics=peak_accuracy=0.905800, final_accuracy=0.900000, step_to_90=6090, step_to_95=8265, active_seconds_to_90=3874.688, active_seconds_to_95=5255.814, step_auc=0.795948, active_time_auc=0.795831; error=none
- `pretrain-adamw-lr-half-seed0`: attempt=2; status=completed; decision=rejected; started=2026-07-22T05:05:56.197964+00:00; finished=2026-07-22T08:11:48.874266+00:00; terminal_event=c5ca5c2d-ed96-4720-98dc-9b505096607a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-adamw-lr-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/7d02118f); checkpoint=retained; metrics=peak_accuracy=0.891000, final_accuracy=0.889000, step_to_90=7830, step_to_95=11310, active_seconds_to_90=5012.789, active_seconds_to_95=7239.140, step_auc=0.759115, active_time_auc=0.758020; error=none
- `pretrain-adamw-lr-wd-half-seed0`: attempt=1; status=pending; decision=pending; started=unknown; finished=unknown; terminal_event=unknown; artifacts=`unavailable`; W&B=unavailable; checkpoint=retained; metrics=unavailable; error=none
- `pretrain-adamw-wd-half-seed0`: attempt=1; status=pending; decision=pending; started=unknown; finished=unknown; terminal_event=unknown; artifacts=`unavailable`; W&B=unavailable; checkpoint=retained; metrics=unavailable; error=none
- `pretrain-muon-lr-half-seed0`: attempt=2; status=completed; decision=rejected; started=2026-07-22T04:59:43.087824+00:00; finished=2026-07-22T08:09:41.711690+00:00; terminal_event=5ffd7b9e-4507-4dff-90cf-b79da5333b2d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-muon-lr-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c066da97); checkpoint=retained; metrics=peak_accuracy=0.860600, final_accuracy=0.859000, step_to_90=7395, step_to_95=16965, active_seconds_to_90=4839.683, active_seconds_to_95=11100.181, step_auc=0.773257, active_time_auc=0.770529; error=none
- `pretrain-muon-lr-wd-half-seed0`: attempt=1; status=pending; decision=pending; started=unknown; finished=unknown; terminal_event=unknown; artifacts=`unavailable`; W&B=unavailable; checkpoint=retained; metrics=unavailable; error=none
- `pretrain-muon-matched-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T01:53:54.219194+00:00; finished=2026-07-22T05:05:46.873716+00:00; terminal_event=2895f963-0ce2-4443-9a9d-d17338f51b14; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-muon-matched-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/8f64f96c); checkpoint=retained; metrics=peak_accuracy=0.869600, final_accuracy=0.865400, step_to_90=6090, step_to_95=14790, active_seconds_to_90=4029.018, active_seconds_to_95=9774.713, step_auc=0.786880, active_time_auc=0.783454; error=none
- `pretrain-muon-wd-half-seed0`: attempt=1; status=pending; decision=pending; started=unknown; finished=unknown; terminal_event=unknown; artifacts=`unavailable`; W&B=unavailable; checkpoint=retained; metrics=unavailable; error=none
<!-- autoresearch-operation:{"content_sha256":"66f8e7cf9a698933c3257e80d4c65d06eb38c88958ab25fc2381abfd1bf85d38","operation_id":"3e92bf8710fac868bd488ccd1c5151f1"} -->

<!-- study:muon-optimizer-v1:phase:no-promotion -->
## muon-optimizer-v1

- Question: Does hybrid Muon improve ViT-S/4 MJEPA pretraining convergence or online-probe validation accuracy relative to AdamW on the fixed CIFAR-10 split?
- Hypothesis: A Muon configuration will meet a preregistered promotion threshold and rank above the AdamW configurations by common-horizon active-time AUC, peak validation accuracy, then active time to the fixed 95% target.
- Mechanisms and exact changes:
  - `adamw-baseline`: Mechanism: Apply AdamW to all trainable student, predictor, and classifier-head parameters. Changes: not recorded.
  - `muon-matched`: Mechanism: Apply Newton-Schulz-orthogonalized momentum updates to 2-D hidden weights, with AdamW-RMS learning-rate adjustment; retain AdamW for the probe head, token-like parameters, and non-2-D parameters. Changes: Replace AdamW with hybrid Muon for eligible 2-D hidden weights.; Use match_rms_adamw scaling with Muon learning rate 0.002, momentum 0.95, and weight decay 0.2.; Keep the auxiliary AdamW learning rate at 0.002.
  - `muon-lr-half`: Mechanism: Use the same hybrid routing and RMS adjustment as muon-matched while reducing only the Muon branch learning rate. Changes: Set the Muon learning rate to 0.001.; Keep the auxiliary AdamW learning rate at 0.002 and weight decay at 0.2.
  - `adamw-lr-half`: Mechanism: Preserve the formal AdamW optimizer and reduce its global learning rate. Changes: Set the AdamW learning rate to 0.001.; Keep AdamW weight decay at 0.2 and betas at [0.85, 0.95].
  - `muon-wd-half`: Mechanism: Preserve matched-rate hybrid Muon and reduce optimizer weight decay. Changes: Set the Muon and auxiliary AdamW weight decay to 0.1.; Keep the Muon and auxiliary AdamW learning rates at 0.002.
  - `adamw-wd-half`: Mechanism: Preserve the formal AdamW optimizer and reduce its global weight decay. Changes: Set the AdamW weight decay to 0.1.; Keep the AdamW learning rate at 0.002.
  - `muon-lr-wd-half`: Mechanism: Use hybrid Muon with the lower preregistered learning rate and weight decay. Changes: Set the Muon learning rate to 0.001 and optimizer weight decay to 0.1.; Keep the auxiliary AdamW learning rate at 0.002.
  - `adamw-lr-wd-half`: Mechanism: Preserve AdamW while using the lower preregistered learning rate and weight decay. Changes: Set the AdamW learning rate to 0.001 and weight decay to 0.1.
- Launch code provenance:
  - `pretrain-adamw-baseline-seed0`: parent=`0e4cb08d0e765af18029186eede61d0bd0bab2cc` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-adamw-lr-half-seed0`: parent=`b7eeb4ad40b546cd7a089c949bc0d76a2c47760c` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-adamw-lr-wd-half-seed0`: parent=`6d6d8b0ab5ff71a820563d500a4c7b02c01166ec` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-adamw-wd-half-seed0`: parent=`6d6d8b0ab5ff71a820563d500a4c7b02c01166ec` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-lr-half-seed0`: parent=`b7eeb4ad40b546cd7a089c949bc0d76a2c47760c` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-lr-wd-half-seed0`: parent=`6d6d8b0ab5ff71a820563d500a4c7b02c01166ec` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-matched-seed0`: parent=`0e4cb08d0e765af18029186eede61d0bd0bab2cc` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-wd-half-seed0`: parent=`6d6d8b0ab5ff71a820563d500a4c7b02c01166ec` (`codex/research/muon-optimizer-v1`), mjepa=`04b33f8e938ab5fea6d863a7871a57aee05e94c7` (`codex/research/vit-small-baseline-v1`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-adamw-baseline-seed0`: attempt=2; status=completed; decision=baseline; started=2026-07-22T01:55:03.116159+00:00; finished=2026-07-22T04:59:39.527157+00:00; terminal_event=fe452302-8a7a-405b-b987-02e7af197fb1; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-adamw-baseline-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/deccdb23); checkpoint=retained; metrics=peak_accuracy=0.905800, final_accuracy=0.900000, step_to_90=6090, step_to_95=8265, active_seconds_to_90=3874.688, active_seconds_to_95=5255.814, step_auc=0.795948, active_time_auc=0.795831; error=none
- `pretrain-adamw-lr-half-seed0`: attempt=2; status=completed; decision=rejected; started=2026-07-22T05:05:56.197964+00:00; finished=2026-07-22T08:11:48.874266+00:00; terminal_event=c5ca5c2d-ed96-4720-98dc-9b505096607a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-adamw-lr-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/7d02118f); checkpoint=deleted-not-recoverable; metrics=peak_accuracy=0.891000, final_accuracy=0.889000, step_to_90=7830, step_to_95=11310, active_seconds_to_90=5012.789, active_seconds_to_95=7239.140, step_auc=0.759115, active_time_auc=0.758020; error=none
- `pretrain-adamw-lr-wd-half-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T11:36:52.197817+00:00; finished=2026-07-22T14:42:41.103998+00:00; terminal_event=4ebfe3ab-5f0c-49a8-a72c-bf5643a0b3a3; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-adamw-lr-wd-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/7fbe0299); checkpoint=retained; metrics=peak_accuracy=0.885400, final_accuracy=0.884000, step_to_90=8265, step_to_95=11310, active_seconds_to_90=5288.659, active_seconds_to_95=7235.340, step_auc=0.759802, active_time_auc=0.758810; error=none
- `pretrain-adamw-wd-half-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T08:23:12.402424+00:00; finished=2026-07-22T11:28:03.572249+00:00; terminal_event=7ba8e6a0-8099-4ad2-863e-a98501ed2cc0; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-adamw-wd-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/dbfde7b9); checkpoint=retained; metrics=peak_accuracy=0.902000, final_accuracy=0.891000, step_to_90=6525, step_to_95=8700, active_seconds_to_90=4150.806, active_seconds_to_95=5534.018, step_auc=0.788175, active_time_auc=0.788027; error=none
- `pretrain-muon-lr-half-seed0`: attempt=2; status=completed; decision=rejected; started=2026-07-22T04:59:43.087824+00:00; finished=2026-07-22T08:09:41.711690+00:00; terminal_event=5ffd7b9e-4507-4dff-90cf-b79da5333b2d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-muon-lr-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c066da97); checkpoint=deleted-not-recoverable; metrics=peak_accuracy=0.860600, final_accuracy=0.859000, step_to_90=7395, step_to_95=16965, active_seconds_to_90=4839.683, active_seconds_to_95=11100.181, step_auc=0.773257, active_time_auc=0.770529; error=none
- `pretrain-muon-lr-wd-half-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T11:28:08.540709+00:00; finished=2026-07-22T14:38:34.709819+00:00; terminal_event=b0791876-240d-49c9-a785-e1b9143d9618; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-muon-lr-wd-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/e7b8523c); checkpoint=retained; metrics=peak_accuracy=0.852400, final_accuracy=0.844600, step_to_90=8265, step_to_95=censored, active_seconds_to_90=5422.712, active_seconds_to_95=censored, step_auc=0.768155, active_time_auc=0.765400; error=none
- `pretrain-muon-matched-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T01:53:54.219194+00:00; finished=2026-07-22T05:05:46.873716+00:00; terminal_event=2895f963-0ce2-4443-9a9d-d17338f51b14; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-muon-matched-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/8f64f96c); checkpoint=deleted-not-recoverable; metrics=peak_accuracy=0.869600, final_accuracy=0.865400, step_to_90=6090, step_to_95=14790, active_seconds_to_90=4029.018, active_seconds_to_95=9774.713, step_auc=0.786880, active_time_auc=0.783454; error=none
- `pretrain-muon-wd-half-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T08:23:12.343155+00:00; finished=2026-07-22T11:36:47.485089+00:00; terminal_event=e9eb8e6b-5702-44fa-b587-dbebd46b65d0; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v1/runs/pretrain-muon-wd-half-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/23faccc1); checkpoint=retained; metrics=peak_accuracy=0.879800, final_accuracy=0.879600, step_to_90=6090, step_to_95=9570, active_seconds_to_90=4033.613, active_seconds_to_95=6337.274, step_auc=0.792865, active_time_auc=0.789115; error=none
<!-- autoresearch-operation:{"content_sha256":"55d714f6533e2a57cb5ebe307a96c235b052abfa94f7982103b219767985f63a","operation_id":"muon-optimizer-v1-final-closeout-v1"} -->

## muon-optimizer-v1 final closeout

- Scope: This result applies to hybrid Muon with PyTorch `match_rms_adamw` adjustment at nominal Muon learning rates 0.001 and 0.002. It does not test the original Muon adjustment at a raw learning rate near 0.02 or a higher matched-RMS rate.
- Learning-rate verification: For the exact ViT-S/4 model, `match_rms_adamw` scales the nominal 0.002 Muon rate to 0.007838-0.022170 across the five eligible matrix shapes; nominal 0.001 scales to 0.003919-0.011085. The study therefore did exercise order-of-magnitude Muon matrix-update scales while keeping the auxiliary AdamW branch at 0.002.
- Result: The AdamW baseline remained best (peak validation accuracy 0.905800; common-horizon active-time AUC 0.795831). The strongest Muon variant was `muon-wd-half` (peak 0.879800; active-time AUC 0.789115), so no candidate met a preregistered promotion threshold.
- Gate resolution: The study ended at `no-promotion`. Replication of an unqualified candidate was forbidden by the preregistered rule, and no winner existed for supervised fine-tuning evaluation; neither stage was launched.
- External telemetry: The final post-retention summary was published to the authorized W&B destination `tidalpaladin/mjepa-cifar10` with metrics and provenance only; no publish errors were recorded.
- Retention: The formal baseline full checkpoint and backbone are retained. All seven rejected variants are `deleted-not-recoverable`; the managed retention ledger records 14 weight files removed across 10 atomic operations, freeing 4,183,031,527 bytes.
- Next falsifiable hypothesis: A separately authorized bounded follow-up can compare original-scaling Muon at raw rates around 0.01-0.02, or matched-RMS Muon above nominal 0.002, against the retained AdamW baseline.
