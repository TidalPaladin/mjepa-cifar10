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
<!-- autoresearch-operation:{"content_sha256":"dcdac5749536918727cc1f52908c0ae395e4b2711a665b88f0718f1d14d33e7f","operation_id":"8b29ae6379a75f87623cf6baca7b22a3"} -->

<!-- study:muon-optimizer-v2:phase:no-promotion -->
## muon-optimizer-v2

- Question: Can a focused sweep of correctly routed Muon learning-rate scaling and weight decay beat the fixed AdamW seed-0 baseline on ViT-S/4 MJEPA pretraining?
- Hypothesis: At least one higher-rate Muon configuration will meet a preregistered accuracy or convergence promotion threshold against the immutable AdamW baseline curve from muon-optimizer-v1.
- Mechanisms and exact changes:
  - `adamw-fixed-v1`: Mechanism: Reuse the committed validation curve from the completed formal baseline to preserve its exact 90% and 95% convergence targets. Changes: not recorded.
  - `muon-match-lr3e3-wd1e1`: Mechanism: Apply Moonshot-style RMS matching to hidden 2-D matrices while increasing only the Muon branch rate above the best prior setting. Changes: Set nominal Muon learning rate to 0.003 with match_rms_adamw scaling.; Set Muon weight decay to 0.1 while holding auxiliary AdamW at learning rate 0.002 and weight decay 0.2.
  - `muon-original-lr1e2-wd1e1`: Mechanism: Use Keller Jordan's original per-matrix adjustment with a raw 0.01 Muon rate and unchanged auxiliary AdamW controls. Changes: Set nominal Muon learning rate to 0.01 with original scaling.; Set Muon weight decay to 0.1 while holding auxiliary AdamW at learning rate 0.002 and weight decay 0.2.
  - `muon-match-lr4e3-wd1e1`: Mechanism: Double the prior matched-RMS rate on hidden matrices without changing routing, momentum, Newton-Schulz iterations, or auxiliary AdamW. Changes: Set nominal Muon learning rate to 0.004 with match_rms_adamw scaling.; Set Muon weight decay to 0.1 while holding auxiliary AdamW at learning rate 0.002 and weight decay 0.2.
  - `muon-original-lr2e2-wd1e1`: Mechanism: Use the original Muon adjustment and its commonly documented raw rate while preserving all non-Muon controls. Changes: Set nominal Muon learning rate to 0.02 with original scaling.; Set Muon weight decay to 0.1 while holding auxiliary AdamW at learning rate 0.002 and weight decay 0.2.
  - `muon-match-lr3e3-wd5e2`: Mechanism: Reduce only hidden-matrix Muon weight decay, leaving the probe and other auxiliary AdamW parameters at the formal baseline decay. Changes: Set nominal Muon learning rate to 0.003 with match_rms_adamw scaling and Muon weight decay to 0.05.; Hold auxiliary AdamW at learning rate 0.002 and weight decay 0.2.
  - `muon-original-lr1e2-wd5e2`: Mechanism: Combine the less aggressive original rate with reduced hidden-matrix decay while preserving auxiliary AdamW controls. Changes: Set nominal Muon learning rate to 0.01 with original scaling and Muon weight decay to 0.05.; Hold auxiliary AdamW at learning rate 0.002 and weight decay 0.2.
- Launch code provenance:
  - `pretrain-muon-match-lr3e3-wd1e1-seed0`: parent=`8ec36b11574d726dcb365eddd1b329a30bb92c85` (`codex/research/muon-optimizer-v2`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-match-lr3e3-wd5e2-seed0`: parent=`7179deed244852e84b8982180760c895a9126447` (`codex/research/muon-optimizer-v2`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-match-lr4e3-wd1e1-seed0`: parent=`7179deed244852e84b8982180760c895a9126447` (`codex/research/muon-optimizer-v2`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-original-lr1e2-wd1e1-seed0`: parent=`8ec36b11574d726dcb365eddd1b329a30bb92c85` (`codex/research/muon-optimizer-v2`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-original-lr1e2-wd5e2-seed0`: parent=`7179deed244852e84b8982180760c895a9126447` (`codex/research/muon-optimizer-v2`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
  - `pretrain-muon-original-lr2e2-wd1e1-seed0`: parent=`7179deed244852e84b8982180760c895a9126447` (`codex/research/muon-optimizer-v2`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`2723d319cdacb0462956bd07cb526683183f625c` (`master`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-muon-match-lr3e3-wd1e1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T16:08:35.381219+00:00; finished=2026-07-22T19:20:28.316252+00:00; terminal_event=e4359c1d-6261-4794-9f75-157b766609ac; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/runs/pretrain-muon-match-lr3e3-wd1e1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/bb370825); checkpoint=retained; metrics=peak_accuracy=0.878000, final_accuracy=0.878000, step_to_90=6525, step_to_95=9570, active_seconds_to_90=4317.355, active_seconds_to_95=6326.194, step_auc=0.795288, active_time_auc=0.791830; error=none
- `pretrain-muon-match-lr3e3-wd5e2-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-23T01:15:08.721803+00:00; finished=2026-07-23T04:27:24.399329+00:00; terminal_event=3ffbb39f-5418-4fdd-9b58-bcff6c21a743; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/runs/pretrain-muon-match-lr3e3-wd5e2-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/5f7d8425); checkpoint=retained; metrics=peak_accuracy=0.841600, final_accuracy=0.837000, step_to_90=6090, step_to_95=censored, active_seconds_to_90=4036.431, active_seconds_to_95=censored, step_auc=0.779555, active_time_auc=0.777080; error=none
- `pretrain-muon-match-lr4e3-wd1e1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T19:51:05.613389+00:00; finished=2026-07-22T23:03:49.607606+00:00; terminal_event=61add10f-37bc-4bae-85c2-265333de2a69; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/runs/pretrain-muon-match-lr4e3-wd1e1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/b03bfb3c); checkpoint=retained; metrics=peak_accuracy=0.904800, final_accuracy=0.901200, step_to_90=5220, step_to_95=7395, active_seconds_to_90=3452.352, active_seconds_to_95=4889.339, step_auc=0.819675, active_time_auc=0.816248; error=none
- `pretrain-muon-original-lr1e2-wd1e1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T16:08:35.437671+00:00; finished=2026-07-22T19:18:34.319659+00:00; terminal_event=f5f5fba0-a9dd-4f5d-aec0-185c5dabb1cf; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/runs/pretrain-muon-original-lr1e2-wd1e1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/8f0284c6); checkpoint=retained; metrics=peak_accuracy=0.894000, final_accuracy=0.894000, step_to_90=6090, step_to_95=8700, active_seconds_to_90=3987.879, active_seconds_to_95=5694.501, step_auc=0.802528, active_time_auc=0.799690; error=none
- `pretrain-muon-original-lr1e2-wd5e2-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-23T01:15:08.783230+00:00; finished=2026-07-23T04:25:40.723823+00:00; terminal_event=1d1d05af-2490-4160-8add-5bd3f7a16eed; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/runs/pretrain-muon-original-lr1e2-wd5e2-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/2a929768); checkpoint=retained; metrics=peak_accuracy=0.896200, final_accuracy=0.892400, step_to_90=6090, step_to_95=7395, active_seconds_to_90=3997.947, active_seconds_to_95=4854.229, step_auc=0.806077, active_time_auc=0.803208; error=none
- `pretrain-muon-original-lr2e2-wd1e1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-22T19:51:05.670987+00:00; finished=2026-07-22T23:01:36.745914+00:00; terminal_event=410157de-0a78-4163-8fe6-c34058d30e4a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/muon-optimizer-v2/runs/pretrain-muon-original-lr2e2-wd1e1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c9716b07); checkpoint=retained; metrics=peak_accuracy=0.891000, final_accuracy=0.889800, step_to_90=5655, step_to_95=8700, active_seconds_to_90=3712.623, active_seconds_to_95=5709.995, step_auc=0.804285, active_time_auc=0.801428; error=none

<!-- autoresearch-operation:{"content_sha256":"e015e21d1efa3cb539fdd3058c8e4d87a6ea6bfa82baacba658d289f109bef40","operation_id":"muon-optimizer-v2-final-closeout-v1"} -->

## muon-optimizer-v2 final closeout

- Scope: Six seed-0 Muon candidates were compared with the immutable AdamW seed-0 reference curve. Muon was limited to eligible internal 2-D weights; the auxiliary AdamW branch remained at learning rate 0.002 and weight decay 0.2.
- Learning-rate interpretation: Matched-RMS nominal rates 0.003 and 0.004 produced shape-adjusted Muon rates of approximately 0.011757-0.033255 and 0.015676-0.044340. Original-scaling nominal rates 0.01 and 0.02 produced shape-adjusted rates of approximately 0.010000-0.028284 and 0.020000-0.056569.
- Best result: `muon-match-lr4e3-wd1e1` nearly matched the AdamW peak (0.904800 versus 0.905800), improved final accuracy (0.901200 versus 0.900000), reached the fixed 95% target in 4,889.339 active seconds versus 5,255.814 (6.97% faster), and improved common-horizon active-time AUC from 0.795831 to 0.816248 (2.57%).
- Gate resolution: No candidate met the preregistered 15% convergence or 10% AUC threshold, and none improved peak accuracy by 0.01. The study ended at `no-promotion`; fixed-reference promotion, replication, and supervised fine-tuning were not authorized by the observed result.
- Cost: The six candidates spanned 44,329 seconds (12:18:49) of wall time and summed to 68,875 run-seconds (19:07:55) across two GPUs.
- External telemetry: The final post-retention summary was published to the authorized W&B destination `tidalpaladin/mjepa-cifar10` with metrics and provenance only; no publish errors were recorded.
- Retention: All six rejected full checkpoints and backbones are `deleted-not-recoverable`. The managed retention ledger records 12 weight files removed across 12 completed deletion operations, freeing 3,195,562,290 bytes. The immutable AdamW metric reference and all legacy artifacts remain untouched.
- Lifecycle: All six terminal notifications were accepted. The final event also verified automatic blocked-goal reactivation and direct app-server wake delivery.
<!-- autoresearch-operation:{"content_sha256":"111438c84a34b25112256c57ce109a7e66bb38ca881f67f29b165edbee39b23e","operation_id":"dbda69f0b75ef540e524097ddb2cf581"} -->

<!-- study:srelu-mlp-v1-smoke:phase:no-promotion -->
## srelu-mlp-v1-smoke

- Question: Can the pinned SReLU study environment train, validate, checkpoint, resume, summarize, and notify through one managed GPU epoch?
- Hypothesis: The one-epoch SReLU smoke run will complete with valid local metrics, isolated MLP dropout, recoverable optimizer state, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `srelu-smoke`: Mechanism: Exercise the same activation and configuration path used by the formal ablations at smoke scale. Changes: Replace SwiGLU with Squared ReLU.; Set the explicit MLP-only dropout override to 0.0 while retaining the smoke attention dropout.
- Launch code provenance:
  - `pretrain-srelu-smoke-seed0`: parent=`698b1fac94c38203172308c86e669ed26e0138e4` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=False; approved_data_classes=none
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-v1-smoke/summary.json`; external_detail=False
- Conclusion: Mechanical validation passed. The one-epoch blinded AdaLN run completed with finite gradients, a positive shuffled-minus-true CLS auxiliary loss gap of 0.476268, a 0.788000 ms isolated-path median, accepted lifecycle notifications, and a checkpoint that restored successfully in an isolated local resume run.
- Follow-up: Proceed to the preregistered four-run seed-0 screening without changing variants or promotion thresholds.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-srelu-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-23T16:15:51.664519+00:00; finished=2026-07-23T16:18:40.794367+00:00; terminal_event=d2dcd34f-e07f-440a-ab3e-78a1473c9534; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-v1-smoke/runs/pretrain-srelu-smoke-seed0`; W&B=offline/unlinked (`6c883963`); checkpoint=retained; metrics=peak_accuracy=0.214000, final_accuracy=0.214000, step_to_90=2812, step_to_95=2812, active_seconds_to_90=157.743, active_seconds_to_95=157.743, step_auc=0.214000, active_time_auc=0.214000; error=none
<!-- autoresearch-operation:{"content_sha256":"1b637dcec023e93f998b82135a7d6304300c2ab7cf6b818cf19b984f86e26cdd","operation_id":"a991918b771dda6e3f73583307fd106c"} -->

<!-- study:srelu-mlp-baseline-v1:phase:no-promotion -->
## srelu-mlp-baseline-v1

- Question: What validation-probe convergence, endpoint quality, and wall-clock cost does the current SwiGLU ViT-S/4 MJEPA baseline achieve in the SReLU study environment?
- Hypothesis: The fresh SwiGLU seed-0 run will complete within 24 hours and yield a recoverable curve suitable as the immutable comparator for adaptive SReLU waves.
- Mechanisms and exact changes:
  - `swiglu-baseline`: Mechanism: Use the existing gated SwiGLU MLP at width 1536 with the unchanged attention stack, optimizer, and training protocol. Changes: not recorded.
- Launch code provenance:
  - `pretrain-swiglu-baseline-seed0`: parent=`aac8d3ce34fe66eb3dc60ae2c55213901c81638b` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-baseline-v1/summary.json`; external_detail=True
- Conclusion: The residual MLP path completed one full train-validation-checkpoint cycle at step 2,812 with a readable checkpoint, online W&B telemetry, a 1.843 ms isolated-path median, and first-cycle plus terminal notifications accepted on their first delivery attempts. Its positive CLS auxiliary shuffle gap (0.487) confirms that the predictor output depends on the learned one-CLS representation. This smoke establishes mechanical validity only; its one-epoch accuracy is not a scientific comparison.
- Follow-up: launch the preregistered residual affine and residual MLP seed-0 candidates in `cls-register-residual-v1` and compare both against the immutable four-CLS reference using the fixed conjunctive equivalence gate.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-swiglu-baseline-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-23T16:23:58.118059+00:00; finished=2026-07-23T19:29:13.382002+00:00; terminal_event=d89931c4-6f0f-4eac-b6da-c693108d263c; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-baseline-v1/runs/pretrain-swiglu-baseline-seed0`; W&B=offline/unlinked (`d0d4e93f`); checkpoint=retained; metrics=peak_accuracy=0.905800, final_accuracy=0.900000, step_to_90=6090, step_to_95=8265, active_seconds_to_90=3889.974, active_seconds_to_95=5278.718, step_auc=0.795948, active_time_auc=0.795738; error=none
<!-- autoresearch-operation:{"content_sha256":"bd2137604675b2e1c8d6f7b5152a442ae60a5be9612135e80c25b5a45430e314","operation_id":"5cc3b18310614f1eda368b23ed797625"} -->

<!-- study:srelu-mlp-width-v1:phase:screening -->
## srelu-mlp-width-v1

- Question: Which Squared ReLU FFN width best matches or improves the SwiGLU seed-0 baseline under nominal-width, compute-equivalent, and parameter-equivalent workloads?
- Hypothesis: The tensor-core-aligned SReLU width 2304 will preserve baseline peak accuracy within 0.005 while improving active-time AUC or time to the fixed 95 percent target enough to qualify for directional tuning.
- Mechanisms and exact changes:
  - `swiglu-fixed-v1`: Mechanism: Reuse the exact validation curve from srelu-mlp-baseline-v1 without scheduling another baseline run. Changes: not recorded.
  - `srelu-h1536`: Mechanism: Replace gated SwiGLU with Squared ReLU while retaining FFN width 1536, all attention shapes, depth, optimizer settings, and dropout. Changes: Set activation to srelu at FFN width 1536.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
  - `srelu-h2304`: Mechanism: Replace gated SwiGLU with Squared ReLU and expand the up-projection to the aligned compute-equivalent width 2304. Changes: Set activation to srelu and FFN width to 2304.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
  - `srelu-h2305`: Mechanism: Replace gated SwiGLU with Squared ReLU and expand the up-projection to exact parameter-equivalent width 2305. Changes: Set activation to srelu and FFN width to 2305.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
- Launch code provenance:
  - `pretrain-srelu-h1536-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h2304-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h2305-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
- Phase: screening
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/summary.json`; external_detail=True
- Conclusion: Seed-0 screening is still running.
- Follow-up: complete the preregistered seed-0 screening trials.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-srelu-h1536-seed0`: attempt=1; status=completed; decision=pending; started=2026-07-23T21:29:10.330041+00:00; finished=2026-07-24T00:08:24.852090+00:00; terminal_event=db036c5d-483f-4604-8f9a-571d8c96a90d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h1536-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/bb3b4dd7); checkpoint=retained; metrics=peak_accuracy=0.889800, final_accuracy=0.888000, step_to_90=9570, step_to_95=13050, active_seconds_to_90=5218.638, active_seconds_to_95=7110.204, step_auc=0.743127, active_time_auc=0.742593; error=none
- `pretrain-srelu-h2304-seed0`: attempt=1; status=running; decision=pending; started=2026-07-23T21:29:10.226972+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h2304-seed0`; W&B=offline/unlinked (`0418f32e`); checkpoint=retained; metrics=unavailable; error=none
- `pretrain-srelu-h2305-seed0`: attempt=1; status=running; decision=pending; started=2026-07-24T00:09:43.882483+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h2305-seed0`; W&B=offline/unlinked (`5b8fd9d6`); checkpoint=retained; metrics=unavailable; error=none
<!-- autoresearch-operation:{"content_sha256":"54fd1968d7096d62ce561bef7dbf67f8749d5f523c9eab213564d4eeb104bbd2","operation_id":"01c6d8f22d1a77f19c6a1fe30760ea1e"} -->

<!-- study:srelu-mlp-width-v1:phase:screening -->
## srelu-mlp-width-v1

- Question: Which Squared ReLU FFN width best matches or improves the SwiGLU seed-0 baseline under nominal-width, compute-equivalent, and parameter-equivalent workloads?
- Hypothesis: The tensor-core-aligned SReLU width 2304 will preserve baseline peak accuracy within 0.005 while improving active-time AUC or time to the fixed 95 percent target enough to qualify for directional tuning.
- Mechanisms and exact changes:
  - `swiglu-fixed-v1`: Mechanism: Reuse the exact validation curve from srelu-mlp-baseline-v1 without scheduling another baseline run. Changes: not recorded.
  - `srelu-h1536`: Mechanism: Replace gated SwiGLU with Squared ReLU while retaining FFN width 1536, all attention shapes, depth, optimizer settings, and dropout. Changes: Set activation to srelu at FFN width 1536.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
  - `srelu-h2304`: Mechanism: Replace gated SwiGLU with Squared ReLU and expand the up-projection to the aligned compute-equivalent width 2304. Changes: Set activation to srelu and FFN width to 2304.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
  - `srelu-h2305`: Mechanism: Replace gated SwiGLU with Squared ReLU and expand the up-projection to exact parameter-equivalent width 2305. Changes: Set activation to srelu and FFN width to 2305.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
- Launch code provenance:
  - `pretrain-srelu-h1536-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h2304-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h2305-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
- Phase: screening
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/summary.json`; external_detail=True
- Conclusion: Seed-0 screening is still running.
- Follow-up: complete the preregistered seed-0 screening trials.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-srelu-h1536-seed0`: attempt=1; status=completed; decision=pending; started=2026-07-23T21:29:10.330041+00:00; finished=2026-07-24T00:08:24.852090+00:00; terminal_event=db036c5d-483f-4604-8f9a-571d8c96a90d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h1536-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/bb3b4dd7); checkpoint=retained; metrics=peak_accuracy=0.889800, final_accuracy=0.888000, step_to_90=9570, step_to_95=13050, active_seconds_to_90=5218.638, active_seconds_to_95=7110.204, step_auc=0.743127, active_time_auc=0.742593; error=none
- `pretrain-srelu-h2304-seed0`: attempt=1; status=completed; decision=pending; started=2026-07-23T21:29:10.391299+00:00; finished=2026-07-24T00:28:50.599391+00:00; terminal_event=d1116df4-4255-464e-a57b-5697c75a2840; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h2304-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/0418f32e); checkpoint=retained; metrics=peak_accuracy=0.882200, final_accuracy=0.882200, step_to_90=9135, step_to_95=13485, active_seconds_to_90=5650.130, active_seconds_to_95=8332.038, step_auc=0.752030, active_time_auc=0.734660; error=none
- `pretrain-srelu-h2305-seed0`: attempt=1; status=running; decision=pending; started=2026-07-24T00:09:43.882483+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h2305-seed0`; W&B=offline/unlinked (`5b8fd9d6`); checkpoint=retained; metrics=unavailable; error=none

## srelu-mlp-width-v1 final-checkpoint signal diagnostic

- Scope: Read-only CPU analysis of the completed SwiGLU seed-0, SReLU-1536 seed-0, and SReLU-2304 seed-0 checkpoints. The in-flight SReLU-2305 run was not read, queried, paused, or modified.
- Protocol: The primary comparison used 32 fixed validation-holdout images, four batches of eight, eval mode, mask seed 1, CPU float32, checkpoint step 17,400, and epoch 399. An eight-image train-mode replication included the trained dropout behavior.
- Gate occupancy: Mean negative-gate fractions were 0.8036 for SwiGLU, 0.9151 for SReLU-1536, and 0.9310 for SReLU-2304. Mean sampled-never-positive channel fractions were 0.0777, 0.3857, and 0.4659, respectively.
- Gradient flow: Negative gates contained 0.7012 of SwiGLU upstream gate-gradient energy, 0.8480 for SReLU-1536, and 0.8632 for SReLU-2304. SwiGLU retained nonzero preactivation gradients at 0.9917 of negative-gate positions on average; SReLU retained none because its derivative is exactly zero there.
- Depth: At layer 11, negative-gate fractions reached 0.9783, 0.9906, and 0.9954, while sampled-never-positive channel fractions reached 0.6335, 0.6237, and 0.7682. Wider SReLU increased rather than relieved inactive-channel burden.
- Replication: Train-mode mean negative-gate fractions were 0.8018, 0.9147, and 0.9303; sampled-never-positive channel fractions were 0.0986, 0.4216, and 0.5141. The direction therefore persisted with dropout enabled.
- Interpretation: The result supports negative-side gradient loss as a plausible contributor to SReLU's weaker convergence. It does not prove causality or when the distribution emerged. Prefer a bounded negative-gradient-preserving activation intervention before broad AdamW tuning; dropout-only tuning is unlikely to fix the eval-mode mechanism.
- Structured result: `research/diagnostics/srelu-mlp-final-checkpoint-signals-v1.json`; full local layer records remain under `logs/research/srelu-mlp-width-v1/diagnostics/final-checkpoints/`.
<!-- autoresearch-operation:{"content_sha256":"5cfd0ecb51204cff238c497ee7c1c2f3435d69e3707f350af4f0b56fb099bb0d","operation_id":"95c7bc8be3c357cc8cae8f239f25c837"} -->

<!-- study:srelu-mlp-bias-v1:phase:no-promotion -->
## srelu-mlp-bias-v1

- Question: Can a positive trainable SReLU up-projection bias reduce early gate starvation and improve convergence without changing the Squared ReLU activation?
- Hypothesis: Initializing the h1536 SReLU MLP up-projection bias to 0.1 or 0.2 will improve common-horizon active-time AUC or time to the fixed 95 percent target while preserving peak validation accuracy within 0.005 of the zero-bias h1536 result.
- Mechanisms and exact changes:
  - `swiglu-fixed-v1`: Mechanism: Reuse the exact validation curve from srelu-mlp-baseline-v1 without scheduling another baseline run. Changes: not recorded.
  - `srelu-h1536-bias0p1`: Mechanism: Initialize every trainable MLP fc1 bias to 0.1 while retaining exact relu(x)^2 activation, h1536 width, dropout, optimizer settings, attention shapes, and depth. Changes: Set mlp_fc1_bias_init to 0.1.; Preserve exact Squared ReLU with no negative-side leakage.
  - `srelu-h1536-bias0p2`: Mechanism: Initialize every trainable MLP fc1 bias to 0.2 while retaining exact relu(x)^2 activation, h1536 width, dropout, optimizer settings, attention shapes, and depth. Changes: Set mlp_fc1_bias_init to 0.2.; Preserve exact Squared ReLU with no negative-side leakage.
- Launch code provenance:
  - `pretrain-srelu-h1536-bias0p1-seed0`: parent=`6a8e265e2a4ed2e9ea72630a53ad3027ff1aafc3` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`06d6cee3a1872e2b70bc236dd7e85a2435a71b67` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h1536-bias0p2-seed0`: parent=`6a8e265e2a4ed2e9ea72630a53ad3027ff1aafc3` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`06d6cee3a1872e2b70bc236dd7e85a2435a71b67` (`codex/research/srelu-mlp-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-bias-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-srelu-h1536-bias0p1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-24T12:31:50.814145+00:00; finished=2026-07-24T15:09:59.722875+00:00; terminal_event=383e5c5b-d2f3-442a-9827-55da570751f5; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-bias-v1/runs/pretrain-srelu-h1536-bias0p1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/7dc1708e); checkpoint=retained; metrics=peak_accuracy=0.873000, final_accuracy=0.873000, step_to_90=10005, step_to_95=14355, active_seconds_to_90=5449.873, active_seconds_to_95=7814.483, step_auc=0.736805, active_time_auc=0.735754; error=none
- `pretrain-srelu-h1536-bias0p2-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-24T12:31:50.867969+00:00; finished=2026-07-24T15:09:05.275993+00:00; terminal_event=5f1f29cb-6572-4e1b-a0aa-0854dff04065; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-bias-v1/runs/pretrain-srelu-h1536-bias0p2-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/79ee8e38); checkpoint=retained; metrics=peak_accuracy=0.868400, final_accuracy=0.868400, step_to_90=11310, step_to_95=17400, active_seconds_to_90=6123.305, active_seconds_to_95=9419.533, step_auc=0.721837, active_time_auc=0.721663; error=none
<!-- autoresearch-operation:{"content_sha256":"45adc0ab16eb0bcd0273d3505f80a82b3a1652fdab16f3250e679eae926164d6","operation_id":"srelu-mlp-bias-v1-final-checkpoint-signals-v1"} -->
## srelu-mlp-bias-v1 final-checkpoint signal diagnostic

- Scope: CPU analysis of the completed SReLU-1536 checkpoints initialized with FC1 biases of 0.1 and 0.2, compared with the completed zero-bias SReLU-1536 checkpoint.
- Protocol: The primary comparison used 32 fixed validation-holdout images, four batches of eight, eval mode, mask seed 1, CPU float32, checkpoint step 17,400, and epoch 399. An eight-image train-mode replication included the trained dropout behavior.
- Outcome: The 0.1 and 0.2 candidates peaked at 0.8730 and 0.8684, respectively, versus 0.8898 for zero-bias SReLU. Their active times to the fixed 95 percent target were 9.9 percent and 32.5 percent slower. Wall times were only 0.7 percent and 1.3 percent shorter, so neither candidate improved quality or convergence per unit time.
- Learned biases: Mean final FC1 biases were -0.0864 and -0.0112 for the 0.1 and 0.2 initializations, versus -0.1647 for zero-bias SReLU. The positive initialization therefore shifted the learned bias but did not preserve a positive final offset.
- Gate occupancy: Mean negative-gate fractions were 0.9146 and 0.9140, nearly unchanged from 0.9151 for zero-bias SReLU. Mean sampled-never-positive channel fractions fell modestly from 0.3857 to 0.3613 and 0.3680.
- Gradient flow: Negative gates contained 0.8605 and 0.8754 of upstream gate-gradient energy, higher than 0.8480 for zero-bias SReLU. Both biased runs retained zero preactivation gradients at every sampled negative-gate position.
- Replication: Train-mode mean negative-gate fractions were 0.9147 and 0.9136; sampled-never-positive channel fractions were 0.4085 and 0.4103. Dropout did not change the direction.
- Interpretation: Positive initialization alone is rejected. The surrounding model adapted so final gate means became more negative even though learned biases remained less negative than the zero-bias reference. A future bias intervention would need to preserve its operating-point shift during training rather than only change initialization.
- Notification recovery: Both terminal events were durable and the controller remained alive, but automatic delivery failed because launch captured a null permission-profile identity while the restarted app reported `:danger-full-access` on resume. Training status was unaffected. The user manually surfaced completion, and the coordinator recovered from the persisted terminal states without live polling.
- Structured result: `research/diagnostics/srelu-mlp-bias-final-checkpoint-signals-v1.json`; full local layer records remain under `logs/research/srelu-mlp-bias-v1/diagnostics/final-checkpoints/`.
<!-- autoresearch-operation:{"content_sha256":"b234b9192d650983dee1e03c5e92c203d2d0990d6509f3ce072d9dbc26054359","operation_id":"beb857959c0fae38e0c7e2869c0c0e8a"} -->

<!-- study:srelu-mlp-width-v1:phase:no-promotion -->
## srelu-mlp-width-v1

- Question: Which Squared ReLU FFN width best matches or improves the SwiGLU seed-0 baseline under nominal-width, compute-equivalent, and parameter-equivalent workloads?
- Hypothesis: The tensor-core-aligned SReLU width 2304 will preserve baseline peak accuracy within 0.005 while improving active-time AUC or time to the fixed 95 percent target enough to qualify for directional tuning.
- Mechanisms and exact changes:
  - `swiglu-fixed-v1`: Mechanism: Reuse the exact validation curve from srelu-mlp-baseline-v1 without scheduling another baseline run. Changes: not recorded.
  - `srelu-h1536`: Mechanism: Replace gated SwiGLU with Squared ReLU while retaining FFN width 1536, all attention shapes, depth, optimizer settings, and dropout. Changes: Set activation to srelu at FFN width 1536.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
  - `srelu-h2304`: Mechanism: Replace gated SwiGLU with Squared ReLU and expand the up-projection to the aligned compute-equivalent width 2304. Changes: Set activation to srelu and FFN width to 2304.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
  - `srelu-h2305`: Mechanism: Replace gated SwiGLU with Squared ReLU and expand the up-projection to exact parameter-equivalent width 2305. Changes: Set activation to srelu and FFN width to 2305.; Preserve attention, depth, AdamW learning rate 0.002, weight decay 0.2, and MLP dropout 0.1.
- Launch code provenance:
  - `pretrain-srelu-h1536-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h2304-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
  - `pretrain-srelu-h2305-seed0`: parent=`939dd4482d35e4b4a50a9decb00404d57af07caa` (`codex/research/srelu-mlp-v1`), mjepa=`35934d979078a0f26a83921e2d80821338f41375` (`codex/research/muon-optimizer-v2`), vit=`52a4a676575bde0e756376a59b001aa55d5d6eaa` (`codex/research/srelu-mlp-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-srelu-h1536-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-23T21:29:10.330041+00:00; finished=2026-07-24T00:08:24.852090+00:00; terminal_event=db036c5d-483f-4604-8f9a-571d8c96a90d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h1536-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/bb3b4dd7); checkpoint=retained; metrics=peak_accuracy=0.889800, final_accuracy=0.888000, step_to_90=9570, step_to_95=13050, active_seconds_to_90=5218.638, active_seconds_to_95=7110.204, step_auc=0.743127, active_time_auc=0.742593; error=none
- `pretrain-srelu-h2304-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-23T21:29:10.391299+00:00; finished=2026-07-24T00:28:50.599391+00:00; terminal_event=d1116df4-4255-464e-a57b-5697c75a2840; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h2304-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/0418f32e); checkpoint=retained; metrics=peak_accuracy=0.882200, final_accuracy=0.882200, step_to_90=9135, step_to_95=13485, active_seconds_to_90=5650.130, active_seconds_to_95=8332.038, step_auc=0.752030, active_time_auc=0.734660; error=none
- `pretrain-srelu-h2305-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-24T00:09:44.049225+00:00; finished=2026-07-24T03:24:35.880637+00:00; terminal_event=411c5780-7b20-4216-8c48-21ec2ffae9a8; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/srelu-mlp-width-v1/runs/pretrain-srelu-h2305-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/5b8fd9d6); checkpoint=retained; metrics=peak_accuracy=0.873200, final_accuracy=0.873200, step_to_90=10440, step_to_95=14790, active_seconds_to_90=7012.789, active_seconds_to_95=9926.468, step_auc=0.736975, active_time_auc=0.706266; error=none
<!-- autoresearch-operation:{"content_sha256":"1dd50f5c0fde6140aee5946e0df4175111d8cb784a408edf7e4ad4c21d7c6942","operation_id":"srelu-mlp-v1-program-closeout-v1"} -->
## srelu-mlp-v1 program closeout

- Decision: Close the linked SReLU program at no-promotion. Do not spend the remaining budget on dropout, AdamW, replication, or supervised evaluation.
- Baseline comparison: SwiGLU reached 0.9058 peak validation accuracy and the fixed 95 percent target in 5,278.718 active seconds. The best SReLU result, h1536, reached 0.8898 and required 7,110.204 active seconds. Its 1,560.742-second wall-time saving did not offset the 34.7 percent slower target convergence or 0.0160 accuracy loss.
- Same-workload comparison: Compute-equivalent h2304 peaked at 0.8822 and reached the target in 8,332.038 active seconds. Parameter-equivalent h2305 peaked at 0.8732 and required 9,926.468 active seconds; its 11,691.831-second wall time was also slower than the 11,115.264-second baseline.
- Bias comparison: Positive FC1 initializations of 0.1 and 0.2 peaked at 0.8730 and 0.8684. Neither preserved the zero-bias SReLU result or improved convergence per unit time.
- Signal evidence: Final-checkpoint analysis found mean negative-gate fractions of 0.8036 for SwiGLU, 0.9151 for SReLU h1536, and 0.9310 for SReLU h2304. SwiGLU retained nonzero preactivation gradients at 0.9917 of sampled negative-gate positions; every sampled SReLU negative-gate gradient was zero. Positive initial bias did not change that result.
- Scope: Six scientific seed-0 runs completed, plus one excluded mechanical smoke run. Ten authorized scientific runs remain unused.
- Timing: The scientific program spanned 81,961.605 wall seconds. Summed run wall time was 62,065.142 seconds.
- Structured result: `research/diagnostics/srelu-mlp-program-closeout-v1.json`.
<!-- autoresearch-operation:{"content_sha256":"52d7db3432aade190ed2313b1c6a80ddbd235a05ef63ff2302d476cd3b61d990","operation_id":"srelu-mlp-v1-review-vit-pin-67eae237"} -->


## srelu-mlp-v1 review dependency amendment

- Recorded: 2026-07-24T19:03:25Z.
- Execution provenance: Width runs used local `vit` commit `52a4a676575bde0e756376a59b001aa55d5d6eaa`; bias runs used local `vit` commit `06d6cee3a1872e2b70bc236dd7e85a2435a71b67`.
- Review dependency: The adopted branch now pins landed `vit` master commit `67eae23786b8e458334b695be8f8a879d6994a43`, which provides graph-connected MLP tracing.
- Equivalence: The local independent MLP dropout setting equaled `hidden_dropout`, so the landed pin preserves that effective dropout configuration.
- Limitation: The landed pin does not apply the local FC1 bias-initialization extension. A fresh biased run would therefore not exactly reproduce its initialization. Completed checkpoints, run-local configurations, execution SHAs, W&B records, and committed diagnostics remain the canonical evidence. The user accepted this approximation for review.
<!-- autoresearch-operation:{"content_sha256":"6386a34dc0b4246e2669e0235d4112fc4d8ae5004d416910144c166501c8826f","operation_id":"20035a070b52725a4933205f9de426a1"} -->

<!-- study:cls-token-adaln-v1-smoke:phase:no-promotion -->
## cls-token-adaln-v1-smoke

- Question: Can the single-CLS AdaLN path train, validate, benchmark, checkpoint, resume, summarize, and notify through one managed GPU epoch?
- Hypothesis: The one-epoch adaln-blind smoke run will complete with the isolated path benchmark, true and shuffled CLS diagnostics, valid gradients, recoverable checkpoint metadata, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-adaln-smoke`: Mechanism: Exercise the same blinded shared-MLP path, diagnostics, checkpoint metadata, and startup benchmark used by the formal candidates at smoke scale. Changes: Use one CLS token and adaln_blind mode.; Use a one-block, one-epoch mechanical configuration.
- Launch code provenance:
  - `pretrain-cls-adaln-smoke-seed0`: parent=`4527116791c02a9431c2a12d1264dcd46e2753d4` (`codex/research/cls-token-adaln-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=False; approved_data_classes=none
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1-smoke/summary.json`; external_detail=False
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-adaln-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-24T20:17:00.639725+00:00; finished=2026-07-24T20:19:27.925032+00:00; terminal_event=4e0dbbd0-fe20-4bd3-91e9-753d5306da08; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1-smoke/runs/pretrain-cls-adaln-smoke-seed0`; W&B=offline/unlinked (`63fee026`); checkpoint=retained; metrics=peak_accuracy=0.221600, final_accuracy=0.221600, step_to_90=2812, step_to_95=2812, active_seconds_to_90=133.375, active_seconds_to_95=133.375, step_auc=0.221600, active_time_auc=0.221600, active_seconds_at_step_horizon=133.375, cls_path_latency_median_ms=0.788000, cls_path_latency_p90_ms=0.796672; error=none
<!-- autoresearch-operation:{"content_sha256":"8e6ca4b5f103bf840a5a10de3766c006ad5f0442fb697d9cf33b2aa9bf7a5f3d","operation_id":"53ded8a23abcd54b80e814de2f3aaf0c"} -->

<!-- study:cls-token-adaln-v1:phase:no-promotion -->
## cls-token-adaln-v1

- Question: Can a single CLS token trained through a visually blinded, shared AdaLN predictor path preserve representation quality while reducing end-to-end and isolated CLS-prediction cost?
- Hypothesis: A single CLS token optimized against masked teacher patch targets through a visually blinded AdaLN MLP path will preserve baseline peak validation-probe accuracy within 0.005, reduce active training time at the common final optimizer-step horizon by at least 5 percent, and reduce isolated CLS-path latency.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Run the current predictor twice, using student visual tokens for the main pass and all four student CLS tokens for the auxiliary pass. Changes: not recorded.
  - `single-cls-legacy`: Mechanism: Retain the full auxiliary cross-attention replay while supplying exactly one student CLS token. Changes: Set num_cls_tokens to 1.; Preserve the legacy auxiliary predictor replay and all other training settings.
  - `single-cls-adaln-blind`: Mechanism: Reuse each main predictor block's AdaLN-Zero MLP and output projection for target-position queries, skip all attention and visual tokens in the auxiliary path, and condition the main path on a constant zero embedding. Changes: Set num_cls_tokens to 1.; Replace the auxiliary full predictor replay with the shared visually blinded AdaLN MLP path.; Use a constant zero embedding for main-predictor AdaLN conditioning.
  - `single-cls-adaln-shared`: Mechanism: Use the same visually blinded AdaLN MLP auxiliary path as adaln-blind and condition the main predictor's shared AdaLN MLPs on the actual student CLS embedding. Changes: Set num_cls_tokens to 1.; Replace the auxiliary full predictor replay with the shared visually blinded AdaLN MLP path.; Condition the main predictor on the actual single CLS embedding.
- Launch code provenance:
  - `pretrain-four-cls-legacy-seed0`: parent=`dbc4dcce577f1291307b8db1947fb3d26cd0bc33` (`codex/research/cls-token-adaln-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-adaln-blind-seed0`: parent=`dbc4dcce577f1291307b8db1947fb3d26cd0bc33` (`codex/research/cls-token-adaln-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-adaln-shared-seed0`: parent=`dbc4dcce577f1291307b8db1947fb3d26cd0bc33` (`codex/research/cls-token-adaln-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-legacy-seed0`: parent=`dbc4dcce577f1291307b8db1947fb3d26cd0bc33` (`codex/research/cls-token-adaln-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-four-cls-legacy-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-24T20:23:07.112490+00:00; finished=2026-07-24T23:30:05.862877+00:00; terminal_event=aeda81a4-527d-4833-a8a5-318f85d43f30; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1/runs/pretrain-four-cls-legacy-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c72c92c8); checkpoint=retained; metrics=peak_accuracy=0.910000, final_accuracy=0.908800, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3936.280, active_seconds_to_95=5053.621, step_auc=0.801230, active_time_auc=0.791184, active_seconds_at_step_horizon=11197.841, cls_path_latency_median_ms=14.198784, cls_path_latency_p90_ms=14.665728; error=none
- `pretrain-single-cls-adaln-blind-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-24T23:22:09.062282+00:00; finished=2026-07-25T02:14:45.550856+00:00; terminal_event=bd409c9f-ef6d-488d-b9b4-40b1faceb122; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1/runs/pretrain-single-cls-adaln-blind-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/0ba4f419); checkpoint=retained; metrics=peak_accuracy=0.833400, final_accuracy=0.827200, step_to_90=11310, step_to_95=censored, active_seconds_to_90=6724.570, active_seconds_to_95=censored, step_auc=0.747240, active_time_auc=0.746737, active_seconds_at_step_horizon=10338.643, cls_path_latency_median_ms=10.262528, cls_path_latency_p90_ms=11.228160; error=none
- `pretrain-single-cls-adaln-shared-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-24T23:30:21.987160+00:00; finished=2026-07-25T02:24:05.263455+00:00; terminal_event=dc57c87f-775f-4fdc-81df-f4483e44478b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1/runs/pretrain-single-cls-adaln-shared-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/e8337169); checkpoint=retained; metrics=peak_accuracy=0.767200, final_accuracy=0.756000, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.695840, active_time_auc=0.695153, active_seconds_at_step_horizon=10407.015, cls_path_latency_median_ms=10.270192, cls_path_latency_p90_ms=10.858496; error=none
- `pretrain-single-cls-legacy-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-24T20:23:07.168827+00:00; finished=2026-07-24T23:21:38.088043+00:00; terminal_event=6fdc2a5a-d5ee-445a-904c-ade27b5a356d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-token-adaln-v1/runs/pretrain-single-cls-legacy-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/1e9fbdfe); checkpoint=retained; metrics=peak_accuracy=0.828600, final_accuracy=0.822800, step_to_90=9135, step_to_95=censored, active_seconds_to_90=5630.461, active_seconds_to_95=censored, step_auc=0.737138, active_time_auc=0.732708, active_seconds_at_step_horizon=10690.491, cls_path_latency_median_ms=13.760000, cls_path_latency_p90_ms=13.969408; error=none
<!-- autoresearch-operation:{"content_sha256":"977ba6fc67fdb78bc146286c14de677e26d3c484299e47df6d07340badd956f2","operation_id":"cls-token-adaln-v1-interpretation-2026-07-25-v1"} -->

## cls-token-adaln-v1 interpretation (2026-07-25)

- Outcome: The blinded AdaLN path reduced active time at step 17,400 by 7.67% and isolated median latency by 27.72%, but lost 0.0766 absolute peak validation accuracy and 0.0444 active-time AUC versus the four-CLS baseline. Shared AdaLN reduced active time by 7.06% and isolated latency by 27.67%, but lost 0.1428 peak accuracy.
- Mechanism evidence: Reducing the legacy model from four CLS tokens to one lost 0.0814 peak accuracy. Blinded AdaLN recovered only 0.0048 over that single-CLS control, while sharing CLS conditioning with the main predictor lost 0.0662 relative to blinded AdaLN. Positive final shuffled-minus-true auxiliary loss gaps for both AdaLN variants show that predictions remained sensitive to CLS identity, but this dependence did not preserve online-probe quality.
- Decision: Reject all three candidates. Do not run paired confirmation, supervised fine-tuning, or the official test set.
- Next falsifiable hypothesis: In a separate preregistered study, add a direct loss from one student CLS token to a pooled teacher-global target alongside the blinded patch-target loss. Require at least 0.05 absolute peak-accuracy recovery over blinded AdaLN while retaining at least a 5% common-step active-time gain and keeping student visual tokens unavailable to the auxiliary predictor.
<!-- autoresearch-operation:{"content_sha256":"854694eea42ba52ec80b767a22ff28343c0e4fe3d3621bf221c038ec933f87e4","operation_id":"cls-token-adaln-v1-retention-2026-07-25-v1"} -->

## cls-token-adaln-v1 retention (2026-07-25)

- Baseline protection: Retained `checkpoint.pt` and `backbone.safetensors` for `pretrain-four-cls-legacy-seed0`.
- Rejected candidates: Deleted `checkpoint.pt` and `backbone.safetensors` for `pretrain-single-cls-legacy-seed0`, `pretrain-single-cls-adaln-blind-seed0`, and `pretrain-single-cls-adaln-shared-seed0`.
- Storage: Freed 2,095,186,459 bytes. The deleted weights are not recoverable; metrics, configuration, provenance, and terminal records remain.
- External record: Republished the study summary to W&B after retention with each rejected run marked `deleted-not-recoverable`.
<!-- autoresearch-operation:{"content_sha256":"c8ac40381d001e121cf97c382bb20d88f85b8213b2b0e3607072baa4ab0528ad","operation_id":"cls-global-target-v1-preregistration-2026-07-25-v1"} -->

## cls-global-target-v1 preregistration (2026-07-25)

- Question: Does direct regression from one student CLS token to a pooled teacher-global target recover representation quality while the blinded AdaLN patch predictor preserves its cost advantage?
- Mechanism: Compute float32 MSE directly between the only student CLS token and the arithmetic mean of all full-image EMA-teacher visual tokens. Keep the existing unit-weight blinded patch-target loss. Do not add a projector, predictor, normalization step, or access from the blinded predictor to student visual tokens.
- Seed-0 screen: Run a fresh four-CLS legacy baseline, a fresh single-CLS blinded-AdaLN control, and direct-global-loss weights 0.1 and 0.5. Do not add fallback variants.
- Promotion: Require at least 0.05 peak validation-accuracy gain over the fresh blinded control. Also require at least 5 percent lower active time at the common final optimizer step, lower isolated median CLS-path latency, and no more than 0.0266 peak-accuracy loss versus the fresh four-CLS baseline. Disable the accuracy-only, time-to-95-only, and AUC-only routes.
- Confirmation: If one candidate qualifies, run fresh baseline and winner trials at seeds 1 and 2. Require the three-seed cost gate and at least two paired active-time and latency improvements. Do not fine-tune or inspect the official test set without confirmation.
- Metrics: Record peak and final online-probe validation accuracy, fixed-target convergence, common-horizon step and active-time AUC, active seconds at the common step, isolated-path latency, CLS-patch alignment, true and shuffled patch-target loss, true and shuffled global-target loss, and student/teacher target norms.
- Data and leakage control: Use the fixed 45,000/5,000 stratified CIFAR-10 training split. The teacher remains detached under inference mode. Reserve the official test set for a confirmed baseline and winner.
- Resources: At most eight scientific pretraining trials, two concurrent jobs, physical GPUs 1 and 2, and 24 hours per job. Require the repository storage reserve before every launch.
- Provenance and tracking: Use `codex/research/cls-global-target-v1`, parent SHA captured at launch, mjepa `c63b014aacc1860e18b0f45aca65fad88396b95e`, vit `67eae23786b8e458334b695be8f8a879d6994a43`, and online W&B destination `tidalpaladin/mjepa-cifar10`. Launch emits metrics, configs, and provenance; summary emits metrics and provenance.
- Managed paths: Specification `research/studies/cls-global-target-v1.yaml`; state and run artifacts under `logs/research/cls-global-target-v1`.
- Stopping and retention: Stop without replication if neither candidate passes both controls. Retain every managed checkpoint and backbone because destructive retention is not authorized.
<!-- autoresearch-operation:{"content_sha256":"d1472600ae60a06601ef879f0e75803669494ed611a2de758a5c9b6271c5bb12","operation_id":"cls-global-target-v1-smoke-preregistration-2026-07-25-v1"} -->

## cls-global-target-v1-smoke preregistration (2026-07-25)

- Purpose: Exercise one online-W&B GPU epoch with global-loss weight 0.1 before the scientific screen. Require finite raw and weighted losses, global and patch shuffle diagnostics, student gradients, isolated-path benchmark output, progress and first-cycle events, checkpoint and backbone files, resume, summary, accepted notification, and retained weights.
- Managed paths: Specification `research/studies/cls-global-target-v1-smoke.yaml`; state and run artifacts under `logs/research/cls-global-target-v1-smoke`.
- Rejection: Do not launch the formal study if any required metric, recovery artifact, or lifecycle transition is missing.
<!-- autoresearch-operation:{"content_sha256":"364c0d164e5b1602709b2c9cca49c55563051debdd68cc1efdb2047f321378fc","operation_id":"40182e799fde5e51e1a19014232d2dc3"} -->

<!-- study:cls-global-target-v1-smoke:phase:no-promotion -->
## cls-global-target-v1-smoke

- Question: Can direct CLS-to-teacher-global regression train, validate, checkpoint, resume, summarize, and notify through one managed GPU epoch?
- Hypothesis: The one-epoch smoke run will produce finite raw and weighted global losses, identity-sensitive global and patch diagnostics, valid student gradients, a recoverable checkpoint, an isolated-path benchmark, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-global-target-smoke`: Mechanism: Exercise the direct student-CLS to pooled-teacher MSE alongside the blinded AdaLN patch-target path. Changes: Use one CLS token and adaln_blind mode.; Add cls_global_target_loss_weight 0.1.; Use a one-block, one-epoch mechanical configuration.
- Launch code provenance:
  - `pretrain-cls-global-target-smoke-seed0`: parent=`12c0abd72ae78f0002bfe6e16afe8bd5197afbf6` (`codex/research/cls-global-target-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-global-target-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-25T11:19:38.393621+00:00; finished=2026-07-25T11:21:53.149084+00:00; terminal_event=67663664-0250-4f6c-8de5-b8787444e410; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1-smoke/runs/pretrain-cls-global-target-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/fce11404); checkpoint=retained; metrics=peak_accuracy=0.217200, final_accuracy=0.217200, step_to_90=2812, step_to_95=2812, active_seconds_to_90=121.405, active_seconds_to_95=121.405, step_auc=0.217200, active_time_auc=0.217200, active_seconds_at_step_horizon=121.405, cls_path_latency_median_ms=0.775168, cls_path_latency_p90_ms=0.790528; error=none
<!-- autoresearch-operation:{"content_sha256":"2bc884140a2406d2e198d62f564fb2233ff7fa9fa455032b16b234aa2d655dbc","operation_id":"cls-global-target-v1-smoke-interpretation-2026-07-25-v1"} -->

## cls-global-target-v1-smoke interpretation (2026-07-25)

- Outcome: Mechanical validation passed. The one-epoch online run completed at optimizer step 2,812 in 121.405 active seconds with peak validation-probe accuracy 0.2172 and isolated-path median latency 0.775168 ms.
- Loss evidence: W&B contains 56 paired raw and weighted global-loss records. Every value is finite, the weighted value equals 0.1 times the raw value exactly, and raw loss declined from 0.591156 to 0.027604 across the recorded history.
- Identity evidence: On the deterministic validation batch, global shuffled-minus-true loss was 0.203268 and blinded patch shuffled-minus-true loss was 0.130524. Both positive gaps show sensitivity to the paired CLS representation.
- Recovery evidence: The first-cycle and terminal notifications were accepted; terminal status is completed with exit code 0; checkpoint.pt and backbone.safetensors are retained. The exact checkpoint restored and exited without another epoch. The first resume attempt exposed a recomputed-benchmark W&B config collision; a regression-tested fix now preserves the launch benchmark on resume.
- Decision: Proceed to the preregistered four-run seed-0 screen after committing and pushing the resume fix, then passing formal preflight and dry-run checks.
<!-- autoresearch-operation:{"content_sha256":"90dc5e0467d05759e1566c4f8aa76654c870a0c3d1089bb06d126da7ed8235ae","operation_id":"61204f8114416cf7e016f770ab00dc30"} -->

<!-- study:cls-global-target-v1:phase:no-promotion -->
## cls-global-target-v1

- Question: Does direct regression from one student CLS token to a pooled teacher-global target recover representation quality while the blinded AdaLN patch predictor preserves its cost advantage?
- Hypothesis: Adding a direct MSE loss from the single student CLS token to the mean of the full teacher visual-token sequence will improve peak validation-probe accuracy by at least 0.05 over a fresh blinded-AdaLN control. A qualifying candidate must also remain within 0.0266 peak accuracy of a fresh four-CLS baseline, reduce common-step active time by at least 5 percent, and reduce isolated CLS-path latency.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Run the current predictor twice, using student visual tokens for the main pass and four student CLS tokens for the auxiliary pass. Changes: not recorded.
  - `single-cls-adaln-blind-control`: Mechanism: Predict masked teacher patches from one CLS-conditioned, attention-free AdaLN MLP path while keeping the main predictor conditioned on zero. Changes: Set num_cls_tokens to 1.; Keep the blinded patch-target loss at unit weight.; Do not add a direct CLS global-target loss.
  - `cls-global-mse-w0p1`: Mechanism: Add projector-free MSE from the single student CLS token to the arithmetic mean of all full-image teacher visual tokens, while retaining the blinded patch-target objective. Changes: Preserve the single-CLS adaln_blind architecture and unit-weight patch losses.; Add cls_global_target_loss_weight 0.1.
  - `cls-global-mse-w0p5`: Mechanism: Add the same projector-free CLS-to-pooled-teacher MSE at five times the lower candidate weight, with no other training change. Changes: Preserve the single-CLS adaln_blind architecture and unit-weight patch losses.; Add cls_global_target_loss_weight 0.5.
- Launch code provenance:
  - `pretrain-cls-global-mse-w0p1-seed0`: parent=`7728a962318b6f5f5135f701675c99f724fab884` (`codex/research/cls-global-target-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-cls-global-mse-w0p5-seed0`: parent=`7728a962318b6f5f5135f701675c99f724fab884` (`codex/research/cls-global-target-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-four-cls-legacy-seed0`: parent=`7728a962318b6f5f5135f701675c99f724fab884` (`codex/research/cls-global-target-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-adaln-blind-control-seed0`: parent=`7728a962318b6f5f5135f701675c99f724fab884` (`codex/research/cls-global-target-v1`), mjepa=`c63b014aacc1860e18b0f45aca65fad88396b95e` (`codex/research/cls-token-adaln-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-global-mse-w0p1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-25T14:25:29.913731+00:00; finished=2026-07-25T17:18:37.462011+00:00; terminal_event=61255dbd-f0d8-442d-8f4f-cb0862ab6fef; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1/runs/pretrain-cls-global-mse-w0p1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/88db23b1); checkpoint=retained; metrics=peak_accuracy=0.752400, final_accuracy=0.746400, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.676905, active_time_auc=0.676617, active_seconds_at_step_horizon=10370.936, cls_path_latency_median_ms=10.264576, cls_path_latency_p90_ms=11.107328; error=none
- `pretrain-cls-global-mse-w0p5-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-25T14:39:08.926694+00:00; finished=2026-07-25T17:32:44.451689+00:00; terminal_event=83a885bc-1d4a-46ee-b7ef-2c20f9f478c9; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1/runs/pretrain-cls-global-mse-w0p5-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c226c536); checkpoint=retained; metrics=peak_accuracy=0.689000, final_accuracy=0.679400, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.620068, active_time_auc=0.619684, active_seconds_at_step_horizon=10399.280, cls_path_latency_median_ms=10.273792, cls_path_latency_p90_ms=10.939392; error=none
- `pretrain-four-cls-legacy-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-25T11:31:57.220139+00:00; finished=2026-07-25T14:38:42.300528+00:00; terminal_event=fd6c54d0-b83d-40bc-9322-89e613371234; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1/runs/pretrain-four-cls-legacy-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/e1b25eec); checkpoint=retained; metrics=peak_accuracy=0.910000, final_accuracy=0.908800, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3920.513, active_seconds_to_95=5038.554, step_auc=0.801230, active_time_auc=0.792308, active_seconds_at_step_horizon=11188.104, cls_path_latency_median_ms=14.180352, cls_path_latency_p90_ms=14.379008; error=none
- `pretrain-single-cls-adaln-blind-control-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-25T11:31:57.281440+00:00; finished=2026-07-25T14:24:51.577541+00:00; terminal_event=36818c8f-a743-4799-a443-deeecc3f829b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-global-target-v1/runs/pretrain-single-cls-adaln-blind-control-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/f8bcfa68); checkpoint=retained; metrics=peak_accuracy=0.833400, final_accuracy=0.827200, step_to_90=11310, step_to_95=censored, active_seconds_to_90=6734.122, active_seconds_to_95=censored, step_auc=0.747240, active_time_auc=0.747036, active_seconds_at_step_horizon=10357.517, cls_path_latency_median_ms=10.071552, cls_path_latency_p90_ms=11.057152; error=none
<!-- autoresearch-operation:{"content_sha256":"f6129d6f288ae97f24d3cd10c11591f629fe25e24148de9be480e9b2886c4771","operation_id":"cls-global-target-v1-interpretation-2026-07-25-v1"} -->

## cls-global-target-v1 interpretation (2026-07-25)

- Outcome: Reject projector-free raw-space MSE from one student CLS token to the mean EMA-teacher visual token. Neither weight recovered representation quality, so the study stops at the four-run seed-0 screen.
- Gate results: Weight 0.1 reduced common-step active time by 7.30 percent and isolated median latency by 27.61 percent versus the four-CLS baseline, but its 0.7524 peak accuracy was 0.0810 below the blinded control and 0.1576 below baseline. Weight 0.5 reduced active time by 7.05 percent and latency by 27.55 percent, but its 0.6890 peak accuracy was 0.1444 below the control and 0.2210 below baseline. Both missed the required 0.8834 peak threshold.
- Dose response: Increasing the loss weight from 0.1 to 0.5 reduced peak accuracy by another 0.0634 and active-time AUC by another 0.0569. The direct objective therefore degraded probe quality more as its influence increased.
- Mechanism evidence: Weight 0.1 ended with true global loss 0.0518, global shuffle gap 0.3657, and patch shuffle gap 0.3699, yet probe quality remained poor. Identity sensitivity alone was not sufficient. Weight 0.5 drove true global loss to 0.00547 while final student and teacher target norms fell to 3.97 and 3.84, and its global and patch shuffle gaps fell to 0.0550 and 0.0580. The coupled norm and gap reductions are consistent with an easy low-scale EMA solution rather than richer global compression.
- Decision: Do not launch confirmation, supervised fine-tuning, or official-test evaluation. The blinded control and both candidates are rejected; the fresh four-CLS run remains the baseline.
- Tracking and timing: All four W&B runs finished, every lifecycle notification was accepted, and summary publication reported no errors. The study spanned 21,647.232 wall seconds; summed run wall time was 42,382.450 seconds.
- Retention: Retain every checkpoint and backbone as preregistered. No destructive retention action is authorized or applied.
- Next falsifiable hypothesis: In a separate preregistered study, replace raw-space regression with a disposable projected global objective that centers and unit-normalizes the pooled teacher target and enforces a batch variance floor on the student projection. Test whether removing scale as an optimization shortcut recovers at least 0.05 peak accuracy over the blinded control while preserving the same active-time and latency gates. Do not treat this as a fallback variant of the closed study.
<!-- autoresearch-operation:{"content_sha256":"02325521a8849efd17eab73e029f56e48c37c11d3c12982024abc06ec6920847","operation_id":"cls-up-project-v1-preregistration-2026-07-25-v1"} -->

## cls-up-project-v1 preregistration (2026-07-25)

- Question: Can one backbone CLS token preserve the strong four-CLS baseline quality when a learned affine map expands it to four context tokens for the unchanged legacy predictor replay?
- Mechanism: Encode one student CLS token, apply a trainable affine map from D to 4D, reshape it to four D-dimensional slots, and use those slots only as context for the auxiliary legacy cross-attention predictor pass. Keep the main visual-token predictor pass unchanged.
- Information bottleneck: The auxiliary path receives no student visual tokens, teacher features, nonlinearity, normalization, or slot embeddings. All four context slots are learned views of the same D-dimensional student CLS state.
- Fixed reference: Reuse `cls-global-target-v1/pretrain-four-cls-legacy-seed0`. Its exact 40-point validation curve is committed at `research/baselines/cls-global-target-v1-four-cls-legacy-seed0.metrics.jsonl` with SHA-256 `81a387a429495ed52cd03f97b646e32f288cccac23bcb063611f975f2b1d35a9`.
- Run plan: Launch only `pretrain-single-cls-projected-seed0`. This candidate-only follow-up uses one scientific pretraining trial, physical GPU 1 or 2, one concurrent job, and a 24-hour timeout. Do not add fallback variants or launch paired confirmation.
- Primary hypothesis gate: Peak online-probe validation accuracy must be at least 0.9050, no more than 0.005 below the fixed baseline peak of 0.9100. The repository’s stricter standard promotion routes remain unchanged; passing only this equivalence gate supports the mechanism hypothesis but is not a reference promotion or confirmation.
- Secondary metrics: Record final accuracy, fixed-target convergence, common-horizon AUC and active time, true and shuffled auxiliary CLS losses, CLS-patch alignment, isolated path latency, and executed parameter count.
- Data and leakage control: Use the fixed 45,000/5,000 stratified CIFAR-10 training split. The teacher remains detached under inference mode. Do not inspect the official test set or launch supervised evaluation.
- Tracking and provenance: Use online W&B at `tidalpaladin/mjepa-cifar10`, group `cls-up-project-v1`. Launch emits metrics, configs, and provenance; summary emits metrics and provenance. The parent and local tandem SHAs are fixed before launch.
- Retention: Retain the candidate checkpoint and backbone because destructive retention is not authorized.
<!-- autoresearch-operation:{"content_sha256":"ef380213c42b5dd3bcff7bb4b3bad7c5068889b22ab4047a0a78f07fbdf30cb0","operation_id":"cls-up-project-v1-smoke-preregistration-2026-07-25-v1"} -->

## cls-up-project-v1-smoke preregistration (2026-07-25)

- Purpose: Exercise one online-W&B GPU epoch with the projected single-CLS path before the scientific candidate.
- Required evidence: Four-token projected auxiliary context, projection gradients, isolated-path latency and parameter count that include the projection, finite training and validation metrics, progress and first-cycle events, checkpoint and backbone files, summary recovery, and accepted lifecycle notifications.
- Managed paths: Specification `research/studies/cls-up-project-v1-smoke.yaml`; state and run artifacts under `logs/research/cls-up-project-v1-smoke`.
- Resources: One mechanical pretraining run, one concurrent job on physical GPU 1 or 2, and a one-hour timeout.
- Rejection: Do not launch the scientific candidate if any required gradient, metric, benchmark, recovery artifact, or lifecycle transition is missing.
- Retention: Retain the smoke checkpoint through formal launch validation.
<!-- autoresearch-operation:{"content_sha256":"dbf48a9f29c7c63aca0e3739516c9ef86ff4f6926a76fbfd81db8929cfc70e25","operation_id":"86725410dea38c713e9ea6b75e280ee7"} -->

<!-- study:cls-up-project-v1-smoke:phase:no-promotion -->
## cls-up-project-v1-smoke

- Question: Can the projected single-CLS path train, validate, benchmark, checkpoint, summarize, and notify through one managed GPU epoch?
- Hypothesis: The one-epoch smoke run will complete with a four-token projected auxiliary context, projection gradients, online W&B telemetry, isolated-path benchmark, first-cycle checkpoint, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-projected-smoke`: Mechanism: Exercise the learned one-to-four CLS projection and unchanged legacy predictor replay at smoke scale. Changes: Use one CLS token and projected_cross_attention mode.; Use a one-block, one-epoch mechanical configuration.
- Launch code provenance:
  - `pretrain-cls-projected-smoke-seed0`: parent=`b28eaf683f44f58c313f342cc94bcc4c0143317d` (`codex/research/cls-up-project-v1`), mjepa=`2b2ed73bdc53c13790c49b3e8bd1c6462b691120` (`codex/research/cls-up-project-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-up-project-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-projected-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-25T18:57:47.488922+00:00; finished=2026-07-25T19:00:36.869067+00:00; terminal_event=34411b30-5fc8-42a1-a80b-4427eb8d3b8a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-up-project-v1-smoke/runs/pretrain-cls-projected-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/774fcc78); checkpoint=retained; metrics=peak_accuracy=0.226200, final_accuracy=0.226200, step_to_90=2812, step_to_95=2812, active_seconds_to_90=152.945, active_seconds_to_95=152.945, step_auc=0.226200, active_time_auc=0.226200, active_seconds_at_step_horizon=152.945, cls_path_latency_median_ms=1.784352, cls_path_latency_p90_ms=1.804288; error=none
<!-- autoresearch-operation:{"content_sha256":"cd59bd9fd9a48d2412b09f7723bc639ae01b47cd261b52c4c474f6bdc04eef19","operation_id":"513ce505f641e947d531ec0f3f2e8a07"} -->

<!-- study:cls-up-project-v1:phase:no-promotion -->
## cls-up-project-v1

- Question: Can one backbone CLS token preserve the strong four-CLS baseline quality when a learned affine map expands it to four context tokens for the unchanged legacy predictor replay?
- Hypothesis: The projected single-CLS candidate will reach at least 0.9050 peak online-probe validation accuracy, no more than 0.005 below the fixed seed-0 four-CLS baseline peak of 0.9100.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Run the predictor twice, using student visual tokens for the main pass and four student CLS tokens for the auxiliary legacy cross-attention pass. Changes: not recorded.
  - `single-cls-projected`: Mechanism: Encode one student CLS token, apply one trainable affine map from D to 4D, reshape it to four D-dimensional context tokens, and use those tokens only in the auxiliary legacy cross-attention predictor pass. Changes: Set backbone num_cls_tokens from 4 to 1.; Set cls_prediction_mode to projected_cross_attention.; Add one affine D-to-4D projection with truncated-normal weight initialization at standard deviation 0.02 and zero bias.
- Launch code provenance:
  - `pretrain-single-cls-projected-seed0`: parent=`9e598eee12b1200258a5e77b3bc72b362a2b3a81` (`codex/research/cls-up-project-v1`), mjepa=`2b2ed73bdc53c13790c49b3e8bd1c6462b691120` (`codex/research/cls-up-project-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-up-project-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-projected-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-25T19:05:47.036135+00:00; finished=2026-07-25T22:05:15.121547+00:00; terminal_event=b59d9ae3-443a-44bb-95ad-a8f0ecc24578; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-up-project-v1/runs/pretrain-single-cls-projected-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/bad73acf); checkpoint=retained; metrics=peak_accuracy=0.872400, final_accuracy=0.872400, step_to_90=7395, step_to_95=14355, active_seconds_to_90=4576.421, active_seconds_to_95=8872.851, step_auc=0.775052, active_time_auc=0.774680, active_seconds_at_step_horizon=10751.296, cls_path_latency_median_ms=14.311936, cls_path_latency_p90_ms=14.539776; error=none
<!-- autoresearch-operation:{"content_sha256":"4e5a2f4453ddb9fb54ce1747396088bbfee199be9a5333d962db377214989e55","operation_id":"cls-register-slots-v1-preregistration-2026-07-26-v1"} -->
## cls-register-slots-v1 preregistration (2026-07-26)

- Question: Can a backbone with exactly one CLS readout match the four-CLS reference when the removed CLS tokens are reclassified as registers and the auxiliary predictor still receives only the one CLS embedding?
- Prior evidence: The raw single-CLS legacy path peaked at 0.8286 and the D-to-4D projected path peaked at 0.8724 versus 0.9100 for four CLS tokens. The final main JEPA loss was also worse for the projected candidate (0.7080 versus 0.5017), which identifies lost backbone prefix capacity as a separate mechanism from predictor context capacity.
- Fixed reference: Reuse `cls-global-target-v1/pretrain-four-cls-legacy-seed0`. Its 40-point validation curve is committed at `research/baselines/cls-global-target-v1-four-cls-legacy-seed0.metrics.jsonl` with SHA-256 `81a387a429495ed52cd03f97b646e32f288cccac23bcb063611f975f2b1d35a9`.
- Initial screen: Run seed 0 for `single-cls-register-legacy`, `single-cls-register-slot-bias`, and `single-cls-register-projected`. Every candidate has one CLS plus seven registers, preserving the baseline prefix length of eight. The auxiliary path and online probe can access only the one CLS token; register and visual tokens remain hidden from the auxiliary replay.
- Predictor factors: Legacy uses the raw CLS. Slot bias broadcasts the CLS to four contexts and adds four learned D-dimensional slot identities. Projected uses the prior dense D-to-4D map. This orders candidates from no expansion parameters, to 4D parameters, to 4D-squared plus 4D parameters.
- Equivalence gate: Require peak accuracy at least 0.9050, step to the fixed 95-percent target no greater than 8,613, active seconds to that target no greater than 5,542.4093, step AUC at least 0.79623, and active-time AUC at least 0.791583. Every threshold must pass. This is a seed-0 equivalence result, not confirmation or a standard reference promotion.
- Selection: Among equivalence-qualified candidates, prefer fewer learned expansion parameters, then active-time AUC, peak accuracy, and time to the 95-percent target. Do not rerun the known 0.8724 four-register projector control.
- Resources and stopping: Launch three candidate-only scientific trials at seed 0, at most two concurrently on physical GPUs 1 and 2, with a 24-hour timeout per run. Four study trials remain reserved, but require a dated amendment before use. Do not launch confirmation or supervised evaluation without a separately authorized linked allocation.
- Data and tracking: Use the fixed 45,000/5,000 stratified CIFAR-10 split and keep the official test set untouched. Use online W&B at `tidalpaladin/mjepa-cifar10`, group `cls-register-slots-v1`; launch emits metrics, configs, and provenance, and summary emits metrics and provenance.
- Retention: Retain every new checkpoint and backbone because destructive retention is not authorized.
<!-- autoresearch-operation:{"content_sha256":"11ead2c247e52a0bb44af5d9a67e71e1d09fd8e396f3eba5b9eca49fc365fa39","operation_id":"cls-register-slots-v1-smoke-preregistration-2026-07-26-v1"} -->
## cls-register-slots-v1-smoke preregistration (2026-07-26)

- Purpose: Exercise one managed online-W&B GPU epoch for the new additive-slot mode and exact-path CLS diagnostic before scientific launch.
- Required evidence: Exactly one backbone CLS token; four predictor contexts equal to the CLS plus learned slot biases; gradients for every slot-bias parameter; no visual or register tokens in the auxiliary path; exact-path shuffled-CLS diagnostics; isolated-path latency and parameter count; finite train and validation metrics; progress and first-cycle events; checkpoint and backbone files; resumable metadata; summary recovery; and accepted lifecycle notifications.
- Managed paths: Specification `research/studies/cls-register-slots-v1-smoke.yaml`; state and run artifacts under `logs/research/cls-register-slots-v1-smoke`.
- Resources: One mechanical pretraining run, one concurrent job on physical GPU 1 or 2, and a one-hour timeout.
- Rejection: Do not launch the scientific screen if any required gradient, metric, benchmark, recovery artifact, or lifecycle transition is missing.
- Retention: Retain the smoke checkpoint through scientific launch validation.
<!-- autoresearch-operation:{"content_sha256":"64d1d66127a190440b847945dd4bf2511dbc536a36ee87dcaaf51be310a11205","operation_id":"f5d177a06408d8e0e79b916feaf673e6"} -->

<!-- study:cls-register-slots-v1-smoke:phase:no-promotion -->
## cls-register-slots-v1-smoke

- Question: Can the one-CLS register and additive-slot path train, validate, benchmark, checkpoint, recover, summarize, and notify through one managed GPU epoch?
- Hypothesis: The smoke run will complete with exactly one backbone CLS token, learned additive slot gradients, online W&B telemetry, an exact-path auxiliary diagnostic, an isolated-path benchmark, a first-cycle checkpoint, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-register-slot-bias-smoke`: Mechanism: Run one CLS token plus four registers and expand only the CLS as four additive predictor contexts. Changes: Use one CLS token and four register tokens.; Use slot_bias_cross_attention for the auxiliary predictor.
- Launch code provenance:
  - `pretrain-cls-register-slot-bias-smoke-seed0`: parent=`5dfe4689d26df650a457b2639cb2e19f19eaaffd` (`codex/research/cls-register-slots-v1`), mjepa=`836e740f43d7d21fc93a39ae351a2157125eebcf` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-slots-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-register-slot-bias-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-26T16:19:52.545753+00:00; finished=2026-07-26T16:22:25.632496+00:00; terminal_event=bfeaac10-5322-4112-8ab2-0f58f9b92246; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-slots-v1-smoke/runs/pretrain-cls-register-slot-bias-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/25a8121e); checkpoint=retained; metrics=peak_accuracy=0.209600, final_accuracy=0.209600, step_to_90=2812, step_to_95=2812, active_seconds_to_90=139.046, active_seconds_to_95=139.046, step_auc=0.209600, active_time_auc=0.209600, active_seconds_at_step_horizon=139.046, cls_path_latency_median_ms=1.754112, cls_path_latency_p90_ms=1.766400; error=none
<!-- autoresearch-operation:{"content_sha256":"f0f743a69f907da84076eb137b7310747e4c43c103b413e5d2bb3dc94451ff79","operation_id":"d7695adbdb67759228c1aab5a6ddb9e2"} -->

<!-- study:cls-register-slots-v1:phase:no-promotion -->
## cls-register-slots-v1

- Question: Can a backbone with one CLS readout recover four-CLS quality and convergence by retaining the same eight-token prefix as one CLS plus seven registers, while the auxiliary predictor receives only that CLS?
- Hypothesis: At least one candidate will remain within 0.005 peak validation accuracy and 10 percent convergence time of the fixed four-CLS reference because the three reclassified register tokens preserve backbone scratch capacity while the single CLS remains the only global readout and auxiliary context source.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Use four CLS tokens as the auxiliary legacy predictor context and mean them for the online probe. Changes: not recorded.
  - `single-cls-register-legacy`: Mechanism: Preserve eight backbone prefix tokens as one CLS plus seven registers; expose only the one CLS token to the unchanged auxiliary legacy predictor. Changes: Set backbone num_cls_tokens from 4 to 1.; Set backbone num_register_tokens from 4 to 7.
  - `single-cls-register-slot-bias`: Mechanism: Preserve one CLS plus seven registers, then add four learned D-dimensional predictor-owned slot biases to four broadcast views of the one CLS token. Changes: Reclassify three CLS tokens as registers.; Expand the only CLS token as CLS plus four learned slot biases for the auxiliary predictor.
  - `single-cls-register-projected`: Mechanism: Preserve one CLS plus seven registers, then apply the previously validated D-to-4D affine expansion only for the auxiliary predictor. Changes: Reclassify three CLS tokens as registers.; Expand the only CLS token with one learned D-to-4D affine map for the auxiliary predictor.
- Launch code provenance:
  - `pretrain-single-cls-register-legacy-seed0`: parent=`fd3f86cd89e7dbe45ace74263db1f52149220616` (`codex/research/cls-register-slots-v1`), mjepa=`836e740f43d7d21fc93a39ae351a2157125eebcf` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-register-projected-seed0`: parent=`fd3f86cd89e7dbe45ace74263db1f52149220616` (`codex/research/cls-register-slots-v1`), mjepa=`836e740f43d7d21fc93a39ae351a2157125eebcf` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-register-slot-bias-seed0`: parent=`fd3f86cd89e7dbe45ace74263db1f52149220616` (`codex/research/cls-register-slots-v1`), mjepa=`836e740f43d7d21fc93a39ae351a2157125eebcf` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-slots-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-register-legacy-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-26T16:25:26.969161+00:00; finished=2026-07-26T19:31:12.390004+00:00; terminal_event=78ba46d3-32ab-4c34-862f-6555bc0b4a46; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-slots-v1/runs/pretrain-single-cls-register-legacy-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/9431ca29); checkpoint=retained; metrics=peak_accuracy=0.842200, final_accuracy=0.833000, step_to_90=8265, step_to_95=censored, active_seconds_to_90=5291.221, active_seconds_to_95=censored, step_auc=0.746893, active_time_auc=0.746012, active_seconds_at_step_horizon=11127.988, cls_path_latency_median_ms=14.040064, cls_path_latency_p90_ms=14.292992; error=none
- `pretrain-single-cls-register-projected-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-26T22:43:46.267423+00:00; finished=2026-07-27T01:48:51.221673+00:00; terminal_event=313f43d9-d741-4149-97bc-f9fff41264e7; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-slots-v1/runs/pretrain-single-cls-register-projected-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/54e660f4); checkpoint=retained; metrics=peak_accuracy=0.886200, final_accuracy=0.886200, step_to_90=6960, step_to_95=10875, active_seconds_to_90=4440.698, active_seconds_to_95=6934.328, step_auc=0.784188, active_time_auc=0.783739, active_seconds_at_step_horizon=11087.592, cls_path_latency_median_ms=14.216192, cls_path_latency_p90_ms=14.586880; error=none
- `pretrain-single-cls-register-slot-bias-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-26T16:25:27.031300+00:00; finished=2026-07-26T19:30:16.558073+00:00; terminal_event=adc4518c-2dae-4dde-8ca3-a2a987251fd4; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-slots-v1/runs/pretrain-single-cls-register-slot-bias-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/838db696); checkpoint=retained; metrics=peak_accuracy=0.837400, final_accuracy=0.827600, step_to_90=8700, step_to_95=censored, active_seconds_to_90=5541.973, active_seconds_to_95=censored, step_auc=0.744620, active_time_auc=0.744265, active_seconds_at_step_horizon=11072.491, cls_path_latency_median_ms=14.126592, cls_path_latency_p90_ms=14.414848; error=none
<!-- autoresearch-operation:{"content_sha256":"f5cc709a4a8a4bc243921f5ac1a588ae8a70d3ca2700f99b8f96839ed9f75850","operation_id":"cls-register-slots-v1-residual-expansion-fallback-v1"} -->
## 2026-07-27 amendment: cls-register-slots-v1 residual expansion fallback

- Operation: `cls-register-slots-v1-residual-expansion-fallback-v1`
- Trigger: all three initial candidates failed the conjunctive seed-0 equivalence gate. The best candidate, `single-cls-register-projected`, peaked at 0.8862 against the 0.9050 floor, reached the 95% target at step 10,875 and 6,934.3 active seconds against ceilings of 8,613 and 5,542.4, and recorded step/active-time AUC of 0.784188/0.783739 against floors of 0.796230/0.791583.
- Interpretation: register reclassification recovered about 0.014 peak accuracy and direct affine expansion recovered another 0.044, while static slot identities, blinded AdaLN, and direct global-target losses did not help.
- New hypothesis: preserving the raw CLS embedding while learning four content-dependent residual views can close the remaining quality and convergence gap.
- Added trials: `single-cls-register-residual-projected` and `single-cls-register-residual-mlp`, both at seed 0 under the unchanged data, baseline reference, metrics, and equivalence thresholds.
- Budget: commit two of four reserved fallback trials, increasing the study cap from three to five. The final two trials remain unavailable without another dated amendment.
- Stopping rule: select the simpler residual affine expansion if it qualifies; otherwise select the residual MLP only if it satisfies every existing threshold. Do not replicate an unqualified result.
- Retention: retain all checkpoints and backbones; destructive retention remains unauthorized.
<!-- autoresearch-operation:{"content_sha256":"fc6c1cd980523456ef36fa5c2f63ac4d1a0dc40a5866c37f29c220a968dfc3da","operation_id":"cls-register-slots-v1-residual-expansion-linked-study-v1"} -->
## 2026-07-27 correction: linked residual expansion study

- Operation: `cls-register-slots-v1-residual-expansion-linked-study-v1`
- Corrects: `cls-register-slots-v1-residual-expansion-fallback-v1` execution packaging only.
- Reason: `launch --dry-run` confirmed that the terminal source study's atomic run registry remains fixed to its original three trials and does not schedule variants added after state creation.
- Resolution: restore `cls-register-slots-v1` to its three-run cap and execute the unchanged two residual candidates in linked study `cls-register-residual-v1` against the same hash-verified four-CLS reference, metrics, thresholds, data split, seed, and retention policy.
- Budget: the linked study consumes the same two reserved trials. Two reserved trials remain unavailable without another dated amendment.
- Scientific effect: none. No run launched before this correction, and the hypotheses, candidate configs, equivalence gate, and stopping rule are unchanged.
<!-- autoresearch-operation:{"content_sha256":"b9b1bf73012755a06f7473c2ffcad36bca792a92c2b588cd4ac2d4766164f007","operation_id":"089236daf9c8ae5f48cba47d48b7d023"} -->

<!-- study:cls-register-residual-v1-smoke:phase:no-promotion -->
## cls-register-residual-v1-smoke

- Question: Can the one-CLS register and residual nonlinear expansion path train, validate, benchmark, checkpoint, recover, summarize, and notify through one managed GPU epoch?
- Hypothesis: The smoke run will complete with exactly one backbone CLS token, seven registers, residual MLP gradients, online W&B telemetry, exact-path diagnostics, an isolated-path benchmark, a first-cycle checkpoint, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-register-residual-mlp-smoke`: Mechanism: Run one CLS token plus seven registers and expand only the CLS as four broadcast residual MLP contexts. Changes: Use one CLS token and seven register tokens.; Use residual_mlp_cross_attention for the auxiliary predictor.
- Launch code provenance:
  - `pretrain-cls-register-residual-mlp-smoke-seed0`: parent=`8ab3058a9d3b23d4e8cf2659d1e2279725297d50` (`codex/research/cls-register-slots-v1`), mjepa=`4d1c577fb57e0883544c908de9bf60d6bcfd909e` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-residual-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-register-residual-mlp-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-27T02:13:08.871921+00:00; finished=2026-07-27T02:15:48.913636+00:00; terminal_event=47afc129-ad41-483c-b834-4d9adfb01cf4; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-residual-v1-smoke/runs/pretrain-cls-register-residual-mlp-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/8554eda1); checkpoint=retained; metrics=peak_accuracy=0.201600, final_accuracy=0.201600, step_to_90=2812, step_to_95=2812, active_seconds_to_90=145.795, active_seconds_to_95=145.795, step_auc=0.201600, active_time_auc=0.201600, active_seconds_at_step_horizon=145.795, cls_path_latency_median_ms=1.843168, cls_path_latency_p90_ms=1.857536; error=none
<!-- autoresearch-operation:{"content_sha256":"a02ed8f90dd4590c25be9a9fb97555da6cef0ccd2342c47d0d7036272eb32f22","operation_id":"06a41123ebc895195c0d2ef598a9f87e"} -->

<!-- study:cls-register-residual-v1:phase:no-promotion -->
## cls-register-residual-v1

- Question: Can a residual decoder of one final CLS embedding recover four-CLS quality and convergence while preserving a one-CLS backbone and excluding all visual and register tokens from the auxiliary path?
- Hypothesis: At least one residual expansion will satisfy every existing seed-0 equivalence threshold because broadcasting the raw CLS preserves its global signal while learned content-dependent residuals provide the four slot-specific views used by the strong legacy predictor.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Use four CLS tokens as the auxiliary legacy predictor context and mean them for the online probe. Changes: not recorded.
  - `single-cls-register-residual-projected`: Mechanism: Preserve one CLS plus seven registers, broadcast the only CLS token to four predictor slots, and add one learned D-to-4D affine residual before the legacy auxiliary predictor. Changes: Reuse the one-CLS plus seven-register backbone.; Replace the direct affine expansion with broadcast CLS plus a learned affine residual.
  - `single-cls-register-residual-mlp`: Mechanism: Preserve one CLS plus seven registers, broadcast the only CLS token to four predictor slots, and add a learned D-to-D-to-4D GELU residual before the legacy auxiliary predictor. Changes: Reuse the one-CLS plus seven-register backbone.; Expand the only CLS token with a residual two-layer GELU MLP.
- Launch code provenance:
  - `pretrain-single-cls-register-residual-mlp-seed0`: parent=`e793e5fe76e69622708226d7b5c50f0ebeda180d` (`codex/research/cls-register-slots-v1`), mjepa=`4d1c577fb57e0883544c908de9bf60d6bcfd909e` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-register-residual-projected-seed0`: parent=`e793e5fe76e69622708226d7b5c50f0ebeda180d` (`codex/research/cls-register-slots-v1`), mjepa=`4d1c577fb57e0883544c908de9bf60d6bcfd909e` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-residual-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-register-residual-mlp-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T02:20:53.948092+00:00; finished=2026-07-27T05:25:28.315405+00:00; terminal_event=87742c8a-fec6-4c5a-9c33-befbc3321f1d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-residual-v1/runs/pretrain-single-cls-register-residual-mlp-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/b6b12721); checkpoint=retained; metrics=peak_accuracy=0.832200, final_accuracy=0.823200, step_to_90=13920, step_to_95=censored, active_seconds_to_90=8847.064, active_seconds_to_95=censored, step_auc=0.737163, active_time_auc=0.736982, active_seconds_at_step_horizon=11057.602, cls_path_latency_median_ms=14.049280, cls_path_latency_p90_ms=14.243808; error=none
- `pretrain-single-cls-register-residual-projected-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T02:20:53.876883+00:00; finished=2026-07-27T05:27:01.550128+00:00; terminal_event=bc4fc416-d250-4b58-9a7a-0b00930d4381; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-register-residual-v1/runs/pretrain-single-cls-register-residual-projected-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/75196036); checkpoint=retained; metrics=peak_accuracy=0.859600, final_accuracy=0.859600, step_to_90=10875, step_to_95=censored, active_seconds_to_90=6973.450, active_seconds_to_95=censored, step_auc=0.752210, active_time_auc=0.751053, active_seconds_at_step_horizon=11150.760, cls_path_latency_median_ms=14.268928, cls_path_latency_p90_ms=14.453728; error=none

<!-- autoresearch-operation:{"content_sha256":"1cfe85030b45d822fc42859a0b0b7d3020db7e118a5d0c9d6016fcc8caa76ebb","operation_id":"research-log-residual-smoke-interpretation-correction-v1"} -->
## 2026-07-27 correction: misplaced residual smoke interpretation

- Operation: `research-log-residual-smoke-interpretation-correction-v1`
- Corrects: the `Conclusion` and `Follow-up` bullets under `srelu-mlp-baseline-v1` that were accidentally replaced in commit `e793e5fe76e69622708226d7b5c50f0ebeda180d`.
- Cause: the patch used generic conclusion text without anchoring the target study header.
- Authoritative `srelu-mlp-baseline-v1` conclusion: the baseline smoke run completed; no candidates were configured for promotion.
- Authoritative `srelu-mlp-baseline-v1` follow-up: record interpretation and the next falsifiable hypothesis.
- Authoritative `cls-register-residual-v1-smoke` conclusion: the residual MLP path completed one full train-validation-checkpoint cycle at step 2,812 with a readable checkpoint, online W&B telemetry, a 1.843 ms isolated-path median, and first-cycle plus terminal notifications accepted on their first delivery attempts. Its positive CLS auxiliary shuffle gap (0.487) confirms that the predictor output depends on the learned one-CLS representation. This smoke establishes mechanical validity only; its one-epoch accuracy is not a scientific comparison.
- Authoritative `cls-register-residual-v1-smoke` follow-up: launch the preregistered residual affine and residual MLP seed-0 candidates in `cls-register-residual-v1` and compare both against the immutable four-CLS reference using the fixed conjunctive equivalence gate.
- Scientific effect: none. This correction changes only the placement of prose interpretation and does not alter any specification, run, metric, decision, artifact, or retention disposition.

<!-- autoresearch-operation:{"content_sha256":"c7dde5bdcec93378eef783344138ecba9a5dbb2d621070d931b56aca446ef5a9","operation_id":"cls-register-slots-v1-partitioned-expansion-fallback-v1"} -->
## 2026-07-27 amendment: channel-partitioned CLS expansion fallback

- Operation: `cls-register-slots-v1-partitioned-expansion-fallback-v1`
- Trigger: both residual candidates failed every fixed seed-0 equivalence threshold. Residual affine peaked at 0.8596 with 0.752210 step AUC; residual MLP peaked at 0.8322 with 0.737163 step AUC. Neither reached the fixed 95-percent convergence target.
- Cross-study evidence: the unrestricted direct D-to-4D affine expansion remains the strongest one-CLS candidate at 0.8862 peak and 0.784188 step AUC. Adding a broadcast raw-CLS residual reduced peak accuracy by 0.0266 and step AUC by 0.031978; adding a nonlinear residual reduced them by 0.0540 and 0.047025.
- Diagnostic evidence: on the same 256-example masked validation batch with seed 0, post-normalization predictor-context pairwise cosine was 0.407 for four backbone CLS tokens, 0.327 for direct dense expansion, 0.423 for residual affine expansion, and 0.580 for residual MLP expansion. This diagnostic informs mechanism selection but is not a promotion metric.
- Interpretation: raw broadcast does not preserve a useful advantage. It makes the four auxiliary contexts more alike and weakens the gradient pressure for the backbone to organize one CLS embedding into distinct predictive components. An unrestricted dense decoder creates diverse contexts but allows every output slot to reuse the full CLS vector.
- New hypothesis: partitioning the only 384-dimensional CLS embedding into four disjoint 96-dimensional channel groups and lifting each group into one predictor token will force slot-specialized information to reside inside the compressed CLS representation. At least one partitioned expansion will improve over unrestricted direct affine expansion and satisfy every unchanged equivalence threshold.
- Added trials:
  - `single-cls-register-partitioned-shared`: split the only CLS into four channel groups, apply one shared learned 96-to-384 lift to every group, and add a learned slot identity. This is the preferred simpler candidate.
  - `single-cls-register-partitioned-independent`: split the only CLS into four channel groups and apply one independent learned 96-to-384 lift per group. This tests whether slot-specific decoding capacity is required.
- Information boundary: the auxiliary path may see only four deterministic channel-partitioned views of the final student CLS. Student visual tokens, register tokens, teacher features, intermediate features, and other samples remain excluded.
- Budget: allocate the final two reserved seed-0 trials. No additional scientific pretraining trial is authorized by this program without a new user-approved budget.
- Gate: reuse the same hash-verified four-CLS reference, fixed data split, seed, metrics, and conjunctive thresholds: peak at least 0.9050, step-to-95 at most 8,613, active-seconds-to-95 at most 5,542.409, step AUC at least 0.796230, and active-time AUC at least 0.791583.
- Stopping rule: select the shared lift if it satisfies every threshold. Otherwise select the independent lift only if it satisfies every threshold. Do not replicate or fine-tune an unqualified result.
- Retention: retain all checkpoints and backbones; destructive retention remains unauthorized.
<!-- autoresearch-operation:{"content_sha256":"985a612a5d8efa35c2d75f781e11b99f261ddf2a3e73a7bb6bed40ce94a1be2b","operation_id":"c330664811986aabe25d55dc2f9eaa2d"} -->

<!-- study:cls-partitioned-slots-v1-smoke:phase:no-promotion -->
## cls-partitioned-slots-v1-smoke

- Question: Can the one-CLS independent channel-partitioned expansion path train, validate, benchmark, checkpoint, recover, summarize, and notify through one managed GPU epoch?
- Hypothesis: The smoke run will complete with exactly one backbone CLS token, seven registers, isolated channel-group gradients, online W&B telemetry, exact-path diagnostics, an isolated-path benchmark, a first-cycle checkpoint, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-register-partitioned-independent-smoke`: Mechanism: Run one CLS token plus seven registers, split only the CLS into four disjoint channel groups, and independently lift the four groups into auxiliary predictor contexts. Changes: Use one CLS token and seven register tokens.; Use partitioned_independent_cross_attention for the auxiliary predictor.
- Launch code provenance:
  - `pretrain-cls-register-partitioned-independent-smoke-seed0`: parent=`d4ed5a724dc38b1a2cbf77b17db15de5ec5fe9ea` (`codex/research/cls-register-slots-v1`), mjepa=`29b359e9b28289a60eb00271d4f9a8d3d8db2a6f` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partitioned-slots-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-register-partitioned-independent-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-27T05:46:04.336281+00:00; finished=2026-07-27T05:48:40.579632+00:00; terminal_event=64925bb5-3974-46b2-aa9b-7b4591423c3d; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partitioned-slots-v1-smoke/runs/pretrain-cls-register-partitioned-independent-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/4195c0aa); checkpoint=retained; metrics=peak_accuracy=0.224400, final_accuracy=0.224400, step_to_90=2812, step_to_95=2812, active_seconds_to_90=140.412, active_seconds_to_95=140.412, step_auc=0.224400, active_time_auc=0.224400, active_seconds_at_step_horizon=140.412, cls_path_latency_median_ms=1.896960, cls_path_latency_p90_ms=1.910784; error=none
<!-- autoresearch-operation:{"content_sha256":"9d00f40dfc51c159ab4ab16255569e18c7bbb46d5a54546b78c0e7e071eaa35b","operation_id":"bdd761af92b1207161add2d6ee00ad78"} -->

<!-- study:cls-partitioned-slots-v1:phase:no-promotion -->
## cls-partitioned-slots-v1

- Question: Can channel-partitioning one final CLS embedding recover four-CLS quality and convergence while preserving a one-CLS backbone and excluding all visual and register tokens from the auxiliary path?
- Hypothesis: At least one partitioned expansion will satisfy every existing seed-0 equivalence threshold because disjoint channel groups force slot-specialized predictive information into the one-CLS bottleneck instead of allowing every decoder slot to reuse the full embedding.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Use four CLS tokens as the auxiliary legacy predictor context and mean them for the online probe. Changes: not recorded.
  - `single-cls-register-partitioned-shared`: Mechanism: Preserve one CLS plus seven registers, split the only CLS into four 96-dimensional channel groups, apply one shared learned 96-to-384 lift, add learned slot identities, and pass the resulting four contexts to the legacy auxiliary predictor. Changes: Reuse the one-CLS plus seven-register backbone.; Expand four disjoint CLS channel groups with one shared lift and four learned biases.
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, split the only CLS into four 96-dimensional channel groups, and apply one independent learned 96-to-384 lift per group before the legacy auxiliary predictor. Changes: Reuse the one-CLS plus seven-register backbone.; Expand four disjoint CLS channel groups with four independent learned lifts.
- Launch code provenance:
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`2148ee5d823a06b6d1760f7df874d2adc3049bd6` (`codex/research/cls-register-slots-v1`), mjepa=`29b359e9b28289a60eb00271d4f9a8d3d8db2a6f` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-register-partitioned-shared-seed0`: parent=`2148ee5d823a06b6d1760f7df874d2adc3049bd6` (`codex/research/cls-register-slots-v1`), mjepa=`29b359e9b28289a60eb00271d4f9a8d3d8db2a6f` (`codex/research/cls-register-slots-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partitioned-slots-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T05:51:51.550172+00:00; finished=2026-07-27T08:56:44.472049+00:00; terminal_event=5cd41b0c-fca5-4998-99ce-b59b23ec5289; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partitioned-slots-v1/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/62c51cd8); checkpoint=retained; metrics=peak_accuracy=0.898000, final_accuracy=0.893200, step_to_90=6960, step_to_95=10005, active_seconds_to_90=4435.171, active_seconds_to_95=6369.559, step_auc=0.787907, active_time_auc=0.787593, active_seconds_at_step_horizon=11073.317, cls_path_latency_median_ms=14.086144, cls_path_latency_p90_ms=14.476288; error=none
- `pretrain-single-cls-register-partitioned-shared-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T05:51:51.481739+00:00; finished=2026-07-27T08:58:16.530314+00:00; terminal_event=b5409bc5-0bfb-411e-a2dc-17a36dbc5a59; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partitioned-slots-v1/runs/pretrain-single-cls-register-partitioned-shared-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/3df535e6); checkpoint=retained; metrics=peak_accuracy=0.889600, final_accuracy=0.886400, step_to_90=7830, step_to_95=11310, active_seconds_to_90=5027.586, active_seconds_to_95=7261.677, step_auc=0.777645, active_time_auc=0.776469, active_seconds_at_step_horizon=11167.929, cls_path_latency_median_ms=14.247936, cls_path_latency_p90_ms=14.576640; error=none
<!-- autoresearch-operation:{"content_sha256":"0381434c69b41e4526b52b9ca23efb65284bf39bd869d25b0a877eb10cbd532b","operation_id":"cls-partitioned-slots-v1-final-interpretation-v1"} -->
## 2026-07-27 interpretation: partitioned CLS expansion result

- Operation: `cls-partitioned-slots-v1-final-interpretation-v1`
- Gate authority: apply the fixed thresholds preregistered in `research/studies/cls-partitioned-slots-v1.yaml`. A candidate must satisfy all five thresholds: peak accuracy at least 0.905000, step-to-95 at most 8,613, active-seconds-to-95 at most 5,542.409, step AUC at least 0.796230, and active-time AUC at least 0.791583. The active-time AUC floor is the fixed preregistered value, not the summary harness's baseline value recomputed at the candidate common horizon.
- Gate results:

  | Candidate | Peak vs floor | Step-to-95 vs ceiling | Active seconds vs ceiling | Step AUC vs floor | Active AUC vs floor |
  |---|---:|---:|---:|---:|---:|
  | Independent partitioned lift | 0.898000 (-0.007000) | 10,005 (+1,392) | 6,369.559 (+827.150) | 0.787907 (-0.008323) | 0.787593 (-0.003990) |
  | Shared partitioned lift | 0.889600 (-0.015400) | 11,310 (+2,697) | 7,261.677 (+1,719.267) | 0.777645 (-0.018585) | 0.776469 (-0.015115) |

- Cross-design evidence: independent partitioning improved the strongest prior one-CLS direct dense expansion from 0.8862 to 0.8980 peak accuracy and from 0.784188 to 0.787907 step AUC. The independent lift exceeded the shared lift by 0.0084 peak accuracy and 0.010262 step AUC. At seed 0, this ordering is consistent with complementary channel subspaces and slot-specific readouts helping, but neither mechanism recovered four-CLS quality or convergence.
- Decision: reject both candidates. Do not replicate, fine-tune, or evaluate either candidate on the official test set. Retain every checkpoint and backbone as specified.
- Budget: these runs consumed the final two authorized scientific pretraining trials. The program now has zero remaining trials; any further scientific pretraining requires a new user-approved budget.
- Next falsifiable hypothesis, pending authorization: replace the fixed contiguous channel split with one learned orthogonal 384-to-384 analysis rotation, then apply four independent 96-to-384 lifts to the rotated disjoint groups. This preserves one final 384-dimensional CLS token and four complementary rank-96 views while allowing the subspace boundaries to align with the learned representation. A candidate must satisfy the same five fixed equivalence gates.
<!-- autoresearch-operation:{"content_sha256":"aeedf7dc06ddb0c75aa5b62d879a1bea64ec821d966fe7a530d0e7c7264659aa","operation_id":"cls-partition-count-v1-preregistration"} -->
## 2026-07-27 preregistration: independent CLS partition-count ablation

- Operation: `cls-partition-count-v1-preregistration`
- Authorization: the user explicitly authorized a bounded follow-up comparing two and eight independent partition contexts. Allocate two scientific seed-0 pretraining trials; one mechanical smoke run is excluded from this budget.
- Updated interpretation: treat the completed four-partition independent design as a practically successful backbone simplification while continuing to report the original strict four-CLS equivalence gate.
- Controlled mechanism: retain one 384-dimensional backbone CLS token, seven registers, the visually blinded auxiliary path, and the existing cross-attention predictor. Change only the number of fixed disjoint CLS channel groups and matching independent lifts.
- Capacity control: for hidden size D and S partitions, the independent lift weights contain S times D times (D / S), or D squared, parameters. Two, four, and eight partitions therefore each use 147,456 projection weights. Including slot biases, the adapters contain 148,224, 148,992, and 150,528 parameters, respectively. Predictor context length is the intended remaining difference.
- Candidates:
  - `single-cls-register-partitioned-independent-2`: split the CLS into two 192-dimensional groups and independently lift them into two predictor contexts. Hypothesis: four groups over-fragment the compressed embedding; two broader groups will preserve practical parity with less context complexity.
  - `single-cls-register-partitioned-independent-8`: split the CLS into eight 48-dimensional groups and independently lift them into eight predictor contexts. Hypothesis: finer specialization will close the residual four-CLS quality and convergence gap despite the longer context sequence.
- Fixed four-CLS reference: peak=0.910000, step-to-95=7,830, active-seconds-to-95=5,038.554, step AUC=0.801230, active-time AUC=0.796583.
- Strict equivalence gate: peak at least 0.905000, step-to-95 at most 8,613, active-seconds-to-95 at most 5,542.409, step AUC at least 0.796230, and active-time AUC at least 0.791583. A candidate must satisfy every threshold.
- Fixed four-partition single-CLS control: peak=0.898000, step-to-95=10,005, active-seconds-to-95=6,369.559, step AUC=0.787907, active-time AUC=0.787593.
- Practical-parity gate: peak at least 0.893000, step-to-95 at most 11,006, active-seconds-to-95 at most 7,006.515, step AUC at least 0.782907, and active-time AUC at least 0.782593. A candidate must satisfy every threshold.
- Selection: prefer two partitions if it satisfies practical parity. Retain four partitions over eight unless the eight-partition candidate satisfies the strict equivalence gate. Report all metric and isolated-path latency deltas regardless of selection.
- Stopping rule: do not replicate, fine-tune, or evaluate an unqualified result on the official test set. Any confirmation round requires separate authorization.
- Retention: retain all checkpoints and backbones; destructive retention is not authorized.
<!-- autoresearch-operation:{"content_sha256":"5b217f1ff6d19d55690a014a11c7b64dc8005ae63b49186e3090318974a25e62","operation_id":"cls-partition-count-v1-smoke-result"} -->
## 2026-07-27 mechanical validation: eight independent CLS partitions

- Operation: `cls-partition-count-v1-smoke-result`
- Scope: excluded one-epoch mechanical run `pretrain-cls-register-partitioned-independent-8-smoke-seed0`; this run does not consume either authorized scientific trial.
- Outcome: completed attempt 1 with exit code 0 at optimizer step 2,812 after 141.474 active seconds. Online W&B run `a0230c54` synchronized successfully.
- Architecture: the persisted config and checkpoint contain one backbone CLS token, seven registers, `partitioned_independent_cross_attention`, and `cls_context_tokens=8`. The saved independent projection has shape 8 by 32 by 4 with eight 32-dimensional biases, and all values are finite.
- Gradient and bottleneck validation: training completed with the model's all-trainable-parameter gradient assertion active on every step. The auxiliary path received only eight learned lifts of disjoint groups from the final student CLS; it did not receive visual or register tokens.
- Validation signal: peak and final online-probe validation accuracy were 0.221400. The CLS auxiliary loss was 0.928148 versus 1.430648 after cross-sample CLS shuffling, for a positive 0.502500 shuffle gap.
- Exact-path benchmark: median latency was 1.884160 ms, p90 latency was 1.899520 ms, and the isolated path contained 13,472 parameters on the miniature model.
- Recovery: CPU restoration recovered backbone, predictor, teacher, optimizer, scheduler, step 2,812, epoch 0, and `cls_context_tokens=8`; the restored partition weight shape was 8 by 32 by 4.
- Lifecycle: first-cycle event `c72ba9c8-f5f7-50cb-af58-4216821d9c03` and terminal event `13048145-c64f-408c-945e-ef228b4a7fc1` were each accepted once by the study-scoped controller under the captured permission and approval context.
- Retention: retained `checkpoint.pt` and `backbone.safetensors`; no weights were deleted.
- Decision: pass the mechanical gate and proceed with the preregistered two- and eight-partition scientific seed-0 trials.
<!-- autoresearch-operation:{"content_sha256":"d93e3b75868790381bf2c4ac451de0341680eaa90418a2ca298d0e17b7844b3d","operation_id":"ed09ae2d929e58471276a743d13a7e62"} -->

<!-- study:cls-partition-count-v1:phase:no-promotion -->
## cls-partition-count-v1

- Question: Does splitting one final CLS embedding into two or eight independently lifted channel partitions improve the practical quality and convergence of the successful four-partition single-CLS design?
- Hypothesis: Two broader partitions will preserve practical parity with less predictor context complexity, while eight finer partitions will reveal whether the residual gap to four backbone CLS tokens comes from insufficient slot specialization.
- Mechanisms and exact changes:
  - `four-cls-legacy`: Mechanism: Use four CLS tokens as the auxiliary legacy predictor context and mean them for the online probe. Changes: not recorded.
  - `single-cls-register-partitioned-independent-2`: Mechanism: Preserve one CLS plus seven registers, split the only CLS into two 192-dimensional channel groups, and independently lift them into two auxiliary predictor contexts. Changes: Reuse the one-CLS plus seven-register backbone.; Change only the independent partition count from four to two.
  - `single-cls-register-partitioned-independent-8`: Mechanism: Preserve one CLS plus seven registers, split the only CLS into eight 48-dimensional channel groups, and independently lift them into eight auxiliary predictor contexts. Changes: Reuse the one-CLS plus seven-register backbone.; Change only the independent partition count from four to eight.
- Launch code provenance:
  - `pretrain-single-cls-register-partitioned-independent-2-seed0`: parent=`9d1d7749cc843bc70e31ef7cc68c9d359fc0d574` (`codex/research/cls-partition-count-v1`), mjepa=`4c6f6f43ab0734c65f2a78aca9a21682bd7bff66` (`codex/research/cls-partition-count-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
  - `pretrain-single-cls-register-partitioned-independent-8-seed0`: parent=`9d1d7749cc843bc70e31ef7cc68c9d359fc0d574` (`codex/research/cls-partition-count-v1`), mjepa=`4c6f6f43ab0734c65f2a78aca9a21682bd7bff66` (`codex/research/cls-partition-count-v1`), vit=`67eae23786b8e458334b695be8f8a879d6994a43` (`codex/research/cls-token-adaln-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partition-count-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-register-partitioned-independent-2-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T15:01:18.838270+00:00; finished=2026-07-27T18:07:04.706143+00:00; terminal_event=ff09b89b-ca7a-4c60-8bdf-fa1e8763e399; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partition-count-v1/runs/pretrain-single-cls-register-partitioned-independent-2-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/59b70484); checkpoint=retained; metrics=peak_accuracy=0.896200, final_accuracy=0.896200, step_to_90=6960, step_to_95=11745, active_seconds_to_90=4454.896, active_seconds_to_95=7511.390, step_auc=0.781475, active_time_auc=0.780721, active_seconds_at_step_horizon=11127.080, cls_path_latency_median_ms=14.138880, cls_path_latency_p90_ms=14.414848; error=none
- `pretrain-single-cls-register-partitioned-independent-8-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T15:01:18.903156+00:00; finished=2026-07-27T18:06:27.822171+00:00; terminal_event=0b58f6ee-8fd0-4eb5-a9b4-40548f68f1bf; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-partition-count-v1/runs/pretrain-single-cls-register-partitioned-independent-8-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/5263f07e); checkpoint=retained; metrics=peak_accuracy=0.842600, final_accuracy=0.842600, step_to_90=13485, step_to_95=censored, active_seconds_to_90=8596.788, active_seconds_to_95=censored, step_auc=0.734497, active_time_auc=0.734240, active_seconds_at_step_horizon=11089.894, cls_path_latency_median_ms=14.358528, cls_path_latency_p90_ms=14.642176; error=none
<!-- autoresearch-operation:{"content_sha256":"d9e2ed31bd51e8c6c8a4c8a9993c2779cb700445911de4bfd7737fb404883b3f","operation_id":"cls-partition-count-v1-final-interpretation-v1"} -->
## 2026-07-27 interpretation: independent CLS partition-count result

- Operation: `cls-partition-count-v1-final-interpretation-v1`
- Gate authority: apply both conjunctive gates preregistered in `research/studies/cls-partition-count-v1.yaml`. Strict equivalence compares with four backbone CLS tokens. Practical parity compares with the accepted four-partition single-CLS control. A candidate must satisfy all five thresholds in the applicable gate.
- Strict four-CLS equivalence:

  | Candidate | Peak vs 0.905 floor | Step-to-95 vs 8,613 ceiling | Active seconds vs 5,542.409 ceiling | Step AUC vs 0.796230 floor | Active AUC vs 0.791583 floor |
  |---|---:|---:|---:|---:|---:|
  | Two partitions | 0.896200 (-0.008800) | 11,745 (+3,132) | 7,511.390 (+1,968.981) | 0.781475 (-0.014755) | 0.780721 (-0.010872) |
  | Eight partitions | 0.842600 (-0.062400) | censored | censored | 0.734497 (-0.061732) | 0.734240 (-0.057344) |

- Practical parity with four partitions:

  | Candidate | Peak vs 0.893 floor | Step-to-95 vs 11,006 ceiling | Active seconds vs 7,006.515 ceiling | Step AUC vs 0.782907 floor | Active AUC vs 0.782593 floor |
  |---|---:|---:|---:|---:|---:|
  | Two partitions | 0.896200 (+0.003200, pass) | 11,745 (+739, fail) | 7,511.390 (+504.875, fail) | 0.781475 (-0.001432, fail) | 0.780721 (-0.001872, fail) |
  | Eight partitions | 0.842600 (-0.050400, fail) | censored | censored | 0.734497 (-0.048410, fail) | 0.734240 (-0.048353, fail) |

- Controlled comparison with four partitions: two partitions changed peak accuracy by -0.001800, step-to-95 by +1,740, active-seconds-to-95 by +1,141.831, step AUC by -0.006432, and active-time AUC by -0.006872. Eight partitions changed peak accuracy by -0.055400 and AUC by -0.053410 step / -0.053353 active time; it never reached the 95-percent target.
- Isolated predictor path: the two-, four-, and eight-partition medians were 14.138880, 14.086144, and 14.358528 ms. Relative to four partitions, two was 0.052736 ms (0.374 percent) slower at the median but 0.061440 ms (0.424 percent) faster at p90; eight was 0.272384 ms (1.934 percent) slower at the median and 0.165888 ms (1.146 percent) slower at p90. Parameter counts were 9,781,376, 9,782,144, and 9,783,680, respectively.
- Interpretation: seed 0 shows a strong intermediate optimum at four partitions, not a monotonic benefit from fewer contexts or finer specialization. Two partitions preserve final quality but provide too few specialized predictor views to match four-partition convergence. Eight 48-dimensional groups over-fragment the single embedding and sharply degrade both quality and convergence.
- Recovery and lifecycle: both final checkpoints restored the backbone, teacher, predictor, optimizer, scheduler, step 17,400, and epoch 399 on CPU. Restored projection shapes were 2 by 384 by 192 and 8 by 384 by 48. Both first-cycle and terminal notifications were accepted once, and the study-scoped event controller was stopped after reconciliation.
- Decision: reject both candidates and retain the four-partition independent single-CLS design. Do not replicate, fine-tune, or use the official test set for either candidate. Retain all checkpoints and backbones; no weights were deleted.
- Cost: the two runs overlapped for a total study span of 3 hours 5 minutes 45.868 seconds. Summed run wall time was 6 hours 10 minutes 54.787 seconds, and summed active training time at the final step was 6 hours 10 minutes 16.974 seconds.
- Budget: the two candidates consumed both authorized scientific trials; zero trials remain.
- Next falsifiable hypothesis, pending authorization: keep four contexts and improve the channel assignment rather than changing the count. A learned orthogonal 384-to-384 analysis rotation before the four independent 96-to-384 lifts would preserve one backbone CLS token and complementary rank-96 views while allowing the partitions to align with learned features. It must satisfy the same strict five-metric gate to replace the current design.
<!-- autoresearch-operation:{"content_sha256":"1b1c9f99e009e879727071d6cbc748268fe2c2c31949b845e8f4a88491120679","operation_id":"cls-context-routing-v1-preregistration"} -->
## 2026-07-27 preregistration: single-pass CLS and visual context routing

- Operation: `cls-context-routing-v1-preregistration`
- Authorization: the user authorized a new goal and execution of a bounded study combining one final backbone CLS token with visible visual context in one predictor forward. Allocate one fresh seed-0 baseline, three seed-0 candidates, and at most four paired seed-1/2 confirmation runs if one candidate qualifies. One mechanical smoke run is excluded from the scientific budget.
- Corrected architecture fact: the active `cross_attention` predictor already uses `CrossAttentionTransformer` blocks containing cross-attention and an MLP, with no target-query self-attention. The earlier leakage concern applied only to decoder-style predictor blocks. The study therefore does not remove a block; it propagates source-visibility masks through the existing cross-attention layers.
- Question: can one raw final CLS token and the visible student visual tokens share one predictor forward while structured source masking preserves the quality and convergence of the successful four-partition single-CLS design and reduces predictor workload?
- Hypothesis: per-token routing will provide the lowest-variance pressure for every image to encode globally predictive information in one CLS token while preserving visual-token prediction. It will meet the cost promotion route against a fresh four-partition baseline: at least 5 percent lower active time at the common final optimizer step, lower isolated predictor-workload latency, and no more than 0.005 peak-accuracy loss.
- Backbone and information boundary: every run uses exactly one backbone CLS token and seven registers. Candidate predictor context contains only visible student visual tokens followed by the one final student CLS token. Registers, teacher features, masked visual tokens, intermediate features, and other samples are excluded. Visual contexts retain their spatial RoPE coordinates; the CLS suffix receives identity/no rotary position.
- Fresh baseline: `single-cls-register-partitioned-independent` retains the accepted four independent 96-to-384 CLS lifts and the existing two predictor forwards, one visual-context pass plus one CLS-context pass.
- Candidates:
  - `single-cls-joint-context-unmasked`: concatenate visible visual tokens and the raw CLS token; every target query sees both sources in one predictor forward.
  - `single-cls-joint-context-sample-routed`: use the same joint context and one predictor forward; route each example as 50 percent joint, 25 percent CLS-only, and 25 percent visual-only, with one visibility pattern shared by all target queries in that example.
  - `single-cls-joint-context-token-routed`: use the same joint context and one predictor forward; independently balance target queries across 50 percent joint, 25 percent CLS-only, and 25 percent visual-only visibility.
- Loss control: candidates produce one prediction tensor and no auxiliary CLS prediction. Multiply their JEPA prediction loss by 2.0 so its coefficient matches the baseline sum of one visual loss plus one CLS loss. Keep every other optimizer, regularizer, data, masking, probe, and training setting fixed.
- Mask integrity: apply the boolean key-visibility mask in every predictor cross-attention layer. Never hide both sources from a query. A routed query must be invariant to perturbations of its hidden source. The training forward must call the predictor exactly once for every candidate and twice for the baseline.
- Primary metrics: peak and final online-probe validation accuracy, step and active seconds to fixed 90/95-percent baseline targets, common-step and common-active-time AUC, and active seconds at the common final optimizer step.
- Mechanism diagnostics: deterministic joint, CLS-only, and visual-only validation losses; joint and CLS-only cross-sample CLS-shuffle gaps; candidate predictor-forward count; CLS and visual gradient coverage; and the complete isolated predictor-workload CUDA-event median, p90, and executed parameter count.
- Promotion: reuse the repository routes relative to the fresh four-partition baseline. Qualify on at least +0.01 peak accuracy, at least 15 percent faster active time to the fixed 95-percent target with no more than 0.005 peak loss, at least 10 percent higher active-time AUC with the same accuracy constraint, or the cost route of at least 5 percent lower common-final-step active time plus lower isolated workload latency with the same accuracy constraint. Rank qualifying candidates by active-time AUC, peak accuracy, then time to the 95-percent target.
- Strict reference: continue to report the immutable four-backbone-CLS seed-0 values: peak 0.910000, step-to-95 7,830, active-seconds-to-95 5,038.554, step AUC 0.801230, and preregistered active-time AUC 0.796583. Strict equivalence still requires peak at least 0.905000, step-to-95 at most 8,613, active-seconds-to-95 at most 5,542.409, step AUC at least 0.796230, and active-time AUC at least 0.791583.
- Mechanistic selection rule: a promoted candidate must use exactly one predictor forward and have a positive deterministic CLS-only shuffle gap. Prefer the unmasked candidate if tied, then sample routing, then token routing, because each step adds masking complexity.
- Confirmation: if one seed-0 candidate qualifies, run fresh paired baseline and winner seeds 1 and 2 under the same protocol. Require the normal three-seed mean gate and at least two paired seeds moving in the qualifying direction. Do not claim statistical significance from three pairs. No supervised fine-tuning or official-test evaluation is authorized in this goal.
- Rejection and stopping: reject a candidate that fails every promotion route, violates the information boundary or mask invariants, performs more than one predictor forward, has a nonpositive CLS-only shuffle gap, or fails gradient, latency, checkpoint, recovery, or notification validation. If no candidate qualifies, stop without replication. Do not add routing probabilities or architectural variants after seeing results without a dated amendment and new authorization.
- Resources: physical GPUs 1 and 2, at most two concurrent jobs, 24 hours per job, eight scientific pretraining trials maximum, and the repository free-space reserve. The current launch estimate is approximately two three-hour rounds for screening.
- Tracking and retention: use online W&B project `tidalpaladin/mjepa-cifar10`, group `cls-context-routing-v1`, emitting declared metrics, configs, and provenance. Retain every checkpoint and backbone because destructive retention is not authorized.
<!-- autoresearch-operation:{"content_sha256":"516d4c9dffe51455d85d0d12db878e07353cf8352d81f9e55de9319315010a44","operation_id":"bed4e65eed883a5af148710b13328fac"} -->

<!-- study:cls-context-routing-v1-smoke:phase:no-promotion -->
## cls-context-routing-v1-smoke

- Question: Can per-query CLS and visual source routing train, validate, benchmark, checkpoint, recover, summarize, and notify through one managed GPU epoch with one predictor forward?
- Hypothesis: The maximum-granularity routed design will complete with one backbone CLS token, seven registers, one predictor forward, valid per-query masks in every predictor layer, complete predictor-workload telemetry, deterministic source diagnostics, and accepted lifecycle notifications.
- Mechanisms and exact changes:
  - `cls-joint-context-token-routed-smoke`: Mechanism: Concatenate visible visual tokens and the raw CLS token, then balance target queries across joint, CLS-only, and visual-only visibility in one predictor forward. Changes: Use one CLS token and seven register tokens.; Use joint_context_token_routed with a JEPA loss coefficient of 2.0.
- Launch code provenance:
  - `pretrain-cls-joint-context-token-routed-smoke-seed0`: parent=`cf2fbc86cb0a901fa20eea177957837969cc9094` (`codex/research/cls-context-routing-v1`), mjepa=`4d95ebac24a18b17a13406cafc404602c5d9e260` (`codex/research/cls-context-routing-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-joint-context-token-routed-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-27T19:24:12.437192+00:00; finished=2026-07-27T19:26:32.852056+00:00; terminal_event=3464b074-e902-4c49-ad0a-fe4ab62f6754; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1-smoke/runs/pretrain-cls-joint-context-token-routed-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/f4f4312d); checkpoint=retained; metrics=peak_accuracy=0.201800, final_accuracy=0.201800, step_to_90=2812, step_to_95=2812, active_seconds_to_90=123.899, active_seconds_to_95=123.899, step_auc=0.201800, active_time_auc=0.201800, active_seconds_at_step_horizon=123.899, cls_path_latency_median_ms=2.567680, cls_path_latency_p90_ms=2.591744; error=none
<!-- autoresearch-operation:{"content_sha256":"a1108e59f801b4ac84e173c12fa97ba01ac68e2a9e6aad5eb4b122fee85a88f8","operation_id":"cls-context-routing-v1-smoke-result"} -->
## 2026-07-27 mechanical validation: single-pass per-query CLS routing

- Operation: `cls-context-routing-v1-smoke-result`
- Scope: excluded one-epoch mechanical run `pretrain-cls-joint-context-token-routed-smoke-seed0`; it does not consume the eight-trial scientific budget.
- Outcome: completed attempt 1 with exit code 0 at optimizer step 2,812 after 123.927 active seconds. Online W&B run `f4f4312d` synchronized successfully.
- Architecture: the persisted config and restored checkpoint contain one backbone CLS token, seven registers, `joint_context_token_routed`, and `jepa_loss_weight=2.0`. The restored predictor contains one `CrossAttentionTransformer` block with cross-attention and an MLP, no query self-attention, and returns one prediction tensor with `pred_with_cls=None`.
- Mask and gradient validation: regression tests prove that every image's four smoke queries receive an exact 2/1/1 joint, CLS-only, and visual-only split; all routed queries are invariant to perturbations of their hidden source; the same boolean mask reaches every cross-attention layer. Training completed with the all-trainable-parameter gradient assertions active for both the backbone and predictor.
- Source diagnostics: deterministic CLS-only loss was 0.782043 versus 1.314513 after cross-sample CLS shuffling, a positive 0.532470 gap. Joint-context loss was 0.651362 versus 0.653949 after CLS shuffling, a positive 0.002587 gap; visual-only loss was 0.648854.
- Complete predictor-workload benchmark: one predictor forward at batch 512, eight visible visual contexts, and 16 target queries took 2.567680 ms median and 2.591744 ms p90 on the RTX 3090; the miniature predictor contained 12,192 parameters.
- Recovery: CPU restoration recovered finite backbone, teacher, and predictor states plus saved optimizer and scheduler states, step 2,812, epoch 0, and W&B ID `f4f4312d`. A restored synthetic forward produced shape 2 by 4 by 32 with no auxiliary CLS prediction.
- Lifecycle: first-cycle event `6d1014ff-3020-50a9-8084-40f85a564fb2` and terminal event `3464b074-e902-4c49-ad0a-fe4ab62f6754` were each accepted once by the study-scoped controller under the captured permission and approval context.
- Retention: retained `checkpoint.pt` and `backbone.safetensors`; no weights were deleted.
- Decision: pass the mechanical gate and proceed with the preregistered fresh baseline and three seed-0 scientific candidates.
<!-- autoresearch-operation:{"content_sha256":"83c68b531b39f67bd2503e2a5b1d1e657531bb6dae02c970ee53ac03847057de","operation_id":"e05a5bcc2000386a344c25370a1ecebc"} -->

<!-- study:cls-context-routing-v1:phase:no-promotion -->
## cls-context-routing-v1

- Question: Can one raw final CLS token and the visible student visual tokens share one predictor forward while structured source masking preserves the quality and convergence of the four-partition single-CLS design and reduces predictor workload?
- Hypothesis: Per-token routing will provide the lowest-variance pressure for every image to encode globally predictive information in one CLS token while preserving visual-token prediction, and will satisfy the preregistered cost route against a fresh four-partition baseline.
- Mechanisms and exact changes:
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, independently lift four disjoint CLS channel partitions, and use separate visual-context and CLS-context predictor forwards. Changes: not recorded.
  - `single-cls-joint-context-unmasked`: Mechanism: Concatenate visible student visual tokens with the raw final CLS token and let every target query attend to both sources in one cross-attention predictor pass. Changes: Remove the learned four-way CLS expansion and auxiliary predictor forward.; Append the raw final CLS token to visible visual context in one predictor forward.
  - `single-cls-joint-context-sample-routed`: Mechanism: In one predictor pass, balance images across 50 percent joint, 25 percent CLS-only, and 25 percent visual-only source visibility, shared by all target queries in each image. Changes: Use the unmasked candidate's raw joint context and single predictor forward.; Apply one balanced source-visibility route per image in every predictor layer.
  - `single-cls-joint-context-token-routed`: Mechanism: In one predictor pass, balance target queries across 50 percent joint, 25 percent CLS-only, and 25 percent visual-only source visibility. Changes: Use the unmasked candidate's raw joint context and single predictor forward.; Apply a balanced source-visibility route per target query in every predictor layer.
- Launch code provenance:
  - `pretrain-single-cls-joint-context-sample-routed-seed0`: parent=`21bf3b08f564c0bd16e7e6517f977285c042e294` (`codex/research/cls-context-routing-v1`), mjepa=`4d95ebac24a18b17a13406cafc404602c5d9e260` (`codex/research/cls-context-routing-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-token-routed-seed0`: parent=`21bf3b08f564c0bd16e7e6517f977285c042e294` (`codex/research/cls-context-routing-v1`), mjepa=`4d95ebac24a18b17a13406cafc404602c5d9e260` (`codex/research/cls-context-routing-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-unmasked-seed0`: parent=`21bf3b08f564c0bd16e7e6517f977285c042e294` (`codex/research/cls-context-routing-v1`), mjepa=`4d95ebac24a18b17a13406cafc404602c5d9e260` (`codex/research/cls-context-routing-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`21bf3b08f564c0bd16e7e6517f977285c042e294` (`codex/research/cls-context-routing-v1`), mjepa=`4d95ebac24a18b17a13406cafc404602c5d9e260` (`codex/research/cls-context-routing-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-joint-context-sample-routed-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T22:10:59.437219+00:00; finished=2026-07-28T00:52:16.841532+00:00; terminal_event=67b15480-6086-444e-9683-8995c0293315; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1/runs/pretrain-single-cls-joint-context-sample-routed-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/a020cc1b); checkpoint=retained; metrics=peak_accuracy=0.848000, final_accuracy=0.848000, step_to_90=8700, step_to_95=censored, active_seconds_to_90=4834.690, active_seconds_to_95=censored, step_auc=0.740760, active_time_auc=0.739762, active_seconds_at_step_horizon=9659.176, cls_path_latency_median_ms=17.975296, cls_path_latency_p90_ms=18.689024; error=none
- `pretrain-single-cls-joint-context-token-routed-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T22:36:55.408876+00:00; finished=2026-07-28T01:18:53.503116+00:00; terminal_event=d36fb0f1-b3a0-4c83-aa84-e6cd65170ead; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1/runs/pretrain-single-cls-joint-context-token-routed-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c4b41cfc); checkpoint=retained; metrics=peak_accuracy=0.888200, final_accuracy=0.888200, step_to_90=7395, step_to_95=11310, active_seconds_to_90=4131.009, active_seconds_to_95=6313.702, step_auc=0.768193, active_time_auc=0.766745, active_seconds_at_step_horizon=9701.281, cls_path_latency_median_ms=18.276352, cls_path_latency_p90_ms=18.989056; error=none
- `pretrain-single-cls-joint-context-unmasked-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-27T19:29:56.614625+00:00; finished=2026-07-27T22:10:32.672473+00:00; terminal_event=f8de470a-9f38-4107-bf84-5495ec171984; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1/runs/pretrain-single-cls-joint-context-unmasked-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/6ddb6777); checkpoint=retained; metrics=peak_accuracy=0.871200, final_accuracy=0.871200, step_to_90=11745, step_to_95=15225, active_seconds_to_90=6496.279, active_seconds_to_95=8416.594, step_auc=0.696790, active_time_auc=0.696237, active_seconds_at_step_horizon=9616.191, cls_path_latency_median_ms=17.468384, cls_path_latency_p90_ms=18.031616; error=none
- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-27T19:29:56.546891+00:00; finished=2026-07-27T22:36:23.237042+00:00; terminal_event=04b641b3-5cae-4fed-969c-f906f5173719; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-context-routing-v1/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/426eb965); checkpoint=retained; metrics=peak_accuracy=0.898800, final_accuracy=0.898200, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3916.467, active_seconds_to_95=5031.586, step_auc=0.794415, active_time_auc=0.777298, active_seconds_at_step_horizon=11162.096, cls_path_latency_median_ms=31.555584, cls_path_latency_p90_ms=32.106495; error=none
<!-- autoresearch-operation:{"content_sha256":"9043e8115a44d6f1308c5f9b61292655c8235585163422d8bff1675f357ff972","operation_id":"cls-context-routing-v1-final-interpretation-v1"} -->
## 2026-07-28 interpretation: single-pass CLS context routing

- Operation: `cls-context-routing-v1-final-interpretation-v1`
- Gate authority: apply the four promotion routes preregistered in `research/studies/cls-context-routing-v1.yaml`. The cost route requires at least 5 percent less active time at step 17,400, lower isolated predictor-workload latency, and no more than 0.005 peak-accuracy loss. A qualifying candidate must also use one predictor forward and retain a positive deterministic CLS-only shuffle gap.
- Controlled seed-0 comparison:

  | Design | Peak accuracy | Step to 95% target | Active seconds to 95% target | Step AUC | Active-time AUC | Active-time gain at step 17,400 | Predictor median | Final CLS-only shuffle gap | Final joint CLS shuffle gap |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Four-partition baseline | 0.898800 | 7,830 | 5,031.586 | 0.794415 | 0.777298 | reference | 31.555584 ms | 0.956971 | not applicable |
  | Unmasked joint context | 0.871200 (-0.027600) | 15,225 (+7,395) | 8,416.594 (+67.275%) | 0.696790 (-0.097625) | 0.696237 (-0.081061) | 13.850% | 17.468384 ms | 0.289860 | 0.001499 |
  | Sample-routed context | 0.848000 (-0.050800) | censored | censored | 0.740760 (-0.053655) | 0.739762 (-0.037536) | 13.465% | 17.975296 ms | 1.801448 | 0.004990 |
  | Token-routed context | 0.888200 (-0.010600) | 11,310 (+3,480) | 6,313.702 (+25.481%) | 0.768193 (-0.026222) | 0.766745 (-0.010553) | 13.087% | 18.276352 ms | 2.392409 | 0.003704 |

- Promotion decision: token routing was the strongest one-pass candidate, but its 0.010600 peak-accuracy loss exceeded the 0.005 cost-route tolerance. It also reached the 95-percent target 25.481 percent later than the baseline and reduced both AUC measures. Unmasked and sample routing lost 0.027600 and 0.050800 peak accuracy, respectively. No candidate satisfied any quality, convergence, AUC, or cost route.
- Predictor cost: all candidates used one predictor forward instead of two and reduced the executed predictor parameter count from 9,782,144 to 9,633,152. Median isolated workload latency fell by 44.642 percent for unmasked, 43.036 percent for sample routing, and 42.082 percent for token routing. Final-step active training time fell by 13.850, 13.465, and 13.087 percent, respectively.
- Mechanism result: routing did force the final CLS token to carry predictive information. Relative to the unmasked design's 0.289860 CLS-only shuffle gap, sample routing reached 1.801448 and token routing reached 2.392409. Token routing outperformed sample routing because every image contributed joint, CLS-only, and visual-only queries instead of assigning one source pattern to the whole image. The joint shuffle gaps remained between 0.001499 and 0.004990, so queries with visual context still drew little incremental predictive information from the CLS token.
- Interpretation: the experiment succeeded mechanically and reduced predictor cost, but the raw one-CLS joint-context objective did not preserve representation quality. The positive CLS-only gaps rule out failure to compress information into the CLS token. The remaining evidence is consistent with interference between visual-context and CLS-context prediction inside the shared pass, especially under hard source removal. This is a seed-0 mechanistic interpretation, not a statistical claim.
- Recovery and lifecycle: all four runs completed at step 17,400 and epoch 399 with online W&B synchronization. Both newly terminal checkpoints restored finite student, teacher, and predictor states plus optimizer and scheduler state on CPU. All first-cycle and terminal events were accepted once under the captured permission and approval context.
- Retention: retained every `checkpoint.pt` and `backbone.safetensors`; no weights were deleted.
- Cost: the four screening runs spanned 5 hours 48 minutes 56.956 seconds. Summed run wall time was 11 hours 10 minutes 18.247 seconds, and summed active training time at step 17,400 was 11 hours 8 minutes 58.744 seconds.
- Decision: reject all one-pass candidates, retain the four-partition independent single-CLS design, and stop without confirmation, supervised fine-tuning, or official-test evaluation.
- Follow-up: do not tune routing probabilities inside this completed study. A future authorized study could test whether separate visual and CLS query groups or output heads inside one batched predictor call preserve objective separation while retaining most of the single-pass cost reduction.
<!-- autoresearch-operation:{"content_sha256":"e536ad95eabda2d12f7e8c5f09cae18c3e16d7392191d4ef89f33349aadcefbf","operation_id":"cls-routing-objective-v1-preregistration"} -->
## 2026-07-28 preregistration: routed CLS objective recovery

- Operation: `cls-routing-objective-v1-preregistration`
- Authorization: the user approved a bounded follow-up in goal mode. Allocate one fresh seed-0 four-partition baseline, three seed-0 candidates, and at most four fresh paired seed-1/2 confirmation runs if one candidate qualifies. One packed mechanical smoke run is excluded from the eight-trial scientific budget.
- Prior diagnosis: the accepted baseline optimizes every teacher target once from visual context and once from CLS context, `L_visual + L_cls`. With a global coefficient of 2.0, the prior 50/25/25 token-routed objective has expected weighting `L_joint + 0.5 L_cls + 0.5 L_visual`. Its 0.010600 peak loss and 44 percent slower step-to-95 therefore confound one-call sharing with changed per-source target coverage and joint-context competition.
- Shared boundary: candidates use one backbone CLS token, seven registers, one cross-attention predictor forward, visible final student visual tokens followed by the raw final student CLS token, identity RoPE for the CLS suffix, and no register, teacher, masked-visual, intermediate, or cross-sample context. The predictor has no query self-attention, so each source mask remains query-local through all layers.
- Fresh baseline: `single-cls-register-partitioned-independent` uses four independent 96-to-384 lifts of the one CLS embedding and separate visual and CLS predictor forwards.
- Candidate 1, budget dual: `single-cls-joint-context-dual-routed` randomly assigns half of each image's 16 targets to visual-only context and half to CLS-only context. It separately normalizes the two route means and sums them. This retains 16 executed queries and one call while restoring equal source-level loss weight.
- Candidate 2, packed dual: `single-cls-joint-context-packed-dual-routed` duplicates every spatial target query inside one call. The first 16 copies are visual-only and the second 16 are CLS-only. It separately normalizes and sums both route means, exactly restoring two-source target coverage with 32 executed queries but no learned CLS expansion or second call.
- Candidate 3, source-balanced token routing: `single-cls-joint-context-token-routed-source-balanced` retains the prior 50/25/25 routes, 16 queries, and coefficient 2.0. On joint queries only, add `log(N_visual)` to the CLS attention logit so one CLS key has source-level prior mass comparable to the visual-key set. Preserve negative-infinity masking for blinded sources.
- Controlled settings: keep backbone, optimizer, augmentations, data split, teacher, mask ratios, regularizers, probe, epochs, and checkpoint behavior fixed. Dual routes use `jepa_loss_weight=1.0` because their visual and CLS means are explicit; source-balanced routing retains 2.0.
- Primary metrics: peak and final online-probe validation accuracy, step and active seconds to fixed 90/95-percent baseline targets, common-step and common-active-time AUC, and active seconds at the common final step.
- Mechanism and cost metrics: deterministic joint, CLS-only, and visual-only losses; joint and CLS-only shuffle gaps; source-invariance tests; student and predictor gradient coverage; predictor-forward count; spatial and executed query counts; and complete isolated predictor-workload CUDA-event median, p90, and parameter count.
- Promotion: apply the repository's four routes relative to the fresh baseline. Qualify on at least +0.01 peak accuracy, at least 15 percent faster active time to the 95-percent target with no more than 0.005 peak loss, at least 10 percent higher active-time AUC with the same accuracy constraint, or at least 5 percent lower active time at the common final step plus lower isolated predictor latency with the same accuracy constraint.
- Practical recovery diagnostic: additionally report whether peak is at least 0.893800, step-to-95 is at most 8,613, active-seconds-to-95 is at most 5,534.744, and isolated predictor latency is at most 29.977805 ms, based on the completed baseline's 0.898800 peak, 7,830 steps, 5,031.586 active seconds, and 31.555584 ms median. This diagnostic does not replace promotion.
- Mechanistic gate: a promoted candidate must execute exactly one predictor forward, maintain the declared query count, pass hard blindness and gradient checks, and have a positive deterministic CLS-only shuffle gap.
- Selection: rank qualifiers by peak accuracy, active time to the 95-percent target, active-time AUC, then isolated latency. On a complete tie prefer budget dual, then source-balanced routing, then packed dual because that order minimizes executed queries and added mechanism.
- Confirmation: if a seed-0 candidate qualifies, run fresh paired baseline and winner seeds 1 and 2. Require the standard three-seed mean gate and at least two paired seeds moving in the qualifying direction. Do not claim statistical significance from three pairs. No supervised fine-tuning or official-test evaluation is authorized.
- Rejection and stopping: reject a candidate that fails every promotion route or any information-boundary, one-call, mask, gradient, latency, checkpoint, recovery, or lifecycle invariant. If no candidate qualifies, stop without confirmation. Do not add variants or tune routing probabilities after observing outcomes without a dated amendment and new authorization.
- Resources and tracking: use physical GPUs 1 and 2, at most two concurrent jobs, 24 hours per job, the repository free-space reserve, and online W&B project `tidalpaladin/mjepa-cifar10` group `cls-routing-objective-v1` with declared metrics, configs, and provenance.
- Retention: retain every checkpoint and backbone because destructive retention is not authorized.
<!-- autoresearch-operation:{"content_sha256":"22a4b318d5598c269a3f32a891e599aaa8f0ae4acb947e3d20584fa026ef6735","operation_id":"589b1dc5702e69816399e38e893f8be8"} -->

<!-- study:cls-routing-objective-v1-smoke:phase:no-promotion -->
## cls-routing-objective-v1-smoke

- Question: Can packed dual routing duplicate every spatial query, blind each copy to one source, normalize both source losses, and complete one managed GPU epoch through a single predictor forward?
- Hypothesis: The packed full-coverage design will train without source leakage or nonfinite values, preserve gradients through the visual and CLS routes, emit exact-path diagnostics and executed-query benchmark metadata, checkpoint, recover, summarize, and notify.
- Mechanisms and exact changes:
  - `cls-joint-context-packed-dual-routed-smoke`: Mechanism: Duplicate each target query inside one predictor call, expose one copy only to visual context and the other only to the raw CLS token, and sum separately normalized route losses. Changes: Use one CLS token and seven register tokens.; Execute two predictor queries per spatial target in one packed cross-attention call.; Use a JEPA loss coefficient of 1.0 because the two route means are summed explicitly.
- Launch code provenance:
  - `pretrain-cls-joint-context-packed-dual-routed-smoke-seed0`: parent=`cafa65331fb68b74656d84bcd67425c7648b706f` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-joint-context-packed-dual-routed-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T02:38:40.092359+00:00; finished=2026-07-28T02:40:57.351111+00:00; terminal_event=2ec7931a-f26b-4f77-a017-f4df3014c5d7; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1-smoke/runs/pretrain-cls-joint-context-packed-dual-routed-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/e698fc8f); checkpoint=retained; metrics=peak_accuracy=0.213600, final_accuracy=0.213600, step_to_90=2812, step_to_95=2812, active_seconds_to_90=120.597, active_seconds_to_95=120.597, step_auc=0.213600, active_time_auc=0.213600, active_seconds_at_step_horizon=120.597, cls_path_latency_median_ms=2.422784, cls_path_latency_p90_ms=2.458592; error=none
<!-- autoresearch-operation:{"content_sha256":"61f80a71aebd1b42218086301f295f7848b1b6208b333309509871fc153c3652","operation_id":"cls-routing-objective-v1-smoke-result"} -->
## 2026-07-28 mechanical validation: packed dual-route CLS objective

- Operation: `cls-routing-objective-v1-smoke-result`
- Scope: excluded one-epoch mechanical run `pretrain-cls-joint-context-packed-dual-routed-smoke-seed0`; it does not consume the eight-trial scientific budget.
- Outcome: completed attempt 1 with exit code 0 at optimizer step 2,812 after 120.597 active seconds. Online W&B run `e698fc8f` synchronized successfully.
- Architecture and objective: the persisted config and restored checkpoint contain one backbone CLS token, seven registers, `joint_context_packed_dual_routed`, and `jepa_loss_weight=1.0`. Every four spatial targets produce eight predictor queries in one forward: four visual-only copies followed by four CLS-only copies. The two route means are separately normalized and summed; `pred_with_cls` remains absent.
- Mask and gradient validation: regression tests prove exact visual-only and CLS-only visibility, no joint routes, hidden-source invariance, and alignment of each duplicated query with its teacher target. The smoke run completed with all-trainable-parameter gradient assertions active for both student and predictor. Final W&B losses were finite, with visual-route JEPA loss 0.841172 and CLS-route JEPA loss 1.000114.
- Mechanism diagnostics: deterministic CLS-only loss was 1.073118 versus 1.482925 after cross-sample CLS shuffling, a positive 0.409807 gap. Joint-context loss was 0.795509 versus 0.856548 after CLS shuffling, a positive 0.061039 gap; visual-only loss was 0.797122.
- Complete predictor-workload benchmark: one packed predictor forward at batch 512, eight visible visual contexts, 16 spatial targets, and 32 executed queries took 2.422784 ms median and 2.458592 ms p90 on the RTX 3090. The miniature predictor contained 12,192 parameters.
- Recovery: CPU restoration recovered finite student, teacher, predictor, optimizer, and scheduler state at step 2,812 and epoch 0 with cumulative elapsed time 120.608 seconds and W&B ID `e698fc8f`. A restored synthetic forward returned shape 2 by 8 by 32 with one attention mask of shape 2 by 1 by 8 by 9 and no auxiliary prediction tensor.
- Lifecycle: first-cycle event `88ce14e8-ced4-5213-bbe6-7ccef65ad164` and terminal event `2ec7931a-f26b-4f77-a017-f4df3014c5d7` were each accepted once through `turn/steer` under the captured `:danger-full-access` permission profile and `never` approval policy.
- Retention: retained `checkpoint.pt` and `backbone.safetensors`; no weights were deleted.
- Decision: pass the mechanical gate and proceed with the preregistered fresh four-partition baseline plus the budget-dual, packed-dual, and source-balanced seed-0 scientific candidates.
- Validation limitation: the vendored skills do not contain the protocol-referenced `quick_validate.py`; repository skill instructions were read in full and the repository quality/test gates remain the executable validation source.
<!-- autoresearch-operation:{"content_sha256":"a2a91db7f7fe05a73d2f6180f7ccecf81ffe6ba6c1afc3df9ced1ab29bbba00d","operation_id":"162a220747eca5c2d11ac52cfb970ad5"} -->

<!-- study:cls-routing-objective-v1:phase:screening -->
## cls-routing-objective-v1

- Question: Can a one-CLS, one-predictor-forward routed design recover the accepted four-partition baseline's accuracy or convergence while retaining a material predictor-cost advantage?
- Hypothesis: Separately normalizing CLS-only and visual-only routes will remove the current objective dilution; packed full coverage will best recover accuracy, while budget-preserving dual routing will retain the strongest latency advantage.
- Mechanisms and exact changes:
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, independently lift four disjoint CLS channel partitions, and use separate visual-context and CLS-context predictor forwards. Changes: not recorded.
  - `single-cls-joint-context-dual-routed`: Mechanism: Randomly split each image's target queries equally between visual-only and CLS-only visibility, compute one route mean per source, and sum the two means from one predictor tensor. Changes: Replace the 50/25/25 joint, CLS-only, and visual-only route mix with a 50/50 CLS-only and visual-only mix.; Replace the global coefficient of 2.0 with separately normalized visual and CLS route losses summed at coefficient 1.0.
  - `single-cls-joint-context-packed-dual-routed`: Mechanism: Duplicate all target queries inside one predictor call, blind the first copy to CLS and the second copy to visual context, and sum separately normalized route losses. Changes: Execute two predictor queries per spatial target but only one predictor forward.; Preserve exact baseline-like visual and CLS target coverage without a learned CLS expansion.
  - `single-cls-joint-context-token-routed-source-balanced`: Mechanism: Retain 50 percent joint, 25 percent CLS-only, and 25 percent visual-only routing, then add log of the visible-visual-key count to the CLS attention logit only on joint queries. Changes: Preserve the prior token-routed query budget and loss coefficient.; Apply a dynamic log visual-context-count bias to the CLS key on joint routes while preserving hard negative-infinity source blindness.
- Launch code provenance:
  - `pretrain-single-cls-joint-context-dual-routed-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-packed-dual-routed-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: screening
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/summary.json`; external_detail=True
- Conclusion: Seed-0 screening is still running.
- Follow-up: complete the preregistered seed-0 screening trials.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-joint-context-dual-routed-seed0`: attempt=1; status=completed; decision=pending; started=2026-07-28T02:50:23.609113+00:00; finished=2026-07-28T05:31:48.591380+00:00; terminal_event=93f53914-87c3-4b59-a9ea-21843765a78c; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-dual-routed-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/00b899eb); checkpoint=retained; metrics=unavailable; error=none
- `pretrain-single-cls-joint-context-packed-dual-routed-seed0`: attempt=1; status=running; decision=pending; started=2026-07-28T05:32:31.929243+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-packed-dual-routed-seed0`; W&B=offline/unlinked (`bbe667e8`); checkpoint=retained; metrics=unavailable; error=none
- `pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`: attempt=1; status=pending; decision=pending; started=unknown; finished=unknown; terminal_event=unknown; artifacts=`unavailable`; W&B=unavailable; checkpoint=retained; metrics=unavailable; error=none
- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=running; decision=pending; started=2026-07-28T02:50:23.371172+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=offline/unlinked (`07ae3a53`); checkpoint=retained; metrics=unavailable; error=none
<!-- autoresearch-operation:{"content_sha256":"ca52b5b5bd5cebd01feb8831391bd8ea4955f508271ee8a117d46131f9e8e0d5","operation_id":"cls-routing-objective-v1-dual-routed-seed0-terminal"} -->

## 2026-07-28 terminal result: route-normalized budget-preserving dual routing

- Operation: `cls-routing-objective-v1-dual-routed-seed0-terminal`
- Run: `pretrain-single-cls-joint-context-dual-routed-seed0`, attempt 1, seed 0, candidate role.
- Terminal outcome: completed with exit code 0 at 2026-07-28T05:31:48.591380+00:00; terminal event `93f53914-87c3-4b59-a9ea-21843765a78c` was accepted once through `turn/steer`.
- Endpoint quality: peak online-probe validation accuracy 0.845200 and final accuracy 0.836800 at optimizer step 17,400. Baseline-relative convergence targets, AUC values, and promotion status remain unavailable until the fresh baseline is terminal.
- Cost: cumulative active time at the final step was 9,667.947 seconds. The isolated one-forward predictor path executed 16 target queries with 9,633,152 parameters at 17.862656 ms median and 18.526209 ms p90 on an RTX 3090.
- CLS diagnostics: final CLS auxiliary shuffle gap was 1.752893 and the global CLS shuffle gap was 1.624766, confirming that the CLS representation carried cross-sample-specific information. The joint-context CLS shuffle gap was only 0.001310, so visual tokens still dominated when both sources were exposed in the diagnostic joint path.
- Provenance: parent `e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd`, mjepa `09fedcbb63b6959f3ec6db414faccd0c46a865dc`, vit `bf15705454975f04912538cdc790d399eea69e67`; online W&B run `00b899eb`.
- Artifacts: retained `checkpoint.pt` and `backbone.safetensors` under the exact managed run directory; no weights were deleted.
- Decision: keep pending until the fresh baseline and remaining seed-0 candidates complete.
<!-- autoresearch-operation:{"content_sha256":"e580f37c347d2bed7645942a67abc58d48ab382069525efbaf622037f53d03c6","operation_id":"65ffbaef9734cddf3f8a74126634f7f1"} -->

<!-- study:cls-routing-objective-v1:phase:screening -->
## cls-routing-objective-v1

- Question: Can a one-CLS, one-predictor-forward routed design recover the accepted four-partition baseline's accuracy or convergence while retaining a material predictor-cost advantage?
- Hypothesis: Separately normalizing CLS-only and visual-only routes will remove the current objective dilution; packed full coverage will best recover accuracy, while budget-preserving dual routing will retain the strongest latency advantage.
- Mechanisms and exact changes:
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, independently lift four disjoint CLS channel partitions, and use separate visual-context and CLS-context predictor forwards. Changes: not recorded.
  - `single-cls-joint-context-dual-routed`: Mechanism: Randomly split each image's target queries equally between visual-only and CLS-only visibility, compute one route mean per source, and sum the two means from one predictor tensor. Changes: Replace the 50/25/25 joint, CLS-only, and visual-only route mix with a 50/50 CLS-only and visual-only mix.; Replace the global coefficient of 2.0 with separately normalized visual and CLS route losses summed at coefficient 1.0.
  - `single-cls-joint-context-packed-dual-routed`: Mechanism: Duplicate all target queries inside one predictor call, blind the first copy to CLS and the second copy to visual context, and sum separately normalized route losses. Changes: Execute two predictor queries per spatial target but only one predictor forward.; Preserve exact baseline-like visual and CLS target coverage without a learned CLS expansion.
  - `single-cls-joint-context-token-routed-source-balanced`: Mechanism: Retain 50 percent joint, 25 percent CLS-only, and 25 percent visual-only routing, then add log of the visible-visual-key count to the CLS attention logit only on joint queries. Changes: Preserve the prior token-routed query budget and loss coefficient.; Apply a dynamic log visual-context-count bias to the CLS key on joint routes while preserving hard negative-infinity source blindness.
- Launch code provenance:
  - `pretrain-single-cls-joint-context-dual-routed-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-packed-dual-routed-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`: parent=`744358964b46e8a5d2f2c0340d5de63352b3d943` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: screening
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/summary.json`; external_detail=True
- Conclusion: Seed-0 screening is still running.
- Follow-up: complete the preregistered seed-0 screening trials.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-joint-context-dual-routed-seed0`: attempt=1; status=completed; decision=pending; started=2026-07-28T02:50:23.609113+00:00; finished=2026-07-28T05:31:48.591380+00:00; terminal_event=93f53914-87c3-4b59-a9ea-21843765a78c; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-dual-routed-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/00b899eb); checkpoint=retained; metrics=peak_accuracy=0.845200, final_accuracy=0.836800, step_to_90=8700, step_to_95=censored, active_seconds_to_90=4836.308, active_seconds_to_95=censored, step_auc=0.744367, active_time_auc=0.744120, active_seconds_at_step_horizon=9667.947, cls_path_latency_median_ms=17.862656, cls_path_latency_p90_ms=18.526209; error=none
- `pretrain-single-cls-joint-context-packed-dual-routed-seed0`: attempt=1; status=running; decision=pending; started=2026-07-28T05:32:31.929243+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-packed-dual-routed-seed0`; W&B=offline/unlinked (`bbe667e8`); checkpoint=retained; metrics=unavailable; error=none
- `pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`: attempt=1; status=running; decision=pending; started=2026-07-28T05:57:05.199475+00:00; finished=unknown; terminal_event=unknown; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`; W&B=offline/unlinked (`e2893098`); checkpoint=retained; metrics=unavailable; error=none
- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=completed; decision=pending; started=2026-07-28T02:50:23.538126+00:00; finished=2026-07-28T05:56:26.741387+00:00; terminal_event=9e2716b1-aca9-4fe0-8a41-3441f676ad8c; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/07ae3a53); checkpoint=retained; metrics=peak_accuracy=0.898800, final_accuracy=0.898200, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3906.090, active_seconds_to_95=5018.652, step_auc=0.794415, active_time_auc=0.778385, active_seconds_at_step_horizon=11143.876, cls_path_latency_median_ms=31.421951, cls_path_latency_p90_ms=32.291840; error=none
<!-- autoresearch-operation:{"content_sha256":"840d3b66cb5892c4a1834a76b30327cb3c7bba79b3be5d8d473d1ea151daa54c","operation_id":"de609e21cd6bc3b5a58193abaf79522e"} -->

<!-- study:cls-routing-objective-v1:phase:no-promotion -->
## cls-routing-objective-v1

- Question: Can a one-CLS, one-predictor-forward routed design recover the accepted four-partition baseline's accuracy or convergence while retaining a material predictor-cost advantage?
- Hypothesis: Separately normalizing CLS-only and visual-only routes will remove the current objective dilution; packed full coverage will best recover accuracy, while budget-preserving dual routing will retain the strongest latency advantage.
- Mechanisms and exact changes:
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, independently lift four disjoint CLS channel partitions, and use separate visual-context and CLS-context predictor forwards. Changes: not recorded.
  - `single-cls-joint-context-dual-routed`: Mechanism: Randomly split each image's target queries equally between visual-only and CLS-only visibility, compute one route mean per source, and sum the two means from one predictor tensor. Changes: Replace the 50/25/25 joint, CLS-only, and visual-only route mix with a 50/50 CLS-only and visual-only mix.; Replace the global coefficient of 2.0 with separately normalized visual and CLS route losses summed at coefficient 1.0.
  - `single-cls-joint-context-packed-dual-routed`: Mechanism: Duplicate all target queries inside one predictor call, blind the first copy to CLS and the second copy to visual context, and sum separately normalized route losses. Changes: Execute two predictor queries per spatial target but only one predictor forward.; Preserve exact baseline-like visual and CLS target coverage without a learned CLS expansion.
  - `single-cls-joint-context-token-routed-source-balanced`: Mechanism: Retain 50 percent joint, 25 percent CLS-only, and 25 percent visual-only routing, then add log of the visible-visual-key count to the CLS attention logit only on joint queries. Changes: Preserve the prior token-routed query budget and loss coefficient.; Apply a dynamic log visual-context-count bias to the CLS key on joint routes while preserving hard negative-infinity source blindness.
- Launch code provenance:
  - `pretrain-single-cls-joint-context-dual-routed-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-packed-dual-routed-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`: parent=`744358964b46e8a5d2f2c0340d5de63352b3d943` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`e9ea4d0eb35f9c9b20051d2f040d1f6751fc0abd` (`codex/research/cls-routing-objective-v1`), mjepa=`09fedcbb63b6959f3ec6db414faccd0c46a865dc` (`codex/research/cls-routing-objective-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-joint-context-dual-routed-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T02:50:23.609113+00:00; finished=2026-07-28T05:31:48.591380+00:00; terminal_event=93f53914-87c3-4b59-a9ea-21843765a78c; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-dual-routed-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/00b899eb); checkpoint=retained; metrics=peak_accuracy=0.845200, final_accuracy=0.836800, step_to_90=8700, step_to_95=censored, active_seconds_to_90=4836.308, active_seconds_to_95=censored, step_auc=0.744367, active_time_auc=0.744120, active_seconds_at_step_horizon=9667.947, cls_path_latency_median_ms=17.862656, cls_path_latency_p90_ms=18.526209; error=none
- `pretrain-single-cls-joint-context-packed-dual-routed-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T05:32:32.098157+00:00; finished=2026-07-28T08:31:37.746160+00:00; terminal_event=eaa86d3c-2f8d-424a-8620-e4506171a62a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-packed-dual-routed-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/bbe667e8); checkpoint=retained; metrics=peak_accuracy=0.852400, final_accuracy=0.846000, step_to_90=7830, step_to_95=censored, active_seconds_to_90=4834.880, active_seconds_to_95=censored, step_auc=0.748325, active_time_auc=0.737120, active_seconds_at_step_horizon=10723.606, cls_path_latency_median_ms=28.362752, cls_path_latency_p90_ms=29.802431; error=none
- `pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T05:57:05.366276+00:00; finished=2026-07-28T08:39:48.652541+00:00; terminal_event=30b0f958-fbfe-4c72-8968-c856d963b1d3; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-joint-context-token-routed-source-balanced-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/e2893098); checkpoint=retained; metrics=peak_accuracy=0.869600, final_accuracy=0.866800, step_to_90=6960, step_to_95=10440, active_seconds_to_90=3906.250, active_seconds_to_95=5851.776, step_auc=0.763940, active_time_auc=0.762500, active_seconds_at_step_horizon=9743.723, cls_path_latency_median_ms=18.356736, cls_path_latency_p90_ms=18.955263; error=none
- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T02:50:23.538126+00:00; finished=2026-07-28T05:56:26.741387+00:00; terminal_event=9e2716b1-aca9-4fe0-8a41-3441f676ad8c; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-routing-objective-v1/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/07ae3a53); checkpoint=retained; metrics=peak_accuracy=0.898800, final_accuracy=0.898200, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3906.090, active_seconds_to_95=5018.652, step_auc=0.794415, active_time_auc=0.778385, active_seconds_at_step_horizon=11143.876, cls_path_latency_median_ms=31.421951, cls_path_latency_p90_ms=32.291840; error=none
<!-- autoresearch-operation:{"content_sha256":"fbdd23772fea60837f71058b6813974e9e16548582f775a947f3c4f19a61d913","operation_id":"3c9dbb70bd819df9922c06e0cfd145aa"} -->

<!-- study:cls-adaln-hard-blind-v2-smoke:phase:no-promotion -->
## cls-adaln-hard-blind-v2-smoke

- Question: Can packed hard-blind AdaLN predict every teacher target once from visual context and once from one CLS token in one managed predictor forward without source leakage?
- Hypothesis: The one-epoch shared-core path will produce two predictions per target, keep blind-query gradients independent of student visual tokens, give nonzero CLS gradients, emit finite route-normalized losses and diagnostics, checkpoint, recover, summarize, and notify.
- Mechanisms and exact changes:
  - `cls-packed-adaln-hard-blind-smoke`: Mechanism: For every target position, run one query through visual cross-attention with null conditioning and a duplicate query through the same AdaLN MLP with one student CLS embedding while skipping the blind cross-attention residual exactly. Changes: Preserve one backbone CLS token plus seven register tokens.; Execute one predictor forward with one visual and one blind prediction per spatial target.; Initialize AdaLN modulation weights at standard deviation 0.001 and its residual gate at 1.0.; Sum separately normalized visual-route and blind-route prediction losses.
- Launch code provenance:
  - `pretrain-cls-packed-adaln-hard-blind-smoke-seed0`: parent=`da0bb15823659ea3b1ff02a79277452b6aab40cc` (`codex/research/cls-adaln-hard-blind-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-packed-adaln-hard-blind-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T14:36:35.343506+00:00; finished=2026-07-28T14:39:04.593056+00:00; terminal_event=48cae450-9ce5-4b2b-a8bc-1ec258635dd5; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2-smoke/runs/pretrain-cls-packed-adaln-hard-blind-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/6b57965f); checkpoint=retained; metrics=peak_accuracy=0.220800, final_accuracy=0.220800, step_to_90=2812, step_to_95=2812, active_seconds_to_90=133.048, active_seconds_to_95=133.048, step_auc=0.220800, active_time_auc=0.220800, active_seconds_at_step_horizon=133.048, cls_path_latency_median_ms=2.638848, cls_path_latency_p90_ms=2.664448; error=none
<!-- autoresearch-operation:{"content_sha256":"acb2472ee1a363fac3a1bc5d6f6d54ea8f9fce54ee053ea7c3efd8f4a1e3e645","operation_id":"eaccf09ef270d230907f4762d3c0c797"} -->

<!-- study:cls-adaln-hard-blind-v2:phase:no-promotion -->
## cls-adaln-hard-blind-v2

- Question: Can a one-CLS, one-predictor-forward hard-blind AdaLN interface match the accepted independent four-partition baseline's accuracy or convergence while reducing predictor cost?
- Hypothesis: Packing full visual and blind target coverage into a hard-separated AdaLN predictor will recover accuracy lost by routed cross-attention; a blind-only low-rank adapter or one-layer token mixer will recover any remaining decoder-interface deficit without exposing student visual tokens to CLS optimization.
- Mechanisms and exact changes:
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, independently lift four disjoint CLS channel partitions, and use separate visual-context and CLS-context predictor forwards. Changes: not recorded.
  - `single-cls-packed-adaln-hard-blind`: Mechanism: Run visual query copies through ordinary cross-attention with null AdaLN conditioning and blind copies through the same predictor MLPs conditioned on one CLS token while exactly skipping blind cross-attention. Changes: Replace the separate four-partition CLS predictor replay with packed visual and blind target copies.; Initialize AdaLN modulation weights at standard deviation 0.001 and its residual gate at 1.0 rather than AdaLN-Zero.; Sum separately normalized visual and blind losses.
  - `single-cls-packed-adaln-hard-blind-adapter`: Mechanism: Preserve the shared hard-blind path and add one RMS-normalized low-rank residual adapter per predictor block only to blind queries. Changes: Add blind-only 384-to-96-to-384 residual adapters after the shared MLP blocks.; Preserve exact visual-token blindness, route coverage, AdaLN initialization, and one predictor forward.
  - `single-cls-packed-adaln-hard-blind-mixer`: Mechanism: Preserve the shared hard-blind path and apply one positional self-attention token-mixing block only to blind target queries before the shared output projection. Changes: Add one blind-only target-query transformer encoder block.; Preserve exact visual-token blindness, route coverage, AdaLN initialization, and one predictor forward.
- Launch code provenance:
  - `pretrain-single-cls-packed-adaln-hard-blind-adapter-seed0`: parent=`02e2b06c5f00c10099fd067259d5da3ffe8ec3f8` (`codex/research/cls-adaln-hard-blind-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-packed-adaln-hard-blind-mixer-seed0`: parent=`02e2b06c5f00c10099fd067259d5da3ffe8ec3f8` (`codex/research/cls-adaln-hard-blind-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-packed-adaln-hard-blind-seed0`: parent=`02e2b06c5f00c10099fd067259d5da3ffe8ec3f8` (`codex/research/cls-adaln-hard-blind-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`02e2b06c5f00c10099fd067259d5da3ffe8ec3f8` (`codex/research/cls-adaln-hard-blind-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-single-cls-packed-adaln-hard-blind-adapter-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T17:44:50.555002+00:00; finished=2026-07-28T20:44:56.274664+00:00; terminal_event=bc388bf0-e5db-42d6-a8dd-816f7ff04b1b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2/runs/pretrain-single-cls-packed-adaln-hard-blind-adapter-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/aa3604db); checkpoint=retained; metrics=peak_accuracy=0.798200, final_accuracy=0.797200, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.718728, active_time_auc=0.717628, active_seconds_at_step_horizon=10786.345, cls_path_latency_median_ms=28.273664, cls_path_latency_p90_ms=29.798401; error=none
- `pretrain-single-cls-packed-adaln-hard-blind-mixer-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T17:52:08.106346+00:00; finished=2026-07-28T20:58:46.930892+00:00; terminal_event=7dd5f5fd-ab7c-4d40-b0f9-defc71ccff0b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2/runs/pretrain-single-cls-packed-adaln-hard-blind-mixer-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/519aa786); checkpoint=retained; metrics=peak_accuracy=0.804600, final_accuracy=0.797800, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.728073, active_time_auc=0.723798, active_seconds_at_step_horizon=11168.953, cls_path_latency_median_ms=31.689728, cls_path_latency_p90_ms=32.854015; error=none
- `pretrain-single-cls-packed-adaln-hard-blind-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T14:43:52.632183+00:00; finished=2026-07-28T17:41:59.029695+00:00; terminal_event=7517901a-73da-4730-a821-eaed1f3525bd; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2/runs/pretrain-single-cls-packed-adaln-hard-blind-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/f0c83c79); checkpoint=retained; metrics=peak_accuracy=0.789000, final_accuracy=0.786000, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.723185, active_time_auc=0.722919, active_seconds_at_step_horizon=10667.464, cls_path_latency_median_ms=27.139584, cls_path_latency_p90_ms=28.783615; error=none
- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T14:43:52.565708+00:00; finished=2026-07-28T17:50:22.965504+00:00; terminal_event=a1059d75-6547-4aca-af2c-a645c3606aa1; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-adaln-hard-blind-v2/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/994d9d19); checkpoint=retained; metrics=peak_accuracy=0.898800, final_accuracy=0.898200, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3905.239, active_seconds_to_95=5019.054, step_auc=0.794415, active_time_auc=0.789581, active_seconds_at_step_horizon=11149.680, cls_path_latency_median_ms=31.489536, cls_path_latency_p90_ms=32.279552; error=none
<!-- autoresearch-operation:{"content_sha256":"6ac799610b406294678c15b93739ceedd8741f1c6bbe233ab04c0b040868bd5a","operation_id":"f7d210f601b5f84947e063794aaaaa40"} -->

<!-- study:cls-teacher-global-v2-smoke:phase:no-promotion -->
## cls-teacher-global-v2-smoke

- Question: Can a centered normalized EMA-attention teacher-global target train one CLS token without leaking teacher or student visual features into its direct optimization path?
- Hypothesis: The one-epoch smoke run will give gradients to the student CLS and online attention pooler, keep the teacher backbone and target pooler stopped, update the target pooler by EMA, emit finite direct and visible-pool losses plus noncollapsed target diagnostics, checkpoint, recover, summarize, and notify.
- Mechanisms and exact changes:
  - `cls-global-ema-attention-smoke`: Mechanism: Center and normalize a one-query EMA attention pool of all teacher visual tokens, regress both the student CLS and an online attention pool of visible student tokens to that stopped target, and retain packed hard-blind AdaLN patch prediction. Changes: Preserve one backbone CLS token plus seven register tokens and one patch-predictor forward.; Add direct CLS and visible-pool squared-L2 losses at weight 0.1 each.; Update the target attention pooler with the same EMA momentum as the backbone teacher.
- Launch code provenance:
  - `pretrain-cls-global-ema-attention-smoke-seed0`: parent=`c7fd7da6532a208e3afdeadb46f15255aa40c0b5` (`codex/research/cls-teacher-global-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-global-ema-attention-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T21:24:13.457745+00:00; finished=2026-07-28T21:27:44.058521+00:00; terminal_event=5d86fc9d-2679-4d08-a955-bc554e601eb8; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2-smoke/runs/pretrain-cls-global-ema-attention-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/5421bf8f); checkpoint=retained; metrics=peak_accuracy=0.198000, final_accuracy=0.198000, step_to_90=2812, step_to_95=2812, active_seconds_to_90=152.399, active_seconds_to_95=152.399, step_auc=0.198000, active_time_auc=0.198000, active_seconds_at_step_horizon=152.399, cls_path_latency_median_ms=2.617856, cls_path_latency_p90_ms=2.647040; error=none
- Mechanical interpretation: Training, checkpointing, EMA updates, online-pooler gradients, lifecycle delivery, and both information boundaries were valid, but the learned target collapsed across samples. The final target channel standard deviation was `0.000237`, the CLS shuffle gap was `0.000028`, and a 128-sample checkpoint audit measured mean pairwise target cosine `1.000000`. The online and target poolers differed by total parameter L2 distance `0.257141`, so failure was not caused by a missing optimizer or EMA update.
- Counterfactual audit: Centered normalized fixed-mean targets from the same teacher features retained channel standard deviation `0.088919` and mean pairwise cosine `0.715758`. The scientific study was not launched. The follow-up will constrain learned attention to produce a convex weighted mean of raw teacher visual tokens, removing the learned value/output transform that created a sample-independent target.
<!-- autoresearch-operation:{"content_sha256":"35e7026e29037d198e8be619b889ba66c460bf092fee647bb94f35711f5a2330","operation_id":"8213ee2391b9393cf22c5b5be3d7b1bd"} -->

<!-- study:cls-teacher-global-v2-convex-smoke:phase:no-promotion -->
## cls-teacher-global-v2-convex-smoke

- Question: Does convex attention-weight pooling prevent the sample-independent collapse of the centered normalized EMA teacher-global target while preserving one CLS token and one patch-predictor forward?
- Hypothesis: Replacing learned value and output projections with learned query/key weights applied directly to raw visual tokens will raise final target channel standard deviation from 0.000237 to at least 0.005, reduce mean pairwise target cosine below 0.99, retain positive CLS and patch shuffle gaps, update every online pooler parameter and its EMA target, checkpoint, recover, summarize, and notify.
- Mechanisms and exact changes:
  - `cls-global-ema-attention-convex-smoke`: Mechanism: Center and normalize a convex attention-weighted mean of all teacher visual tokens, regress both the student CLS and a convex online pool of visible student tokens to that stopped target, and retain packed hard-blind AdaLN patch prediction. Changes: Preserve one backbone CLS token plus seven register tokens and one patch-predictor forward.; Learn one query, one key projection, and per-head QK normalization without learned value or output projections.; Average per-head softmax weights and apply them directly to raw visual tokens.; Add direct CLS and visible-pool squared-L2 losses at weight 0.1 each.; Update the target weight pooler with the same EMA momentum as the backbone teacher.
- Launch code provenance:
  - `pretrain-cls-global-ema-attention-convex-smoke-seed0`: parent=`35758fc8adf2ce75fd4d6adf772d4743a818b5e0` (`codex/research/cls-teacher-global-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2-convex-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-global-ema-attention-convex-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T21:37:46.041963+00:00; finished=2026-07-28T21:41:58.480914+00:00; terminal_event=aa173043-6a92-4db6-b63f-ce23bd1c0222; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2-convex-smoke/runs/pretrain-cls-global-ema-attention-convex-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/d2a65c16); checkpoint=retained; metrics=peak_accuracy=0.211600, final_accuracy=0.211600, step_to_90=2812, step_to_95=2812, active_seconds_to_90=135.614, active_seconds_to_95=135.614, step_auc=0.211600, active_time_auc=0.211600, active_seconds_at_step_horizon=135.614, cls_path_latency_median_ms=2.673664, cls_path_latency_p90_ms=2.707456; error=none
- Mechanical interpretation: The convex repair passed every preregistered smoke gate. The final target channel standard deviation was `0.065640`, mean pairwise target cosine was `0.840899`, the CLS true-versus-shuffled loss gap was `+0.240344`, and the hard-blind patch gap was `+0.488445`. These values preserve sample diversity and show that both prediction paths depend on their intended representations.
- Pooler audit: All six trainable online-pooler tensors moved relative to their EMA targets. The final normalized attention entropy was `0.685961`, maximum token weight was `0.299075`, and the direct convex weights remained nonnegative and unit-sum by construction and test. The preregistered four-run `cls-teacher-global-v2` scientific screen is approved to launch.
<!-- autoresearch-operation:{"content_sha256":"3cae77340e823ee552d137d2e83dffbb118dbd0f867e94871b885f00dbd45b91","operation_id":"3c2e9946b1f7bbc501fc035c3cb313ad"} -->

<!-- study:cls-teacher-global-v2:phase:no-promotion -->
## cls-teacher-global-v2

- Question: Can a centered normalized full-teacher target make one hard-blind CLS token match the accepted independent four-partition predictor while preserving one backbone CLS token and one predictor forward?
- Hypothesis: Removing raw target norm and common-mode information will avoid the representation shrinkage seen with direct raw-mean MSE. Fixed-mean or learned EMA-attention pooling will improve seed-0 peak validation accuracy by at least 0.05 over a fresh packed hard-blind control; a qualifying candidate will trail a fresh independent-partition baseline by no more than 0.005 while reducing common-step active time and isolated CLS-path latency by at least 5 percent.
- Mechanisms and exact changes:
  - `single-cls-register-partitioned-independent`: Mechanism: Preserve one CLS plus seven registers, independently lift four disjoint CLS channel partitions, and use separate visual-context and CLS-context predictor forwards. Changes: not recorded.
  - `single-cls-packed-adaln-hard-blind-control`: Mechanism: Use packed visual and hard-blind CLS-conditioned target copies without any teacher-global objective. Changes: Preserve one CLS plus seven registers and one predictor forward.; Add no teacher-global target or pooling module.
  - `cls-global-centered-mean`: Mechanism: Regress the centered normalized student CLS and visible-token mean to the stopped centered normalized mean of all full-image teacher visual tokens. Changes: Preserve the packed hard-blind architecture and patch losses.; Add direct CLS and visible-pool squared-L2 losses at weight 0.1 each.; Add no trainable pooling parameters.
  - `cls-global-centered-ema-attention`: Mechanism: Regress the centered normalized student CLS and an online attention-weighted mean of visible student tokens to a stopped centered normalized EMA attention-weighted mean of every full-image teacher visual token. Changes: Preserve the packed hard-blind architecture and patch losses.; Add direct CLS and visible-pool squared-L2 losses at weight 0.1 each.; Learn only multi-head query/key attention weights, apply their head average directly to raw visual tokens, and update the target copy with the backbone teacher momentum.
- Launch code provenance:
  - `pretrain-cls-global-centered-ema-attention-seed0`: parent=`8e980b01e53247c900db3528fd6d4732d616e2fc` (`codex/research/cls-teacher-global-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-cls-global-centered-mean-seed0`: parent=`e60300ebbce5ae75a9314927225f35e8e60a78cb` (`codex/research/cls-teacher-global-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-packed-adaln-hard-blind-control-seed0`: parent=`e60300ebbce5ae75a9314927225f35e8e60a78cb` (`codex/research/cls-teacher-global-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-single-cls-register-partitioned-independent-seed0`: parent=`e60300ebbce5ae75a9314927225f35e8e60a78cb` (`codex/research/cls-teacher-global-v2`), mjepa=`d17ec9e937a61022b9bc6d8676082226bfda7f94` (`codex/research/cls-adaln-hard-blind-v2`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-cls-global-centered-ema-attention-seed0`: attempt=2; status=completed; decision=rejected; started=2026-07-29T00:58:42.405080+00:00; finished=2026-07-29T04:02:34.296266+00:00; terminal_event=982aa343-3b00-4e07-9553-7939ca39a423; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2/runs/pretrain-cls-global-centered-ema-attention-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/ce3e90a2); checkpoint=retained; metrics=peak_accuracy=0.753200, final_accuracy=0.753200, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.627650, active_time_auc=0.623481, active_seconds_at_step_horizon=11013.451, cls_path_latency_median_ms=27.815424, cls_path_latency_p90_ms=29.106176; error=none
- `pretrain-cls-global-centered-mean-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-29T00:43:52.878867+00:00; finished=2026-07-29T03:42:56.918261+00:00; terminal_event=f945b87b-bf98-404b-b081-b94084aeef2a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2/runs/pretrain-cls-global-centered-mean-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/ad143833); checkpoint=retained; metrics=peak_accuracy=0.749600, final_accuracy=0.749600, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.609355, active_time_auc=0.608318, active_seconds_at_step_horizon=10725.078, cls_path_latency_median_ms=27.402752, cls_path_latency_p90_ms=28.691456; error=none
- `pretrain-single-cls-packed-adaln-hard-blind-control-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-28T21:45:21.253099+00:00; finished=2026-07-29T00:43:24.197328+00:00; terminal_event=b67ab5be-a7e9-4ab4-854d-e8341f0fd9e0; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2/runs/pretrain-single-cls-packed-adaln-hard-blind-control-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/a86d6def); checkpoint=retained; metrics=peak_accuracy=0.789000, final_accuracy=0.786000, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.723185, active_time_auc=0.723019, active_seconds_at_step_horizon=10664.064, cls_path_latency_median_ms=27.000833, cls_path_latency_p90_ms=28.887039; error=none
- `pretrain-single-cls-register-partitioned-independent-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-28T21:45:21.177663+00:00; finished=2026-07-29T00:52:03.183807+00:00; terminal_event=fd58a496-24da-4378-ba07-5bb82be30ed6; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/cls-teacher-global-v2/runs/pretrain-single-cls-register-partitioned-independent-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c03c5a38); checkpoint=retained; metrics=peak_accuracy=0.898800, final_accuracy=0.898200, step_to_90=6090, step_to_95=7830, active_seconds_to_90=3914.960, active_seconds_to_95=5032.607, step_auc=0.794415, active_time_auc=0.789285, active_seconds_at_step_horizon=11180.176, cls_path_latency_median_ms=31.500800, cls_path_latency_p90_ms=32.129089; error=none
- Gate interpretation: Both teacher-global candidates passed the one-CLS, seven-register, one-forward, stopped-target, target-diversity, and positive-shuffle mechanical gates. Neither passed the required `+0.05` peak-accuracy gain over the `0.7890` hard-blind control: fixed mean reached `0.7496` (`-0.0394`) and convex EMA attention reached `0.7532` (`-0.0358`). They also missed the baseline quality floor by `0.1492` and `0.1456`, respectively. Fixed mean improved common-step active time by only `4.07%` and convex attention by `1.49%`, below the `5%` cost gate, although their isolated median latency gains were `13.01%` and `11.70%`.
- Mechanism interpretation: The objectives successfully forced sample identity through the blinded CLS path, but the resulting information was not useful enough for the online classifier. Final target standard deviations were `0.015653` and `0.015159`, pairwise target cosines were `0.034937` and `0.022283`, and CLS shuffle gaps were `+1.727306` and `+1.718045` for mean and convex pooling. At the same time, final visual-only loss rose from `0.080432` in the hard-blind control to `0.268799` and `0.228470`, while true hard-blind patch loss rose from `0.092011` to `0.293578` and `0.250083`. This indicates objective interference with the shared backbone and patch-prediction task rather than target collapse or information-boundary leakage.
- Pooler interpretation: Convex EMA attention produced only a `+0.0036` peak-accuracy gain over fixed mean while costing more active time and latency. Its final normalized attention entropy was `0.850361` and maximum token weight was `0.067447`, so it remained broadly distributed rather than isolating a small set of globally useful patches. The learned weighting did not change the scientific conclusion.
- Recovery note: The convex run's first managed attempt stopped before W&B initialization or any optimizer step because its config-derived tag exceeded W&B's 64-character limit. The bounded-tag repair and immutable-authority retry repair were regression-tested and pushed before attempt 2 reused the same W&B ID and original wake context. This infrastructure-only retry did not alter the scientific configuration.
- Decision: Reject centered normalized full-teacher pooling as the next route for closing the independent-partition gap at the tested weights and interfaces. Do not launch confirmation or supervised evaluation. Retain all four checkpoints and the archived failed-attempt artifacts.

<!-- autoresearch-operation:{"operation_id":"cls-token-line-closeout-default-adoption-v1"} -->
## 2026-07-29 closeout: adopt four independent CLS partitions

- Scope: close the CLS-token experiment line and select the repository default. This is an engineering adoption decision authorized by the user, not a retrospective claim that the candidate passed the preregistered strict four-CLS promotion gate.
- Selected architecture: one backbone CLS token, seven register tokens, `partitioned_independent_cross_attention`, and four 96-to-384 independent lifts from disjoint CLS channel partitions. The visual-context and blinded CLS-context predictor passes remain separate.
- Selection evidence:

  | Design | Peak accuracy | Step to 95% target | Step AUC | Active-time AUC | Decision |
  |---|---:|---:|---:|---:|---|
  | Four backbone CLS tokens, fixed reference | 0.910000 | 7,830 | 0.801230 | 0.796583 | Historical reference |
  | Two independent partitions | 0.896200 | 11,745 | 0.781475 | 0.780721 | Reject |
  | Four independent partitions | 0.898000 | 10,005 | 0.787907 | 0.787593 | Adopt as default |
  | Eight independent partitions | 0.842600 | censored | 0.734497 | 0.734240 | Reject |

- Robustness evidence: fresh four-partition controls in the routing, hard-blind AdaLN, and teacher-global follow-ups reached `0.898800` peak and `0.898200` final accuracy. Their step-to-95 value was `7,830`; common-step active time ranged from `11,143.876` to `11,180.176` seconds. These repeated seed-0 controls support the practical stability of the selected design but do not replace paired multi-seed confirmation.
- Rationale: four partitions preserve most of the four-CLS quality while simplifying the backbone to the standard single-CLS layout. Two partitions retained endpoint quality but converged more slowly, and eight partitions over-fragmented the embedding. Every tested one-forward routed or AdaLN alternative lost substantially more accuracy.
- Configuration migration: `config/pretrain/vit-small.yaml` and `config/finetune/vit-small.yaml` now select the one-CLS, seven-register layout. Completed studies retain the former defaults through explicit `vit-small-four-cls-legacy.yaml` configs.
- Limitations: all architecture selection evidence is seed 0, the selected design did not pass the original strict equivalence thresholds, and no new supervised or official-test evaluation was run for this closeout.
- Retention: preserve all previously retained checkpoints and backbones. No artifact deletion is authorized by this closeout.
<!-- autoresearch-operation:{"content_sha256":"940c18db9a2c982ef085406c9bc40ddb7944e028bc801b6173e521c5063e13c7","operation_id":"6b1cc5fa99ade5f88ed4c44ce142f72b"} -->

<!-- study:lejepa-masked-collapse-v1-smoke:phase:no-promotion -->
## lejepa-masked-collapse-v1-smoke

- Question: Does the shared-student masked-target LeJEPA path complete one full train-validation-checkpoint cycle with SigREG, its projector, and a stopped online probe?
- Hypothesis: The shared full-view target will remain differentiable for the masked prediction and SigREG losses while the detached probe updates only its classifier head.
- Mechanisms and exact changes:
  - `projected-both-smoke`: Mechanism: Remove the EMA teacher, regularize projected masked and full CLS views with SigREG, retain masked image modeling and the auxiliary partitioned CLS predictor, and detach probe inputs. Changes: Use the student for both masked context and full-view target encoding.; Apply the convex LeJEPA objective with lambda 0.05 and a 2048-2048-64 projector.; Stop probe gradients at the full-view student CLS token.
- Launch code provenance:
  - `pretrain-projected-both-smoke-seed0`: parent=`d4d577c0780dc01c0e509576deb5f5ae858329c1` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`e0bb8c22feb40eb284abdd14b65006c1a09e2af2` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-projected-both-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-30T14:44:16.725726+00:00; finished=2026-07-30T14:46:14.234738+00:00; terminal_event=cca984b9-68c1-408f-9b91-ada6382a4c90; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-smoke/runs/pretrain-projected-both-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/5d88497e); checkpoint=retained; metrics=peak_accuracy=0.167600, final_accuracy=0.167600, step_to_90=44, step_to_95=44, active_seconds_to_90=69.738, active_seconds_to_95=69.738, step_auc=0.167600, active_time_auc=0.167600, active_seconds_at_step_horizon=69.738, cls_path_latency_median_ms=31.474688, cls_path_latency_p90_ms=31.978497; error=none
<!-- autoresearch-operation:{"content_sha256":"083a732372e06a35265a88ba16d91ee4851e4c5c2efbc1714789e4d3df001096","operation_id":"5604b1f053abb05fef43718e6bc1dab1"} -->

<!-- study:lejepa-masked-collapse-v1-feasibility:phase:no-promotion -->
## lejepa-masked-collapse-v1-feasibility

- Question: Which masked-image shared-target SigREG design prevents collapse without an EMA teacher or self-supervised stop-gradient?
- Hypothesis: SigREG on a low-dimensional projector will keep both CLS and patch representations noncollapsed while the masked prediction task preserves image-specific structure.
- Mechanisms and exact changes:
  - `shared-nosigreg`: Mechanism: Use the shared full-view student target with masked visual and auxiliary CLS prediction, but no explicit distribution regularizer. Changes: Remove the EMA teacher and Gram anchoring.; Preserve masked image modeling and four independent CLS predictor partitions.; Set SigREG weight to zero.
  - `direct-both-l005`: Mechanism: Apply the convex LeJEPA objective with lambda 0.05 directly to both CLS views. Changes: Add direct SIGReg to masked-context and full-target CLS embeddings.; Keep the auxiliary partitioned CLS prediction task.
  - `proj64-both-l005`: Mechanism: Project both masked-context and full-target CLS embeddings through 384-2048-2048-64 before SigREG. Changes: Add a trainable three-layer SigREG projector.; Apply SIGReg to both projected views with lambda 0.05.
  - `proj64-target-l005`: Mechanism: Apply projected SigREG only to the full-view student CLS with lambda 0.05. Changes: Retain the 384-2048-2048-64 projector.; Remove the masked-context CLS from SIGReg.
- Launch code provenance:
  - `pretrain-direct-both-l005-seed0`: parent=`c91d097f0e82527fe42d3cfc4fe2daba2c30f87c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`e0bb8c22feb40eb284abdd14b65006c1a09e2af2` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-proj64-both-l005-seed0`: parent=`c91d097f0e82527fe42d3cfc4fe2daba2c30f87c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`e0bb8c22feb40eb284abdd14b65006c1a09e2af2` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-proj64-target-l005-seed0`: parent=`c91d097f0e82527fe42d3cfc4fe2daba2c30f87c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`e0bb8c22feb40eb284abdd14b65006c1a09e2af2` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-shared-nosigreg-seed0`: parent=`c91d097f0e82527fe42d3cfc4fe2daba2c30f87c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`e0bb8c22feb40eb284abdd14b65006c1a09e2af2` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-feasibility/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-direct-both-l005-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T14:52:24.121878+00:00; finished=2026-07-30T15:58:44.215617+00:00; terminal_event=2f8c22b8-7adc-40e2-9137-4ae20490b645; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-feasibility/runs/pretrain-direct-both-l005-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/d7ddd08a); checkpoint=retained; metrics=peak_accuracy=0.168200, final_accuracy=0.091000, step_to_90=435, step_to_95=435, active_seconds_to_90=406.551, active_seconds_to_95=406.551, step_auc=0.110380, active_time_auc=0.110478, active_seconds_at_step_horizon=3960.106, cls_path_latency_median_ms=30.865920, cls_path_latency_p90_ms=31.887360; error=none
- `pretrain-proj64-both-l005-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T18:50:10.296893+00:00; finished=2026-07-30T19:57:39.304845+00:00; terminal_event=aa946d6e-d5a7-4bfb-94a8-1ae4c52b4cac; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-feasibility/runs/pretrain-proj64-both-l005-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/f5247bff); checkpoint=retained; metrics=peak_accuracy=0.100400, final_accuracy=0.100000, step_to_90=435, step_to_95=435, active_seconds_to_90=412.074, active_seconds_to_95=412.074, step_auc=0.100020, active_time_auc=0.100020, active_seconds_at_step_horizon=4020.215, cls_path_latency_median_ms=31.438848, cls_path_latency_p90_ms=32.409599; error=none
- `pretrain-proj64-target-l005-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T18:50:10.369558+00:00; finished=2026-07-30T19:57:16.284875+00:00; terminal_event=9fc93eb5-58ed-4b24-b413-0d26bf67eac4; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-feasibility/runs/pretrain-proj64-target-l005-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/3d4e9d9a); checkpoint=retained; metrics=peak_accuracy=0.106800, final_accuracy=0.100000, step_to_90=435, step_to_95=435, active_seconds_to_90=406.182, active_seconds_to_95=406.182, step_auc=0.098400, active_time_auc=0.098392, active_seconds_at_step_horizon=3976.001, cls_path_latency_median_ms=30.847487, cls_path_latency_p90_ms=31.624191; error=none
- `pretrain-shared-nosigreg-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-30T14:52:24.052011+00:00; finished=2026-07-30T15:58:46.880085+00:00; terminal_event=29f732f5-8a36-4d03-a971-c4d93ca9f34b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-feasibility/runs/pretrain-shared-nosigreg-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/ea254dc1); checkpoint=retained; metrics=peak_accuracy=0.100000, final_accuracy=0.100000, step_to_90=435, step_to_95=435, active_seconds_to_90=407.529, active_seconds_to_95=407.529, step_auc=0.100000, active_time_auc=0.100000, active_seconds_at_step_horizon=3962.771, cls_path_latency_median_ms=31.417344, cls_path_latency_p90_ms=32.247807; error=none
<!-- autoresearch-operation:{"content_sha256":"aee0734eed184375f2fa4937dece733357b3ae20e430f22f88ada33c789bb165","operation_id":"lejepa-masked-collapse-v1-feasibility-interpretation-v1"} -->

- Collapse-gate interpretation: Every run failed the preregistered last-three-validation gate. The no-SigREG control ended at chance (`0.1000` peak/final), with minimum target-CLS and patch-mean standard deviations of `0.001402` and `0.002094`. Direct both-view SigREG produced only a transient probe improvement (`0.1682` peak, `0.0910` final); over the last three validations its minimum target-CLS standard deviation was `0.028564`, minimum effective-rank fraction was `0.004157`, maximum pairwise cosine was `0.994848`, and minimum patch-mean standard deviation was `0.000152`. These miss the corresponding `0.10`, `0.10`, `0.90`, and `0.10` thresholds.
- Projector interpretation: Neither projected variant regularized its own output distribution successfully. Target-only projected SigREG had last-three minima of `0.005462` projected standard deviation and `0.067955` effective-rank fraction, with maximum top-eigenvalue fraction `0.612090`; both-view projected SigREG reached only `0.002841`, `0.074182`, and `0.424788`. The required values were standard deviation in `[0.50, 1.50]`, effective-rank fraction at least `0.50`, and top-eigenvalue fraction at most `0.20`. Both raw patch representations remained nearly sample-independent, and neither run exceeded `0.1068` probe accuracy.
- Mechanism decision: Reject CLS-only SigREG at lambda `0.05`, with or without the tested `384-2048-2048-64` projector. The masked predictor can minimize its objective while full-view patch means and CLS embeddings become nearly constant, so regularizing only CLS views does not close the shortcut. The next masked-only screen will target visual-token or pooled-patch distributions directly, test stronger SigREG weights, and remove the auxiliary CLS-prediction pressure in selected variants. Local/global crops remain deferred because untested masked-only mechanisms remain within the authorized budget.
- Retention: Keep all four checkpoints and backbones. No destructive retention was applied.
<!-- autoresearch-operation:{"content_sha256":"3c055d010e9208d85212e700fd262888fa2223fc86faf139568d4148a22a37b6","operation_id":"lejepa-masked-collapse-v1-patch-screen-protocol-v1"} -->

## 2026-07-30 protocol: LeJEPA masked patch-aware collapse screen

- Study: `lejepa-masked-collapse-v1-patch-screen`.
- Question: Can direct SigREG on both CLS and per-image patch means prevent deterministic evaluation collapse while keeping the shared-student masked image modeling task?
- Fixed ladder: add patch-mean SigREG at lambda `0.05`; remove auxiliary CLS prediction; raise lambda to `0.20`; then remove attention dropout, hidden dropout, and stochastic depth.
- Eligibility: every one of the last three validation records must satisfy the preregistered CLS, patch-mean, visual-shuffle, finite-value, and `0.50` peak-probe thresholds in the committed study specification.
- Allocation: four seed-0 runs now; two masked-tuning runs remain; six paired-confirmation and six crop-fallback runs remain from the user-authorized 22-run program. No crop views are introduced in this screen.
- Provenance: parent commit is resolved at launch; `mjepa=d024b0caced600d059f22eec146339aeec600ff5`; `vit=bf15705454975f04912538cdc790d399eea69e67`; fixed CIFAR-10 45,000/5,000 split; W&B destination `tidalpaladin/mjepa-cifar10`.
- Retention: preserve every checkpoint and backbone. Destructive retention remains unauthorized.
<!-- autoresearch-operation:{"content_sha256":"768aa7fccc616af3e19541d652f1053db6a455e0cb4c0e64e179cb4121d39110","operation_id":"d927958cfb7fcd8a1f799025a983a022"} -->

<!-- study:lejepa-masked-collapse-v1-patch-smoke:phase:no-promotion -->
## lejepa-masked-collapse-v1-patch-smoke

- Question: Does direct CLS-plus-patch-mean SigREG complete one deterministic shared-target masked train-validation-checkpoint cycle?
- Hypothesis: The patch-aware loss will preserve shared-target gradients, update the predictor and student, keep the probe boundary detached, and emit finite collapse diagnostics.
- Mechanisms and exact changes:
  - `clspatch-deterministic-smoke`: Mechanism: Apply direct SigREG to context and target CLS embeddings and patch means at lambda 0.20 without auxiliary CLS prediction or stochastic regularizers. Changes: Add pooled visual features to the SigREG input.; Disable auxiliary CLS prediction.; Remove attention dropout, hidden dropout, and stochastic depth.
- Launch code provenance:
  - `pretrain-clspatch-deterministic-smoke-seed0`: parent=`1f1494b2ad070ce75b650827ef3c28eb799e4325` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-clspatch-deterministic-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-30T20:21:14.577107+00:00; finished=2026-07-30T20:22:54.549269+00:00; terminal_event=275a4d96-5cb2-4e15-bb9f-914294ed0584; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-smoke/runs/pretrain-clspatch-deterministic-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/050cd14c); checkpoint=retained; metrics=peak_accuracy=0.204200, final_accuracy=0.204200, step_to_90=44, step_to_95=44, active_seconds_to_90=74.678, active_seconds_to_95=74.678, step_auc=0.204200, active_time_auc=0.204200, active_seconds_at_step_horizon=74.678, cls_path_latency_median_ms=31.202816, cls_path_latency_p90_ms=31.998976; error=none
<!-- autoresearch-operation:{"content_sha256":"39350344dfafdebdc3b5333101af30f49117a6c7bc9e8fc9e058c76461980461","operation_id":"lejepa-masked-collapse-v1-patch-smoke-interpretation-v1"} -->

- Mechanical interpretation: The patch-aware deterministic shared-target path passed its one-epoch contract. Training, validation, checkpointing, online W&B tracking, first-cycle delivery, terminal delivery, and retained artifact recovery completed with finite metrics and exit code `0`.
- Representation evidence: At step `44`, target CLS and patch-mean standard deviations were `0.782529` and `0.632348`; mean pairwise cosines were `0.020554` and `0.021325`; top-eigenvalue fractions were `0.222952` and `0.266698`; visual-target shuffled relative improvement was `0.413107`; and validation probe accuracy was `0.204200`. These are not promotion evidence, but they reject immediate scalar and directional collapse seen in the CLS-only feasibility screen.
- Limitation and decision: One epoch cannot establish the preregistered last-three-validation effective-rank or `0.50` probe gates; effective-rank fractions were only `0.022786` for CLS and `0.016760` for patch means. Approve the committed four-run 100-epoch patch-aware screen without changing its hypotheses, metrics, or masks.
- Retention: Keep the checkpoint and backbone. No destructive retention was applied.
<!-- autoresearch-operation:{"content_sha256":"41ed9c4d15e1a46e02f88ae3d494f9b91c055d538079a75b532e5fd523f28ae7","operation_id":"e777a0ab08acdc9015bc10dd0b6714fd"} -->

<!-- study:lejepa-masked-collapse-v1-patch-screen:phase:no-promotion -->
## lejepa-masked-collapse-v1-patch-screen

- Question: Which direct CLS-plus-patch-mean SigREG repair prevents deterministic evaluation collapse while preserving masked image prediction and a shared student target?
- Hypothesis: Applying SigREG directly to both CLS and pooled visual features will close the patch-collapse shortcut; removing auxiliary CLS prediction, increasing lambda, or removing stochastic regularizers will identify the remaining source of collapse one mechanism at a time.
- Mechanisms and exact changes:
  - `clspatch-both-l005-aux`: Mechanism: Apply direct SigREG to context and target CLS embeddings and patch means at lambda 0.05. Changes: Add the mean visual-token embedding from each masked and full student view to SigREG.; Preserve the four-partition auxiliary CLS prediction task.; Preserve the feasibility screen's stochastic regularizers and lambda.
  - `clspatch-both-l005-noaux`: Mechanism: Remove auxiliary CLS prediction while retaining masked visual prediction and direct CLS-plus-patch SigREG at lambda 0.05. Changes: Disable the auxiliary CLS predictor and its loss.
  - `clspatch-both-l020-noaux`: Mechanism: Raise the convex SigREG weight from 0.05 to 0.20 after removing auxiliary CLS prediction. Changes: Change only LeJEPA lambda from 0.05 to 0.20.
  - `clspatch-both-l020-noaux-deterministic`: Mechanism: Remove attention dropout, hidden dropout, and stochastic depth from the stronger no-auxiliary design. Changes: Set backbone and inherited predictor attention dropout, hidden dropout, and drop-path rate to zero.
- Launch code provenance:
  - `pretrain-clspatch-both-l005-aux-seed0`: parent=`53853c311f2cebca14aff0e66ebee9812f9d4355` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-clspatch-both-l005-noaux-seed0`: parent=`53853c311f2cebca14aff0e66ebee9812f9d4355` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-clspatch-both-l020-noaux-deterministic-seed0`: parent=`53853c311f2cebca14aff0e66ebee9812f9d4355` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-clspatch-both-l020-noaux-seed0`: parent=`53853c311f2cebca14aff0e66ebee9812f9d4355` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-screen/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-clspatch-both-l005-aux-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-30T20:24:28.954185+00:00; finished=2026-07-30T21:32:23.360714+00:00; terminal_event=e1e03e24-686c-419e-8692-27149d4fc0f1; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-screen/runs/pretrain-clspatch-both-l005-aux-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/b053b542); checkpoint=retained; metrics=peak_accuracy=0.305600, final_accuracy=0.305600, step_to_90=1740, step_to_95=4350, active_seconds_to_90=1609.226, active_seconds_to_95=4005.951, step_auc=0.272480, active_time_auc=0.268711, active_seconds_at_step_horizon=4005.951, cls_path_latency_median_ms=31.598592, cls_path_latency_p90_ms=32.045055; error=none
- `pretrain-clspatch-both-l005-noaux-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T20:24:29.029036+00:00; finished=2026-07-30T21:25:37.663278+00:00; terminal_event=e3bf2ebd-42e9-4068-96c9-302298aba728; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-screen/runs/pretrain-clspatch-both-l005-noaux-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/35722da3); checkpoint=retained; metrics=peak_accuracy=0.259000, final_accuracy=0.259000, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.240800, active_time_auc=0.240199, active_seconds_at_step_horizon=3597.883, cls_path_latency_median_ms=30.595073, cls_path_latency_p90_ms=31.486977; error=none
- `pretrain-clspatch-both-l020-noaux-deterministic-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T21:33:17.406831+00:00; finished=2026-07-30T22:31:40.132323+00:00; terminal_event=e131a8a9-af8e-4f51-a3f9-51ee19bd20ae; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-screen/runs/pretrain-clspatch-both-l020-noaux-deterministic-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/ef0707aa); checkpoint=retained; metrics=peak_accuracy=0.308400, final_accuracy=0.308400, step_to_90=2175, step_to_95=3915, active_seconds_to_90=1731.062, active_seconds_to_95=3106.060, step_auc=0.266460, active_time_auc=0.266325, active_seconds_at_step_horizon=3450.058, cls_path_latency_median_ms=30.769665, cls_path_latency_p90_ms=31.345663; error=none
- `pretrain-clspatch-both-l020-noaux-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T21:33:17.335564+00:00; finished=2026-07-30T22:34:10.787926+00:00; terminal_event=d229c61b-0bea-4da4-b873-4d8dbadec2e2; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-patch-screen/runs/pretrain-clspatch-both-l020-noaux-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/8d434d83); checkpoint=retained; metrics=peak_accuracy=0.286600, final_accuracy=0.286600, step_to_90=3045, step_to_95=censored, active_seconds_to_90=2547.168, active_seconds_to_95=censored, step_auc=0.263380, active_time_auc=0.262174, active_seconds_at_step_horizon=3634.086, cls_path_latency_median_ms=31.304704, cls_path_latency_p90_ms=32.068607; error=none

## 2026-07-30 interpretation and amendment: LeJEPA masked tuning

- Collapse result: Direct CLS-plus-patch-mean SigREG at lambda `0.20` prevented deterministic evaluation collapse. Both lambda-`0.20` runs passed every fixed representation threshold on all three final validations. The deterministic run ended with CLS and patch effective-rank fractions `0.248569` and `0.235366`; the stochastic run ended at `0.310163` and `0.223252`. Mean standard deviations remained above `0.84`, pairwise cosines remained below `0.008`, and top-eigenvalue fractions remained below `0.035`.
- Performance result: No run passed the complete gate because peak probe accuracy stayed below `0.50`. Deterministic lambda `0.20` was the screen leader at `0.3084`, followed by lambda `0.05` with auxiliary CLS prediction at `0.3056`, stochastic lambda `0.20` at `0.2866`, and lambda `0.05` without auxiliary prediction at `0.2590`.
- Fixed directional evidence: Removing stochastic regularizers improved lambda-`0.20` peak accuracy by `0.0218`. At lambda `0.05`, retaining auxiliary CLS prediction improved peak accuracy by `0.0466` and raised final patch effective rank from `0.079895` to `0.098291`, just below the `0.10` gate.
- Protocol amendment: Use the final two masked-only trials to combine auxiliary CLS prediction with deterministic training and compare lambda `0.10` against `0.20`. This holds masks, augmentations, direct feature sources, optimizer, shared-student target, and detached probe boundary fixed. If neither run passes the complete gate, masked-only optimization is exhausted and the reserved crop fallback becomes eligible.
- Allocation: Ten of 22 authorized scientific pretraining runs will be used after this two-run amendment: four feasibility, four patch screen, and two masked tuning. Six paired-confirmation and six crop-fallback runs remain reserved. Mechanical smokes remain excluded.
- Retention: Keep all four patch-screen checkpoints and backbones. No destructive retention was applied.
<!-- autoresearch-operation:{"content_sha256":"8da18d2aeffd5f203830af1fe332de6a774c210c29ab04438ce426347eda10d8","operation_id":"09a0be7d7444ce2d54bbe3af311d0867"} -->

<!-- study:lejepa-masked-collapse-v1-masked-tuning:phase:no-promotion -->
## lejepa-masked-collapse-v1-masked-tuning

- Question: Can auxiliary CLS prediction and deterministic training recover semantic probe quality from the noncollapsed patch-aware shared-student mechanism without introducing local or global crops?
- Hypothesis: Combining the two independently favorable screen choices will improve probe accuracy over 0.3084, while lambda 0.10 will preserve the collapse margin and allocate more weight to masked visual and auxiliary CLS prediction than lambda 0.20.
- Mechanisms and exact changes:
  - `clspatch-both-l010-aux-deterministic`: Mechanism: Apply direct CLS-plus-patch-mean SigREG to both masked and full views at lambda 0.10 with auxiliary CLS prediction and deterministic encoder/predictor paths. Changes: Restore four-partition auxiliary CLS prediction to the deterministic patch-aware mechanism.; Use the midpoint lambda 0.10 to balance masked prediction and SigREG.; Keep attention dropout, hidden dropout, and stochastic depth at zero.
  - `clspatch-both-l020-aux-deterministic`: Mechanism: Apply the same deterministic auxiliary design at lambda 0.20. Changes: Change only LeJEPA lambda from 0.10 to 0.20.
- Launch code provenance:
  - `pretrain-clspatch-both-l010-aux-deterministic-seed0`: parent=`07f1aff5ff963e6611d53ba91b8d0393ce1d15bf` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-clspatch-both-l020-aux-deterministic-seed0`: parent=`07f1aff5ff963e6611d53ba91b8d0393ce1d15bf` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`d024b0caced600d059f22eec146339aeec600ff5` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-masked-tuning/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-clspatch-both-l010-aux-deterministic-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-30T22:40:09.020100+00:00; finished=2026-07-30T23:45:03.862392+00:00; terminal_event=54892a51-7f2c-4d58-8784-5d144ea7fe6f; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-masked-tuning/runs/pretrain-clspatch-both-l010-aux-deterministic-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/15d51cb3); checkpoint=retained; metrics=peak_accuracy=0.320400, final_accuracy=0.320400, step_to_90=1740, step_to_95=3480, active_seconds_to_90=1562.162, active_seconds_to_95=3102.901, step_auc=0.279780, active_time_auc=0.278983, active_seconds_at_step_horizon=3874.822, cls_path_latency_median_ms=31.504384, cls_path_latency_p90_ms=32.321537; error=none
- `pretrain-clspatch-both-l020-aux-deterministic-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-30T22:40:09.094591+00:00; finished=2026-07-30T23:44:29.938169+00:00; terminal_event=a01e8a83-d0c8-4e3d-a1a1-68e93b890a9b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-masked-tuning/runs/pretrain-clspatch-both-l020-aux-deterministic-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/5e41be71); checkpoint=retained; metrics=peak_accuracy=0.297600, final_accuracy=0.297600, step_to_90=4350, step_to_95=censored, active_seconds_to_90=3840.627, active_seconds_to_95=censored, step_auc=0.257950, active_time_auc=0.257816, active_seconds_at_step_horizon=3840.627, cls_path_latency_median_ms=30.977535, cls_path_latency_p90_ms=31.719423; error=none

## 2026-07-30 interpretation and amendment: LeJEPA multiview fallback

- Masked-only outcome: Lambda `0.10` was the final masked-only winner at `0.3204` peak and final probe accuracy, versus `0.2976` for lambda `0.20`. The winner passed every fixed representation threshold on all three final validations and ended with CLS and patch effective-rank fractions `0.193975` and `0.187840`.
- Diagnosis: The winner's visual-target shuffled relative improvement reached `0.915939`, so masked prediction remained strongly input-dependent, but the detached CLS probe stayed `0.5854` below the accepted `0.9058` seed-0 baseline. SigREG prevented collapse but did not impose same-image invariance across independent augmentations.
- Protocol amendment: Preserve the masked visual and auxiliary CLS tasks and add the invariance term from the official LeJEPA minimal objective. Screen two global views, four global views, two global plus two moderate local views, an official-style projector, lambda `0.05`, and invariance weight `2.0`. The first global view remains the masked-task anchor; every full crop view retains student gradients; only the probe boundary detaches.
- Promotion discipline: A run must remain noncollapsed and reach at least `0.85` peak, `0.84` final, and `0.75` step AUC before using the six reserved paired-confirmation trials. If none qualifies, stop the pretraining program without downstream evaluation.
- Allocation: Sixteen of 22 authorized scientific pretraining runs are committed after this screen: four feasibility, four patch screen, two masked tuning, and six multiview screen. Six fresh paired-confirmation runs remain reserved. Mechanical smokes remain excluded.
- Retention: Keep both masked-tuning checkpoints and backbones. No destructive retention was applied.
<!-- autoresearch-operation:{"content_sha256":"20ca71feec0e3bc04850331bb3e28ac0d669a08fc49fc4cf25b4a87c8519e04c","operation_id":"1e046e8a8b809167b93bd40774cb982a"} -->

<!-- study:lejepa-masked-collapse-v1-multiview-smoke:phase:no-promotion -->
## lejepa-masked-collapse-v1-multiview-smoke

- Question: Does masked JEPA plus same-image LeJEPA invariance complete one global/local train-validation-checkpoint cycle?
- Hypothesis: The multiview branch will preserve gradients through every student view, add finite invariance and SigREG losses, retain the detached probe boundary, and leave masked prediction operational.
- Mechanisms and exact changes:
  - `multiview-g2l2-direct-smoke`: Mechanism: Combine the existing masked visual and auxiliary CLS objectives with direct CLS-plus-patch invariance across two global and two moderate local views. Changes: Keep one global view as the masked image-modeling anchor.; Add one independently augmented global view and two moderate local views.; Apply same-image invariance and SigREG without teacher, stop-gradient, or projector.
- Launch code provenance:
  - `pretrain-multiview-g2l2-direct-smoke-seed0`: parent=`085ea49458be9a37230a1af7a3ecb67d1c0069e9` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-smoke/summary.json`; external_detail=True
- Conclusion: The baseline smoke run completed; no candidates were configured for promotion.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-multiview-g2l2-direct-smoke-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-31T00:04:49.215023+00:00; finished=2026-07-31T00:08:38.671089+00:00; terminal_event=f3a8ecef-5639-41f3-a764-45b82aab5a6e; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-smoke/runs/pretrain-multiview-g2l2-direct-smoke-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/10e9bc2d); checkpoint=retained; metrics=peak_accuracy=0.218000, final_accuracy=0.218000, step_to_90=44, step_to_95=44, active_seconds_to_90=209.861, active_seconds_to_95=209.861, step_auc=0.218000, active_time_auc=0.218000, active_seconds_at_step_horizon=209.861, cls_path_latency_median_ms=31.455744, cls_path_latency_p90_ms=32.092159; error=none

## 2026-07-31 interpretation: LeJEPA multiview smoke

- Mechanical result: The four-view global/local path completed its train, validation, checkpoint, W&B, and lifecycle cycle with exit code `0`. Losses remained finite, gradient assertions passed, peak allocated GPU memory stayed near `5.6 GiB`, and the run did not exhaust memory.
- Representation evidence: At step `44`, CLS and patch mean standard deviations were `0.759590` and `0.721350`, mean pairwise cosines were `0.013630` and `0.006020`, and top-eigenvalue fractions were `0.327880` and `0.370350`. Visual-target shuffled relative improvement was `0.783316`.
- Limitation and decision: Effective-rank fractions of `0.015080` for CLS and `0.011160` for patches after one epoch are not promotion evidence. The smoke validates mechanics only, so proceed with the committed six-run screen unchanged and evaluate collapse and semantic quality over the full 100 epochs.
- Retention: Keep the smoke checkpoint and backbone. No destructive retention was applied.
<!-- autoresearch-operation:{"content_sha256":"a390aa94cc6a1f17ccd143a71901ed92f299090d1ff438ac4c58cc6cfc46fd1f","operation_id":"2d5e491f2d8a3a385f1886d073e7520b"} -->

<!-- study:lejepa-masked-collapse-v1-multiview-screen:phase:no-promotion -->
## lejepa-masked-collapse-v1-multiview-screen

- Question: Which LeJEPA same-image view design restores semantic probe quality while retaining masked prediction and noncollapsed shared-student features?
- Hypothesis: Same-image invariance across independent augmented views will supply the semantic signal missing from masked-only SigREG; global-view count, moderate local crops, projection, lambda, and invariance weight determine whether that signal approaches the accepted 0.9058 probe baseline.
- Mechanisms and exact changes:
  - `multiview-g2-direct-l010-w1`: Mechanism: Add LeJEPA invariance between two independently augmented full student views while retaining masked visual prediction, auxiliary CLS prediction, and direct CLS-plus-patch SigREG. Changes: Keep one global view as the masked prediction anchor.; Add one independent global view with the existing crop scale.; Apply invariance to direct CLS and patch-mean features with weight 1.0.
  - `multiview-g4-direct-l010-w1`: Mechanism: Increase the direct invariance branch from two to four global views. Changes: Add two more independent global views without changing crop scale, loss weights, or feature source.
  - `multiview-g2l2-direct-l010-w1`: Mechanism: Use two global views at scale 0.75-1.00 and two local views at scale 0.30-0.75. Changes: Replace two additional global views with local crop views while holding total view count and losses fixed.
  - `multiview-g2l2-proj64-l010-w1`: Mechanism: Apply the four-view invariance and SigREG objectives after a 384-2048-2048-64 projector. Changes: Add the pretraining-only SigREG/invariance projector; preserve all crop and loss settings.
  - `multiview-g2l2-direct-l005-w1`: Mechanism: Lower the direct four-view convex SigREG weight from 0.10 to 0.05. Changes: Change only LeJEPA lambda.
  - `multiview-g2l2-direct-l010-w2`: Mechanism: Double direct four-view invariance weight from 1.0 to 2.0. Changes: Change only invariance loss weight.
- Launch code provenance:
  - `pretrain-multiview-g2-direct-l010-w1-seed0`: parent=`dcab90bfd14ee9df0d8a5cc113d50e7747ba257c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-multiview-g2l2-direct-l005-w1-seed0`: parent=`dcab90bfd14ee9df0d8a5cc113d50e7747ba257c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-multiview-g2l2-direct-l010-w1-seed0`: parent=`dcab90bfd14ee9df0d8a5cc113d50e7747ba257c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-multiview-g2l2-direct-l010-w2-seed0`: parent=`dcab90bfd14ee9df0d8a5cc113d50e7747ba257c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-multiview-g2l2-proj64-l010-w1-seed0`: parent=`dcab90bfd14ee9df0d8a5cc113d50e7747ba257c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
  - `pretrain-multiview-g4-direct-l010-w1-seed0`: parent=`dcab90bfd14ee9df0d8a5cc113d50e7747ba257c` (`codex/research/lejepa-masked-collapse-v1`), mjepa=`8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6` (`codex/research/lejepa-masked-collapse-v1`), vit=`bf15705454975f04912538cdc790d399eea69e67` (`codex/research/cls-context-routing-v1`)
- Phase: no-promotion
- Winner: none
- External tracker: provider=W&B; account=tidalpaladin; project=mjepa-cifar10; authorized=True; approved_data_classes=metrics, configs, provenance
- Detail location: local summary and raw metrics under `/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/summary.json`; external_detail=True
- Conclusion: No seed-0 candidate met a promotion threshold.
- Follow-up: record interpretation and the next falsifiable hypothesis.
- Checkpoint disposition: see each run below; deleted weights are not recoverable.

- `pretrain-multiview-g2-direct-l010-w1-seed0`: attempt=1; status=completed; decision=baseline; started=2026-07-31T00:16:42.995612+00:00; finished=2026-07-31T02:12:37.241708+00:00; terminal_event=f40cf4fd-8883-4e21-8bff-42b58534cf2b; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/runs/pretrain-multiview-g2-direct-l010-w1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/a634b715); checkpoint=retained; metrics=peak_accuracy=0.356000, final_accuracy=0.356000, step_to_90=3071, step_to_95=3948, active_seconds_to_90=4863.325, active_seconds_to_95=6243.900, step_auc=0.298324, active_time_auc=0.298084, active_seconds_at_step_horizon=6934.046, cls_path_latency_median_ms=31.417344, cls_path_latency_p90_ms=32.365566; error=none
- `pretrain-multiview-g2l2-direct-l005-w1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-31T17:46:41.479009+00:00; finished=2026-07-31T20:48:45.105641+00:00; terminal_event=5e39d216-7a38-4fd2-bff3-2ba5fd139852; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/runs/pretrain-multiview-g2l2-direct-l005-w1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/e1a0d066); checkpoint=retained; metrics=peak_accuracy=0.404200, final_accuracy=0.404200, step_to_90=2632, step_to_95=3071, active_seconds_to_90=6547.451, active_seconds_to_95=7636.186, step_auc=0.314644, active_time_auc=0.287180, active_seconds_at_step_horizon=10903.994, cls_path_latency_median_ms=31.426032, cls_path_latency_p90_ms=32.340992; error=none
- `pretrain-multiview-g2l2-direct-l010-w1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-31T14:03:33.215614+00:00; finished=2026-07-31T17:05:25.967713+00:00; terminal_event=f95cd2ad-c5a4-44b7-9a16-e0c403c4062a; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/runs/pretrain-multiview-g2l2-direct-l010-w1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c58c890d); checkpoint=retained; metrics=peak_accuracy=0.370800, final_accuracy=0.370800, step_to_90=2632, step_to_95=3071, active_seconds_to_90=6538.740, active_seconds_to_95=7627.448, step_auc=0.315795, active_time_auc=0.295940, active_seconds_at_step_horizon=10893.038, cls_path_latency_median_ms=31.364096, cls_path_latency_p90_ms=32.066559; error=none
- `pretrain-multiview-g2l2-direct-l010-w2-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-31T17:46:41.557399+00:00; finished=2026-07-31T20:47:15.121025+00:00; terminal_event=28c0be8b-f786-4900-9030-b4274078c59e; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/runs/pretrain-multiview-g2l2-direct-l010-w2-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/634e04b8); checkpoint=retained; metrics=peak_accuracy=0.406800, final_accuracy=0.406800, step_to_90=2193, step_to_95=3071, active_seconds_to_90=5413.042, active_seconds_to_95=7573.192, step_auc=0.328744, active_time_auc=0.301509, active_seconds_at_step_horizon=10814.686, cls_path_latency_median_ms=30.849536, cls_path_latency_p90_ms=31.689728; error=none
- `pretrain-multiview-g2l2-proj64-l010-w1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-31T14:03:33.289121+00:00; finished=2026-07-31T17:07:25.086618+00:00; terminal_event=7b1ae54e-95e4-484c-a19e-2b25d03a6da6; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/runs/pretrain-multiview-g2l2-proj64-l010-w1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/f172136e); checkpoint=retained; metrics=peak_accuracy=0.305400, final_accuracy=0.305400, step_to_90=censored, step_to_95=censored, active_seconds_to_90=censored, active_seconds_to_95=censored, step_auc=0.274323, active_time_auc=0.262075, active_seconds_at_step_horizon=11012.247, cls_path_latency_median_ms=30.846463, cls_path_latency_p90_ms=31.504383; error=none
- `pretrain-multiview-g4-direct-l010-w1-seed0`: attempt=1; status=completed; decision=rejected; started=2026-07-31T00:16:43.070985+00:00; finished=2026-07-31T03:17:28.536042+00:00; terminal_event=576607d4-847e-40f5-baa2-e9b35828cf00; artifacts=`/home/tidal/Documents/mjepa-cifar10/logs/research/lejepa-masked-collapse-v1-multiview-screen/runs/pretrain-multiview-g4-direct-l010-w1-seed0`; W&B=[run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/a016731b); checkpoint=retained; metrics=peak_accuracy=0.353400, final_accuracy=0.353400, step_to_90=2632, step_to_95=3948, active_seconds_to_90=6518.724, active_seconds_to_95=9750.503, step_auc=0.306195, active_time_auc=0.289026, active_seconds_at_step_horizon=10825.422, cls_path_latency_median_ms=30.869504, cls_path_latency_p90_ms=31.717377; error=none

## 2026-07-31 interpretation and program decision: LeJEPA multiview screen

- Semantic result: Doubling invariance weight was the screen leader at `0.4068` peak/final probe accuracy and `0.328744` step AUC. Lambda `0.05` followed at `0.4042`; the direct local-crop default reached `0.3708`; two and four global views reached `0.3560` and `0.3534`; and the projected design reached `0.3054`. The leader improved peak accuracy by `0.0864` and step AUC by `0.048964` over the `0.3204` masked-only winner, but remained `0.4990` below the accepted `0.9058` seed-0 baseline.
- View and loss effects: Four global views did not improve on two and used `3891.376` more active seconds. Replacing two global views with moderate local crops improved peak accuracy by `0.0174`. Within the local-crop design, lowering lambda from `0.10` to `0.05` improved peak accuracy by `0.0334`, while doubling invariance weight improved it by `0.0360` and produced the highest step AUC.
- Collapse result: No run passed every preregistered eligibility threshold on all three final validations. The direct variants retained substantial variance, low pairwise cosine, finite outputs, and strong shuffled-target improvement, but at least one CLS or patch effective-rank fraction remained below `0.10`. The projector variant collapsed in raw representation space: final CLS standard deviation was `0.007828`, mean pairwise cosine was `0.999264`, and effective-rank fraction was `0.033420`, even though projected CLS effective rank reached `0.346508`.
- Promotion decision: No candidate reached the `0.50` collapse-eligibility probe floor or the plausible-equivalence floors of `0.85` peak, `0.84` final, and `0.75` step AUC. Per the fixed stopping rule, do not run paired confirmation, do not use the official test set, and do not launch full, 10-shot, or 100-shot downstream evaluation.
- Program conclusion: Shared-student SigREG can prevent obvious variance/cosine collapse without an EMA teacher, and same-image multiview invariance improves semantic probing over the masked-only design. In this implementation, however, neither masked prediction plus invariance nor the tested LeJEPA-style projector approaches the accepted teacher baseline. The relaxed equivalence condition is not met.
- Future hypothesis: If a separate program is authorized, test whether masked prediction and augmentation invariance need separate representation subspaces or a scheduled loss balance. This hypothesis is outside the completed program and its 22-run authorization; it is not authorized by this result.
- Allocation and retention: Sixteen of 22 authorized scientific pretraining runs were used; the six confirmation runs remain unused because the promotion gate failed. Mechanical smokes were excluded. Keep every checkpoint and backbone; no destructive retention was applied.

## 2026-08-01 protocol: LeJEPA convergence probe calibration

- Study: `lejepa-convergence-v1-probe`.
- Question: Does the moving online probe materially understate the final linear separability of the retained teacher-free SigREG representations?
- Hypothesis: A fresh frozen-backbone probe will improve the best shared-student validation accuracy by at least `0.10`, but accuracy below `0.60` or a remaining teacher gap above `0.20` will identify representation convergence as the primary limitation.
- Evaluation: Reuse all six retained multiview checkpoints and the accepted teacher checkpoint. Train only fresh probe heads on the fixed 45,000-example training split and evaluate on the fixed 5,000-example validation split. The official test set remains prohibited.
- Recipes: Compare final-layer CLS without added normalization against concatenated final-two-layer CLS features with fixed LayerNorm. Sweep six preregistered AdamW learning rates with weight decay `1e-6` and linear-warmup cosine decay. The encoder remains in evaluation mode without gradients.
- External tracking: W&B destination `tidalpaladin/mjepa-cifar10`; launch emits the authorized non-sensitive `metrics`, `configs`, and `provenance` classes.
- Decision: If probe lag does not explain the deficit, preregister a fresh pretraining screen that isolates learning rate, weight decay, and schedule before changing loss composition or views.
- Retention: Keep feature caches, full probe curves, calibration results, and every source checkpoint. No destructive retention is authorized.
<!-- autoresearch-operation:{"operation_id":"lejepa-convergence-v1-probe-protocol-v1"} -->

## 2026-08-01 result: LeJEPA convergence probe smoke

- Study: `lejepa-convergence-v1-probe-smoke`; run: `multiview-g2l2-direct-l010-w2-smoke-seed0`; status: completed.
- Mechanical result: Both final-layer CLS and normalized final-two-layer CLS paths extracted the fixed 45,000/5,000 split, trained fresh frozen-backbone probe heads, published online metrics, and wrote atomic feature-cache, result, metadata, and terminal artifacts in `20.89` active seconds.
- Provenance: parent `897b2ccffb3cb33985cbb902c64fe1fd4de0082c`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`; source checkpoint SHA-256 `c007638f3a5a0c6e7c68db6ccbcd786dda9fa9e609b396dc0c51ba13baf820e5`.
- Tracker and recovery: W&B [run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/9iahsx2t); notify-wake watch `c6f0de42-b95f-48df-a620-d28b7f19605e` closed with `succeeded` and accepted delivery.
- Interpretation: The one-epoch, one-learning-rate accuracies are mechanical evidence only. They do not alter the scientific calibration hypothesis, fixed learning-rate sweep, or decision thresholds.
- Decision and retention: Approve the seven-checkpoint calibration unchanged. Retain the feature cache and all source weights; no destructive retention was applied.
<!-- autoresearch-operation:{"operation_id":"lejepa-convergence-v1-probe-smoke-result-v1"} -->

## 2026-08-01 result: LeJEPA convergence frozen-probe calibration

- Study: `lejepa-convergence-v1-probe`; status: completed. Two GPU shards processed all seven preregistered checkpoints in `477.94` wall seconds and `787.56` summed active seconds; every run, aggregate worker terminal, and notify-wake watch `214b8e49-4b01-4ed3-af45-fa616d40722e` closed successfully.
- Provenance: parent `adaf8e85f8b5371a2baf4a6894d3cd8ebadfc05c`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`; manifest SHA-256 `fbf4cfe120652184e5cb123c94dd5f5540be54056fa656fe9837ce92c405fff5`. Structured summary: `logs/research/lejepa-convergence-v1-probe/summary.json` with SHA-256 `b3c931b007ba7139da1c7be3ef3cab33b83413db9eaea874b2862fb8329e9d8a`.

| Source | Online probe | Frozen final CLS | Frozen normalized final-two CLS | Calibration gain |
| --- | ---: | ---: | ---: | ---: |
| teacher baseline | 0.9058 | 0.8952 | 0.9004 | -0.0054 |
| G2 direct, lambda 0.10, invariance 1 | 0.3560 | 0.4268 | 0.4604 | +0.1044 |
| G4 direct, lambda 0.10, invariance 1 | 0.3534 | 0.4358 | 0.4780 | +0.1246 |
| G2L2 direct, lambda 0.10, invariance 1 | 0.3708 | 0.4596 | 0.4970 | +0.1262 |
| G2L2 projected, lambda 0.10, invariance 1 | 0.3054 | 0.3856 | 0.3924 | +0.0870 |
| G2L2 direct, lambda 0.05, invariance 1 | 0.4042 | 0.5014 | 0.5260 | +0.1218 |
| G2L2 direct, lambda 0.10, invariance 2 | 0.4068 | 0.5202 | 0.5768 | +0.1700 |

- Decision: Probe lag is material because the best shared checkpoint gained `0.1700`, above the preregistered `0.10` threshold. Representation convergence remains primary because `0.5768` is below the `0.60` shared floor and `0.3236` behind the calibrated teacher, above the `0.20` maximum gap. The normalized final-two-layer recipe won every checkpoint; all shared raw and normalized recipes selected the sweep boundary learning rate `0.03`, while the teacher's normalized recipe selected `0.003`.
- Tracker: All seven W&B runs completed online and contain exact summary metrics and provenance. The per-epoch history reset from step 99 to 0 for the second recipe, so W&B rejected that recipe's curve history as nonmonotonic; the complete curves remain in each local `result.json`. The logger now offsets recipe steps monotonically before any subsequent calibration.
- Next action: Use the fixed frozen calibration as the terminal evaluation and preregister a fresh four-run optimizer/schedule screen. Give the detached online classifier a low-decay, higher-learning-rate parameter group to reduce measurement lag without changing encoder gradients; isolate lower encoder learning rate and lower weight decay, then test their official-recipe combination with cosine-shaped OneCycle decay.
- Retention: Retain all feature caches, complete curves, W&B runs, and source checkpoints. No destructive retention was applied.
<!-- autoresearch-operation:{"operation_id":"lejepa-convergence-v1-probe-result-v1"} -->

## 2026-08-01 protocol: LeJEPA convergence optimizer/schedule screen

- Study: `lejepa-convergence-v1-optimizer-screen`; linked smoke: `lejepa-convergence-v1-optimizer-smoke`.
- Trigger: Frozen calibration found a material `+0.1700` probe gain but left the best shared representation at `0.5768`, below the `0.60` floor and `0.3236` behind the calibrated teacher. Optimizer and schedule changes are therefore justified before altering the objective or view topology.
- Baseline: Fresh 100-epoch G2L2 direct run with lambda `0.10`, invariance weight `2.0`, AdamW learning rate `0.002`, weight decay `0.2`, and 5% linear warmup followed by a constant rate.
- Initial variants: Isolate encoder learning rate `0.0005`; isolate encoder weight decay `0.05`; combine both with 10% warmup and cosine-shaped OneCycle decay to maximum learning rate divided by 1000. The OneCycle candidate also cycles AdamW beta1 and is recorded as an approximation rather than an exact reproduction of the official LeJEPA scheduler.
- Measurement amendment: All four runs remove classifier dropout and place only the detached classifier head in an AdamW group with learning rate `0.01` and weight decay `1e-6`. A regression test proves the shared-student model's `heads` selector contains only classifier weight and bias; encoder gradients remain blocked at the classifier boundary. Terminal selection uses the same fixed frozen final-two-layer normalized probe as the completed calibration.
- Controls: Fixed 45,000/5,000 split, no official test set, seed 0 screen, 100 epochs, validation every 10 epochs, effective batch size 1024, identical masking, two global plus two local views, shared-student gradients, and full online W&B provenance.
- Allocation: Four seed-0 runs initially. A managed promotion may add only fresh baseline/winner seeds 1 and 2, for at most eight scientific pretraining runs. The one-epoch mechanical smoke is excluded.
- Promotion: Require `+0.02` peak online accuracy, 10% lower active time to the fixed 95% target, or 5% higher active-time AUC with no more than `0.005` peak loss for speed/AUC routes. Calibrate every seed-0 checkpoint before choosing the optimizer control for a separate preregistered loss-interaction and view-cost study.
- Retention: Keep every checkpoint, curve, cache, and tracker run. No destructive retention is authorized.
<!-- autoresearch-operation:{"operation_id":"lejepa-convergence-v1-optimizer-protocol-v1"} -->

## 2026-08-01 result: LeJEPA optimizer-screen mechanical smoke

- Study: `lejepa-convergence-v1-optimizer-smoke`; run: `pretrain-g2l2-probe-group-smoke-seed0`; status: completed with exit code `0`.
- Mechanical result: The shared-student G2L2 baseline completed one train, validation, checkpoint, and W&B cycle at optimizer step `44` in `121.837` active seconds. The detached classifier-only parameter group ran without changing the pretraining boundaries. Peak and final online probe accuracy were `0.2024`; these one-epoch values are not scientific promotion evidence.
- Representation checks: Final CLS and patch-mean standard deviations were `0.760708` and `0.694544`, mean pairwise cosines were `0.023537` and `0.009581`, and visual-target shuffled relative improvement was `0.722906`. All recorded fractions were finite. The smoke did not show variance or cosine collapse.
- Provenance and artifacts: Parent `7356cc76a3efc812343a1b29839ce30e392b9217`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`. W&B [run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/b9589e96); local summary `logs/research/lejepa-convergence-v1-optimizer-smoke/summary.json`. The checkpoint and backbone remain retained.
- Notification finding: Training was unaffected, but the first-cycle and terminal deliveries were initially blocked because the persistent research controller had been wrapped in a generic process watch. That watch owned a different goal-wait lease. The exact controller was stopped after terminal reconciliation, and the terminal research event was explicitly requeued and accepted.
- Workflow repair: Add `research.py notify-wait`, which binds an owned goal wait to the direct controller's PID, Linux start ticks, exact registered root, study set, command line, and durable startup record. Regression tests cover stat parsing, identity rejection, lease inputs, and the controller record. Use this path before the full optimizer screen.
- Decision: Approve the four-run optimizer and schedule screen unchanged after the workflow repair passes the repository gate and the launch commit is pushed. No destructive retention was applied.
<!-- autoresearch-operation:{"operation_id":"lejepa-convergence-v1-optimizer-smoke-result-v1"} -->
<!-- autoresearch-operation:{"content_sha256":"a304350105487201139f18281ef1cafd686e18556b28fe0c2b0a9db099872156","operation_id":"lejepa-convergence-v1-notification-recovery-v1"} -->


## 2026-08-01 amendment: optimizer-screen notification recovery

- Observation: The lower-learning-rate first-cycle source was written at `01:51:49.173671` UTC and queued `0.091` seconds later, but its delivery attempt remained at zero. The baseline source was written at `01:51:57.379` UTC and was not queued until manual reconciliation. The controller recorded neither a completed nor failed post-event notification sweep.
- Boundary: The exact owned goal lease remained healthy and the shared notify-wake preflight reported automatic delivery available. The adapter controller had entered the post-event path far enough to queue one event, but its exact blocked stack was unavailable after the process exited. It had also been launched inside an interactive command session instead of as a detached process.
- Recovery: A scoped reconciliation and one-shot worker accepted both pending events through the current turn, released the exact lease, and reactivated the goal. Both trainers continued normally and retained their checkpoints throughout the notification outage.
- Repair: Controller reconciliation and inotify now select only the requested study. Notification sweeps have a `60`-second bound and controller-local `5`-second retry. The new `start-controller` command launches a detached exact process, verifies its durable startup identity, and reuses a matching live controller.
- Scientific impact: None. No run, hyperparameter, comparison, promotion threshold, checkpoint, or tracker record changed. The two optimizer-screen runs remain in progress under their preregistered protocol.
<!-- autoresearch-operation:{"content_sha256":"d3f689dca6078ee7a1daeb7dbe7de40546bce89856019ea01df3e307eb35556a","operation_id":"lejepa-convergence-v1-optimizer-screen-baseline-seed0-attempt1-terminal"} -->


## 2026-08-01 terminal: optimizer-screen baseline seed 0

- Identity: `pretrain-g2l2-w2-lr2e3-wd2e1-constant-seed0`, attempt `1`, terminal event `7c9aa799-20d6-4378-b77c-7df4f98b0da2`. The run started at `01:33:23.874492` UTC and completed with exit code `0` at `04:35:18.279645` UTC on physical GPU 1.
- Online result: Peak and final validation accuracy were `0.4190` at `10,895.01` active seconds. Every variance, effective-rank, eigenvalue, cosine, shuffled-target, and finite-value gate passed on epochs 79, 89, and 99. The online accuracy remained below the preregistered `0.50` eligibility floor.
- Provenance: Parent `2e44e71f36aecf52752ac227b86df0af26fa153e`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`. W&B [run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/3f51007e).
- Recovery and decision: The v2 terminal notification was accepted on its first `turn/steer` attempt. Both `checkpoint.pt` and `backbone.safetensors` remain retained. The fixed frozen probe is pending, so no promotion decision is recorded.
<!-- autoresearch-operation:{"content_sha256":"0d1c23085c14f899a25302812da1a0e59290dd187de8a92b49a2e923412fbb83","operation_id":"lejepa-convergence-v1-optimizer-screen-lr5e4-seed0-attempt1-terminal"} -->


## 2026-08-01 terminal: optimizer-screen lower-learning-rate seed 0

- Identity: `pretrain-g2l2-w2-lr5e4-wd2e1-constant-seed0`, attempt `1`, terminal event `9c2dfc3e-bf5e-4f89-b2aa-91f1f11c7b06`. The run started at `01:33:23.953075` UTC and completed with exit code `0` at `04:33:54.076830` UTC on physical GPU 2.
- Online result: Peak and final validation accuracy were `0.4286` at `10,810.84` active seconds, a `0.0096` accuracy increase and `84.17` active-second reduction relative to the seed-0 baseline. The last-three patch effective-rank fractions were `0.06484`, `0.06960`, and `0.07303`, so epochs 79 and 89 missed the preregistered `0.07` floor. Peak online accuracy also remained below the `0.50` eligibility floor.
- Provenance: Parent `2e44e71f36aecf52752ac227b86df0af26fa153e`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`. W&B [run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/6c589f64).
- Recovery and decision: The v2 terminal notification was accepted on its first `turn/steer` attempt. Both `checkpoint.pt` and `backbone.safetensors` remain retained. The fixed frozen probe is pending, so no promotion decision is recorded.
<!-- autoresearch-operation:{"content_sha256":"0388b2be429d06100eb39be84e8240016ac46882e79a7d952016854facc7e276","operation_id":"lejepa-convergence-v1-optimizer-screen-onecycle-seed0-attempt1-terminal"} -->


## 2026-08-01 terminal: optimizer-screen OneCycle seed 0

- Identity: `pretrain-g2l2-w2-lr5e4-wd5e2-onecycle-seed0`, attempt `1`, terminal event `d27c8fd5-a0fb-4e4a-a205-d6570d32c282`. The run started at `04:36:19.462162` UTC and completed with exit code `0` at `07:36:47.382483` UTC on physical GPU 2.
- Online result: Peak accuracy was `0.3788`, final accuracy was `0.3770`, and final active time was `10,809.01` seconds. Peak accuracy was `0.0402` below the seed-0 baseline, while active time was `86.00` seconds lower. The last-three CLS effective-rank fractions were `0.06877`, `0.06941`, and `0.06969`; the patch fractions were `0.05675`, `0.05740`, and `0.05767`. All six values missed the preregistered `0.07` floor. Variance, eigenvalue, cosine, shuffled-target, and finite-value checks passed.
- Provenance: Parent `c72d47d865375d7c5d90f63f1063ab85d03de2da`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`. W&B [run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/45a29970).
- Recovery and decision: The v2 terminal notification was accepted on its first `turn/steer` attempt. Both `checkpoint.pt` and `backbone.safetensors` remain retained for the fixed frozen probe. The run is ineligible under the online and late-rank gates, but final optimizer selection remains pending until every seed-0 checkpoint is calibrated.
<!-- autoresearch-operation:{"content_sha256":"0a69e17ad4115a5952241e4120e5a8aa42d54ced69cda640931e403a4e07b01f","operation_id":"lejepa-convergence-v1-optimizer-screen-wd5e2-seed0-attempt1-terminal"} -->


## 2026-08-01 terminal: optimizer-screen lower-weight-decay seed 0

- Identity: `pretrain-g2l2-w2-lr2e3-wd5e2-constant-seed0`, attempt `1`, terminal event `f2b61acb-8a7c-4245-868f-a370421f72b2`. The run started at `04:36:19.383323` UTC and completed with exit code `0` at `07:38:08.836702` UTC on physical GPU 1.
- Online result: Peak and final accuracy were `0.4480` at `10,890.72` active seconds. Peak accuracy was `0.0290` above the seed-0 baseline, while active time was `4.29` seconds lower. Every last-three variance, effective-rank, eigenvalue, cosine, shuffled-target, and finite-value gate passed. Peak online accuracy remained below the preregistered absolute `0.50` eligibility floor.
- Provenance: Parent `c72d47d865375d7c5d90f63f1063ab85d03de2da`; `mjepa=8f9eab6beb6a0e1f9547e90ed8ce0d5e7bde42c6`; `vit=bf15705454975f04912538cdc790d399eea69e67`. W&B [run](https://wandb.ai/tidalpaladin/mjepa-cifar10/runs/c54a8bc8).
- Recovery and decision: The v2 terminal notification was accepted on its first `turn/steer` attempt. Both `checkpoint.pt` and `backbone.safetensors` remain retained for the fixed frozen probe. This run leads the online comparison, but optimizer selection remains pending until all four retained checkpoints use the preregistered frozen-probe recipe.
<!-- autoresearch-operation:{"content_sha256":"fd042e435cff2c2e33593de29966e0574aadee5235cbb1ccde46a6956980f855","operation_id":"lejepa-convergence-v1-optimizer-probe-protocol-v1"} -->


## 2026-08-01 protocol: LeJEPA optimizer frozen-probe calibration

- Study: `lejepa-convergence-v1-optimizer-probe`; source study: `lejepa-convergence-v1-optimizer-screen`.
- Trigger: All four seed-0 optimizer runs completed. Lower weight decay led online at `0.4480`, compared with `0.4190` for the original optimizer, `0.4286` for lower learning rate, and `0.3788` for OneCycle. Every source remained below the preregistered `0.50` online eligibility floor.
- Hypothesis: Lowering AdamW weight decay from `0.20` to `0.05` will preserve at least a `0.02` accuracy gain under the fixed normalized final-two-layer frozen probe. Lower learning rate and OneCycle are expected not to close the calibrated teacher gap.
- Evaluation: Freeze the accepted teacher checkpoint and all four retained optimizer checkpoints. Reuse only the fixed `last-two-cls-layernorm` recipe, six-head learning-rate bank, 100-epoch probe schedule, 45,000-example training split, and 5,000-example validation split from the completed calibration. The official test set remains prohibited.
- Selection boundary: Apply the optimizer-screen pretraining eligibility gates before comparing frozen accuracy. Frozen results diagnose convergence and select a control only among eligible sources; they cannot override the `0.50` online floor. If no optimizer source is eligible, retain the original optimizer as the control for a separately preregistered loss-interaction and view-cost study.
- External tracking: W&B destination `tidalpaladin/mjepa-cifar10`, group `lejepa-convergence-v1-optimizer-probe`; launch emits authorized `metrics`, `configs`, and `provenance`.
- Retention: Keep feature caches, complete curves, results, and every source checkpoint. No destructive retention is authorized.
