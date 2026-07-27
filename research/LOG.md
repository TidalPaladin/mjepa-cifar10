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
