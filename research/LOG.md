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
