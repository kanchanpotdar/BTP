# Saved run artifacts

The project owner requested that the available downloaded Kaggle results and selected checkpoint archives be kept with the public repository. ZIP names retain their run meaning even where the source export used the word `private`. Raw dataset audio, frozen manifests, and the full per-epoch learning-rate archive are not included.

## Result exports

| Phase | Saved files | Scope |
|---|---|---|
| 4 | [Smoke results](exports/phase4/processor_model_smoke_results.zip) | Processor/model engineering outputs |
| 5B | [Corrected B0 results](exports/phase5b/zero_shot_validation_results.zip) | Full 732-clip validation aggregates and saved predictions |
| 6B | [Pilot results](exports/phase6b/pilot_results.zip) | P1 pilot outputs |
| 6C | [Equivalence](exports/phase6c/equivalence.json), [diagnostics](exports/phase6c/diagnostics.jsonl) | Validation-generation audit; diagnostic rows may contain identifiers and text |
| 7 F1 | [Results](exports/phase7/f1_results.zip), [plots](exports/phase7/f1_plots.zip), [validation audit](exports/phase7/f1_validation_audit.zip) | Full fine-tuning and follow-up audit |
| 7 F1 | [Penalty-sweep results](exports/phase7/repetition_penalty_results.zip), [plots](exports/phase7/repetition_penalty_plots.zip) | Validation-only decoding sweep |
| 7 F1 | [Learning-rate results](exports/phase7/f1_lr_sweep_results.zip) | Compact `5e-6` and `2e-5` run export; excludes redundant epoch states |
| 8 | [Final-test aggregates and plot](exports/phase8/test_aggregate_plots.zip), [prediction rows](exports/phase8/test_predictions.zip) | Predeclared B0 and selected F1 on 683 frozen test clips; prediction ZIP contains IDs and transcripts |

## Available model and training-state archives

These ZIPs use **Git LFS**. Clone with Git LFS installed, or download them individually from GitHub. Each archive was checked as a valid ZIP before staging.

| Run | Selected archive | Selection basis |
|---|---|---|
| 6A | [Training smoke state](models/phase6a_training_smoke_state.zip) | Engineering resume artifact; no CER selection |
| 6B | [P1 best checkpoint](models/phase6b_pilot_best_checkpoint.zip) | Best pilot validation epoch |
| 7 F1 | [Original F1 best checkpoint](models/phase7_f1_best_epoch3_checkpoint.zip) | Epoch 3 validation CER |
| 7 F1 LR | [`5e-6` best checkpoint](models/phase7_f1_lr5e6_best_epoch3_checkpoint.zip) | Epoch 3 validation CER |
| 7 F1 LR | [`2e-5` selected checkpoint](models/phase7_f1_lr2e5_best_epoch5_checkpoint.zip) | Epoch 5 validation CER; preselected for final test |

No LoRA checkpoint ZIP was present in the supplied local downloads. Phase 8's saved ZIPs were checked against the frozen test manifest and independently rescored; see the [test report](../docs/PHASE8_RESULTS.md). The original executed Phase 8 Kaggle notebook has not yet been supplied as a downloaded file.
