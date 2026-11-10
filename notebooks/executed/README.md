# Executed Kaggle notebooks

These are copies of notebooks downloaded from Kaggle with their saved output cells. They preserve the supplied run record; they were not re-executed for this GitHub update. Source and public-results notebooks remain in the parent `notebooks/` folder.

| Phase | Executed notebook | Saved output cells | What it records |
|---|---|---:|---|
| 4 | [Processor/model smoke](phase4/processor_model_smoke.ipynb) | 5 | Pinned-model and processor engineering checks |
| 5B | [Zero-shot validation](phase5b/zero_shot_validation.ipynb) | 34 | B0 inference and saved plots; see the corrected aggregate report for all 732 clips |
| 6A | [Training smoke](phase6a/training_smoke.ipynb) | 12 | Two-update AMP and checkpoint/resume gate |
| 6B | [One-hour pilot](phase6b/pilot.ipynb) | 20 | Three-epoch pilot and full-validation measurements |
| 7 F1 | [Full fine-tuning](phase7/f1_full_finetuning.ipynb) | 79 | Full-data F1 run, diagnostics, and saved figures |
| 7 F1 | [Repetition-penalty sweep](phase7/f1_repetition_penalty_sweep.ipynb) | 28 | Validation-only decoding comparison |
| 7 F1 | [Learning-rate sweep](phase7/f1_learning_rate_sweep.ipynb) | 98 | Independent `5e-6` and `2e-5` runs and plots |
| 7 L1 | [LoRA sweep](phase7/lora_sweep.ipynb) | 100 | Three LoRA configurations and comparison figure |

Phase 6C's saved equivalence and diagnostic outputs are in [`artifacts/exports/phase6c/`](../../artifacts/exports/phase6c/); no executed Phase 6C notebook was found among the supplied downloads. Phase 8 has a [source notebook](../Phase-8-Final-Sealed-Test-Kaggle.ipynb), [saved-output results notebook](../Phase-8-Final-Test-Results-Public.ipynb), and [checked aggregates](../../docs/PHASE8_RESULTS.md). Its original executed Kaggle notebook has not yet been supplied as a downloaded file.

Some output cells contain prediction-level text and local run metadata. The numeric conclusions to cite are the checked phase reports under [`docs/`](../../docs/).
