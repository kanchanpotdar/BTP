# Phase 7 L1 — LoRA comparison results

## Scope and evaluation

Three independent LoRA runs started from the pinned `neurlang/ipa-whisper-base` revision `949f7c4e4ee374b5202bc5886f6e4552e5ae2371`. Each used the frozen 3,629-clip training set and evaluated all 732 validation clips (64,137 reference code points). The 683-clip test set was not used. Runs used the same NFD/whitespace normalization, corpus CER implementation, validation batch 8, and repetition penalty 1.08.

Training configuration: effective batch 8 (physical batch 2 × accumulation 4), FP16 AMP, AdamW weight decay 0.01, at most 10 epochs, validation once per epoch, patience 3, and best-checkpoint selection by full-validation corpus CER. The rank-8 configurations train 294,912 of 72,888,832 parameters (0.405%); rank 16 trains 589,824 parameters (0.809%).

## Results

| LoRA configuration | Best epoch | Best validation CER | S | D | I | Training time | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rank 8, alpha 16, LR `1e-4` | 7 / 10 | **60.68416%** | 17,154 | 4,538 | 17,229 | 71.872 min | 1,216.46 MiB |
| Rank 8, alpha 16, LR `2e-5` | 7 / 10 | 65.79509% | 20,124 | 4,354 | 17,721 | 71.568 min | 1,216.46 MiB |
| Rank 16, alpha 32, LR `1e-4` | 3 / 6 | 61.12540% | 17,491 | 4,274 | 17,439 | 42.844 min | 1,222.51 MiB |

Times are the saved run-level training-plus-validation wall time. Rank 8 at `1e-4` and rank 8 at `2e-5` reached the 10-epoch ceiling; rank 16 stopped at epoch 6 after three consecutive validations without improvement. The best checkpoint is selected by CER, not by the final epoch.

## Comparison with full fine-tuning

The validation-selected full fine-tuning candidate used learning rate `2e-5`, Epoch 5, and repetition penalty 1.08. It reached **47.23950% validation CER**, with S/D/I = 14,503 / 5,583 / 10,212, over the same 732 clips and 64,137 reference code points.

The best LoRA run (rank 8, LR `1e-4`) is 13.44466 absolute CER points higher than this full fine-tuning candidate. It used less GPU memory, but has substantially more substitutions and insertions. Rank 16 did not improve on rank 8 at `1e-4`; changing LoRA's learning rate to the full fine-tuning rate `2e-5` also did not improve the result. **LoRA is complete as a comparison and was not selected.** The current validation-selected model remains full fine-tuning at `2e-5`, Epoch 5.

These are validation comparisons, not test-set estimates. Do not tune the model or decoding policy on the sealed test set.

## Reproducibility and privacy

The results were transcribed from the executed `lora-sweep-for-btp.ipynb` outputs. Training was not rerun to prepare this report. Public reporting contains aggregate metrics only. Audio, manifests, IDs, raw/normalized prediction text, checkpoints, optimizer state, and provenance remain private.
