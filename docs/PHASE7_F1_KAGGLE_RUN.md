# Phase F1 Kaggle run instructions

F1 is the first full supervised experiment. It starts from the original pinned `neurlang/ipa-whisper-base` checkpoint; it does **not** warm-start from P1.

## Private inputs

Attach three private Kaggle inputs:

1. A full frozen-training bundle with `bundle_type=phase7_f1_train_v1`, containing all 3,629 training clips, `train_manifest.csv`, `bundle_record.json`, and `SHA256SUMS.txt`.
2. The existing Phase 5 validation bundle with `bundle_type=phase5_validation_baseline`, containing all 732 validation clips.
3. `scripts/phase7_f1_train.py`.

Do not attach the test bundle, the master manifest, prediction exports, or the P1 checkpoint. The runner discovers extracted Kaggle directories and ZIP inputs by their bundle markers.

## Declared configuration

- Physical training batch: 2 clips.
- Gradient accumulation: 4 micro-batches.
- Effective training batch: 8 clips per optimizer update.
- Validation batch: 8 (proved equivalent to single-example decoding in Phase 6C).
- Maximum epochs: 10.
- Early stopping: 3 consecutive non-improving validation CER values.
- Minimum improvement: zero initially.
- Learning rate: `1e-5`; AdamW weight decay: `0.01`.
- FP16 AMP training; FP32 feature extraction and validation.
- Validation and checkpoint once per epoch.
- Best checkpoint: minimum corpus-level NFD character error rate.

The runner uses lazy CPU feature extraction, so it does not retain a multi-gigabyte feature tensor for all 3,629 clips. GPU tensors are released after every micro-batch and validation batch. If a T4 OOM occurs, restart cleanly and rerun with physical batch 1 and accumulation 8; this preserves effective batch 8.

## Run

Open `notebooks/Phase-7-F1-Kaggle-Training.ipynb`, choose a T4 GPU, enable Internet only to fetch the pinned model if necessary, and run all cells. The notebook invokes:

```bash
python scripts/phase7_f1_train.py \
  --physical-batch 2 --accumulation 4 --validation-batch 8 \
  --max-epochs 10 --patience 3 --learning-rate 1e-5 --weight-decay 0.01
```

Progress bars show extraction, training updates, validation batches, epoch duration, CER, peak memory, and early-stopping status. Private outputs are written to `/kaggle/working/phase7_F1_private/` and include epoch checkpoints, resumable training state, aggregate history, private validation predictions, and `F1_report_private.json`.

No test data is read. Do not publish the F1 output directory or checkpoint ZIP.
