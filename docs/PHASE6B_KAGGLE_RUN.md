# Phase 6B Kaggle P1 pilot — prepared, not run

Phase 6A passed its two-update training and checkpoint-resume engineering gate. Phase 6B is the planned **controlled one-hour supervised pilot**. It starts again from the pinned pretrained IPA-Whisper Base checkpoint, trains on a fixed subset of the frozen training split, and selects an epoch by corpus CER on the **entire 732-clip validation set**. It does not use the 683-clip test set or carry Phase 6A weights forward.

## Private inputs to upload

Create **two private Kaggle Datasets** and attach both with **Add Input**:

1. **Pilot training:** local private file `outputs/phase6b_pilot_1h_v1/phase6b_pilot_train_bundle.zip`, about 60 MB. It contains 581 full frozen-train clips, 1.002867 hours, across all 15 training components, plus a private manifest, references and checksums. Its private membership hash is checked by the notebook. It is a deterministic subset prepared by [`scripts/prepare_phase6b_pilot.py`](../scripts/prepare_phase6b_pilot.py); do not regenerate it with a different selection after looking at validation results.
2. **Full validation:** existing local private file `outputs/phase5a_validation_bundle_20261101_155452_045404_9fb2cd/phase5_validation_bundle.zip`, about 139 MB. It contains all 732 frozen validation clips, 1.881808 hours and 64,137 normalized reference code points. Reuse the exact B0 bundle; do not upload test or the full raw/master dataset.

The notebook locates extracted `bundle_record.json` markers below `/kaggle/input`. If Kaggle exposes a ZIP as a file, the notebook verifies its paths and extracts it into `/kaggle/working/phase6b_extracted_inputs/`. Keep both datasets and the Kaggle notebook/output **private**: the manifests and per-clip predictions contain IDs and transcript text. Do not put the archives, extracted folders, checkpoint, executed notebook or prediction JSONL in GitHub.

## Kaggle run

1. Import [`Phase-6B-Kaggle-Pilot.ipynb`](../notebooks/Phase-6B-Kaggle-Pilot.ipynb). Select **Tesla T4 GPU** and enable Internet to fetch the pinned public model. The first cell pins Transformers `4.57.1`, Tokenizers `0.22.2`, Hub `0.36.2` and Safetensors `0.8.0`, leaving Kaggle PyTorch/CUDA intact. If packages install, restart the session and run from the top. The notebook checks for the measured PyTorch `2.11.0` stack and T4 and stops if either changed.
2. Run the input cells. They verify every file checksum, frozen membership, train/validation separation, NFD references, label round trips, full audio and `[80,3000]` features. An input mismatch must be fixed before training.
3. Run the remaining cells. The declared pilot is seed 0, **3 epochs maximum**, micro-batch 2, accumulation 4, AdamW `1e-5`/weight decay `0.01`, linear decay with 5% warmup, FP16 AMP training and FP32 greedy validation. It evaluates all 732 validation clips at each epoch and saves each epoch's model and resumable optimizer/scheduler/scaler/RNG state. Selection is the minimum full-validation corpus CER. Runtime and memory are measured, not promised.
4. Keep the explicit B0 generation configuration in the notebook: no forced decoder IDs, no language/task override, greedy decoding, `suppress_tokens=[]`, inherited `begin_suppress_tokens=[220,50257]`, maximum 448 tokens. Phase 6A's saved model default differed; this override prevents an undeclared evaluation-policy change.
5. Output is `/kaggle/working/phase6b_P1_private/`. Save the executed notebook, `P1_report_private.json`, `aggregate_history.json`, all private validation JSONL files, and the epoch checkpoints in private storage. For a resumed session, attach the prior checkpoint as another private Kaggle input and set `RESUME_CHECKPOINT` to its extracted `epoch_XX` directory. Resume only the same declared run; the fingerprint checks data and configuration. Preserve the earlier best checkpoint if it was in a previous session.

**Stop and review** if input/hash/target checks fail, loss or gradients remain nonfinite, no real update occurs, repeated validation decode failures appear, or a checkpoint cannot reload. If GPU memory is insufficient, restart a clean run with micro-batch 1 and accumulation 8 and record the deviation. A runtime limit can be handled at an epoch boundary using the saved checkpoint; do not call an incomplete pilot complete.

## Gate and next decision

The run must report all 581 training clips, all 732 validation clips each epoch, exact S/D/I and 64,137-character denominator, validation CER history, decode failures/cap hits, updates, elapsed time, peak allocated GPU memory and a reloadable checkpoint. A lower training loss alone is not evidence of improved transcription. Compare the best validation CER with B0's **103.12924%** using the same policy, inspect a small private error sample, and decide whether the recipe and measured resource cost support the full **10.851637-hour F1** run. F1 must start independently from the pinned pretrained checkpoint. Final test remains sealed.
