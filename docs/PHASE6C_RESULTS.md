# Phase 6C — validation optimization and F1 preparation

**Status: PASS — validation batching is equivalent and F1 is ready to prepare.**

## Measured audit

The private audit used the saved Phase 6B P1 Epoch 3 checkpoint and 12 deterministic clips sampled across the validation-duration range. It did not train, access the test split, or change the scoring contract.

| Path | Runtime | Peak allocated GPU memory | Result |
|---|---:|---:|---|
| Single clip per `generate()` | 10.99 s | 380.6 MiB | Reference |
| Batch 4 | 5.49 s | 663.3 MiB | Exact match; 2.00× faster |
| Batch 8 | 5.14 s | 941.7 MiB | Exact match; 2.14× faster |

Both batch sizes produced identical generated token IDs, raw decoded text, normalized text, and aggregate S/D/I counts. Batch 8 is the selected validation batch size because it passed equivalence and used modest measured memory on the T4.

The effective generation policy was recorded explicitly: decoder start 50258, automatic language detection, transcribe token 50359, greedy one-beam decoding, no ordinary suppression, beginning suppression `[220, 50257]`, EOS 50257 and maximum length 448. The Transformers warning about `forced_decoder_ids` is a configuration fallback (`[[1, null], [2, 50359]]`), not an inference crash.

## What the examples show

The private diagnostic examples show a real mixture of errors: repeated syllable-like sequences, over-generation and high insertions on longer clips, and one 448-token no-EOS cap hit. This supports proceeding with the declared F1 run while keeping the existing decoding/scoring policy unchanged. It does not justify changing the CER contract or claiming that the remaining errors are purely tokenizer failures.

## F1 configuration (historical preparation)

Start independently from the original pinned `neurlang/ipa-whisper-base` revision, not from P1. Use all 3,629 frozen training clips (10.851637 h), the same 732 validation clips, learning rate `1e-5`, AdamW weight decay `0.01`, FP16 AMP training, FP32 validation, effective batch 8, validation/checkpoint once per epoch, and three initial epochs. Use physical batch 2 with gradient accumulation 4 unless T4 measurements require the declared fallback batch 1/accumulation 8. Select the checkpoint by minimum full-validation corpus NFD CER; keep the 683-clip test split sealed.

The measured Phase 6C result supports batched validation at batch 8. Actual F1 runtime must still be measured. A reasonable estimate is approximately **10 minutes training plus 18–20 minutes batched validation, plus setup/checkpoint overhead: about 30–40 minutes total** on a comparable T4. This is an estimate, not a guarantee.

Evidence: `outputs/phase6c_audit_20261102/phase6c_equivalence.json` and the private diagnostic JSONL in the same directory.

## Subsequent run

F1 later used a maximum of 10 epochs with patience 3, completed six epochs, and selected Epoch 3. The initial three-epoch plan and runtime estimate above are historical. See [verified F1 results](PHASE7_F1_RESULTS.md).
