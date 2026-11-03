# Narration: next steps

Updated 3 November 2026 after completion of the Phase 7 repetition-penalty sweep.

## Completed evidence

- Phase 3.5B is frozen: 3,629 train, 732 validation, 683 test clips; membership remains private.
- Phase 4 processor/tokenizer/model engineering and Phase 5A private validation preparation passed their recorded checks.
- **Phase 5B full-validation B0 measurement is complete:** 732 clips / 1.881808 hours, 64,137 reference code points, S/D/I 36,780 / 4,171 / 25,193; **103.12924% corpus CER**. No failed/empty predictions; 130 cap hits without EOS, 52 U+FFFD outputs, all retained in CER.
- Stale preflight plots/runtime fields were corrected from saved predictions without model inference. All six validation plots cover 732 clips; four separate processor illustrations use two training-only examples. See [Phase 5B results](PHASE5B_RESULTS.md).
- The supplied executed notebook confirms checksum/scorer/FP32/processor gates and full B0 completion. Its four processor figures were recreated anonymously from hash-checked training WAVs, without model weights or inference. Phase 5B is complete for its declared scope.

- **Phase 6A training smoke passed:** eight frozen-train clips, two actual FP16 AMP optimizer updates including checkpoint/resume, finite gradients, nonzero weight changes, and saved optimizer/scheduler/scaler/RNG state. No validation CER was measured. See [aggregate Phase 6A results](PHASE6A_RESULTS.md).
- **Phase 6B P1 completed:** 581 train clips / 1.002867 hours, three epochs, all 732 validation clips each epoch. CER improved from 149.92% to 109.10%; B0 remains better at 103.13%. Insertions remain the dominant Epoch 3 error class.
- **Phase 6C passed:** batch-4 and batch-8 validation matched single-example token/text/normalization and aggregate S/D/I results on the deterministic 12-clip audit. Batch 8 is selected; it was 2.14× faster than single-example generation with 941.7 MiB peak allocated memory. The private diagnostics show repeated sequences, high insertions and one cap-hit/no-EOS example.
- **Phase 7 F1 completed:** full supervised fine-tuning used all 3,629 frozen training clips, with all 732 validation clips evaluated each epoch. Epoch 3 was selected by minimum corpus validation CER at **51.34634%** (15,940 substitutions, 4,323 deletions, 12,669 insertions; 730 EOS-terminated outputs; 2 cap hits). The run stopped after six epochs under the declared patience rule. See [Phase 7 F1 results](PHASE7_F1_RESULTS.md).
- **Phase 7 validation audit completed:** re-decoding Epoch 3 with an explicit attention mask and `task="transcribe"` reproduced the original predictions on all 732 validation clips. The complete validation-only repetition-penalty sweep selected `1.08` at **48.39172% CER** (14,542 substitutions, 5,739 deletions, 10,756 insertions; 732 EOS endings; zero cap hits). The earlier `1.05` result was 48.54920%; the margin between 1.05 and 1.08 is small and remains validation-only. `no_repeat_ngram_size=3` was worse and is not selected. See [the public sweep report](PHASE7_F1_REPETITION_PENALTY_SWEEP_RESULTS.md).

## Next: full-fine-tuning LR check, matched adaptation comparison, then sealed test

1. Run the small full-fine-tuning learning-rate comparison at `5e-6` and `2e-5`, retaining the existing `1e-5` F1 reference. Keep the same frozen split, normalization, CER and validation-selected decoding candidate.
2. Run the matched LoRA comparison from the original pinned `neurlang/ipa-whisper-base` checkpoint, not from an F1 checkpoint. LoRA may use its own declared learning rate; do not force it to use the full-fine-tuning rate.
3. Select one final model and decoding configuration using validation CER and diagnostics. Only then evaluate once on the sealed 683-clip test set. Do not use test results to tune decoding or training.

Preserve unresolved identity/label uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication. The LoRA comparison and final test remain unmeasured. The repetition-penalty improvement is validation evidence only and must not be presented as held-out test performance.
