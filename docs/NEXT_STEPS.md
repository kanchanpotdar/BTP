# Narration: next steps

Updated 10 November 2026 after the learning-rate and LoRA comparisons, and preparation of the sealed test evaluator. Test results remain pending.

## Completed evidence

- Phase 3.5B is frozen: 3,629 train, 732 validation, 683 test clips; membership remains private.
- Phase 4 processor/tokenizer/model engineering and Phase 5A private validation preparation passed their recorded checks.
- **Phase 5B full-validation B0 measurement is complete:** 732 clips / 1.881808 hours, 64,137 reference code points, S/D/I 36,780 / 4,171 / 25,193; **103.12924% corpus CER**. No failed/empty predictions; 130 cap hits without EOS, 52 U+FFFD outputs, all retained in CER.
- Stale preflight plots/runtime fields were corrected from saved predictions without model inference. All six validation plots cover 732 clips; four separate processor illustrations use two training-only examples. See [Phase 5B results](PHASE5B_RESULTS.md).
- The supplied executed notebook confirms checksum/scorer/FP32/processor gates and full B0 completion. Its four processor figures were recreated anonymously from hash-checked training WAVs, without model weights or inference. Phase 5B is complete for its declared scope.

- **Phase 6A training smoke passed:** eight frozen-train clips, two actual FP16 AMP optimizer updates including checkpoint/resume, finite gradients, nonzero weight changes, and saved optimizer/scheduler/scaler/RNG state. No validation CER was measured. See [aggregate Phase 6A results](PHASE6A_RESULTS.md).
- **Phase 6B P1 completed:** 581 train clips / 1.002867 hours, three epochs, all 732 validation clips each epoch. The latest checked rerun improved from 159.66% to 107.26% CER; B0 remains better at 103.13%. Insertions remain the dominant Epoch 3 error class.
- **Phase 6C passed:** batch-4 and batch-8 validation matched single-example token/text/normalization and aggregate S/D/I results on the deterministic 12-clip audit. Batch 8 is selected; it was 2.14× faster than single-example generation with 941.7 MiB peak allocated memory. The private diagnostics show repeated sequences, high insertions and one cap-hit/no-EOS example.
- **Phase 7 F1 completed:** full supervised fine-tuning used all 3,629 frozen training clips, with all 732 validation clips evaluated each epoch. Epoch 3 was selected by minimum corpus validation CER at **51.34634%** (15,940 substitutions, 4,323 deletions, 12,669 insertions; 730 EOS-terminated outputs; 2 cap hits). The run stopped after six epochs under the declared patience rule. See [Phase 7 F1 results](PHASE7_F1_RESULTS.md).
- **Phase 7 validation audit completed:** re-decoding Epoch 3 with an explicit attention mask and `task="transcribe"` reproduced the original predictions on all 732 validation clips. The complete validation-only repetition-penalty sweep selected `1.08` at **48.39172% CER** (14,542 substitutions, 5,739 deletions, 10,756 insertions; 732 EOS endings; zero cap hits). The earlier `1.05` result was 48.54920%; the margin between 1.05 and 1.08 is small and remains validation-only. `no_repeat_ngram_size=3` was worse and is not selected. See [the public sweep report](PHASE7_F1_REPETITION_PENALTY_SWEEP_RESULTS.md).

## Subsequent completed validation evidence

- **Full fine-tuning learning-rate comparison:** `5e-6` reached 51.05477% validation CER at Epoch 3; the earlier `1e-5` reference with repetition penalty 1.08 reached 48.39172%; `2e-5` reached **47.23950%** at Epoch 5. See [the matched comparison](PHASE7_F1_LR_RESULTS.md).
- **LoRA comparison:** three independent configurations were measured. Best LoRA was rank 8, learning rate `1e-4`, Epoch 7 at 60.68416% validation CER. It was not selected over full fine-tuning. See [the LoRA report](PHASE7_LORA_RESULTS.md).
- **Final selection:** `neurlang/ipa-whisper-base` full fine-tuning at `2e-5`, Epoch 5, repetition penalty 1.08. This decision uses validation only.

## Next: finish the sealed test and document the result

1. Run the locally prepared Phase 8 self-contained notebook on the frozen 683-clip test bundle for the predeclared B0 and selected F1 configurations.
2. Verify both aggregate and prediction archives cover all 683 clips, reconcile S/D/I and reference-character denominators, and inspect generation failures. Do not choose a different checkpoint or decoding setting using test scores.
3. Publish a concise Phase 8 report with the measured CER, runtime, error patterns, and uncertainty once the saved test outputs are available.

Preserve unresolved identity/label uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication. No test CER is claimed by this documentation update.
