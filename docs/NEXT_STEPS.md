# Narration: next steps

Updated 2 November 2026 after reconciliation of the saved P1, F1 and F1 validation-audit exports.

## Completed evidence

- Phase 3.5B is frozen: 3,629 train, 732 validation, 683 test clips; membership remains private.
- Phase 4 processor/tokenizer/model engineering and Phase 5A private validation preparation passed their recorded checks.
- **Phase 5B full-validation B0 measurement is complete:** 732 clips / 1.881808 hours, 64,137 reference code points, S/D/I 36,780 / 4,171 / 25,193; **103.12924% corpus CER**. No failed/empty predictions; 130 cap hits without EOS, 52 U+FFFD outputs, all retained in CER.
- Stale preflight plots/runtime fields were corrected from saved predictions without model inference. All six validation plots cover 732 clips; four separate processor illustrations use two training-only examples. See [Phase 5B results](PHASE5B_RESULTS.md).
- The supplied executed notebook confirms checksum/scorer/FP32/processor gates and full B0 completion. Its four processor figures were recreated anonymously from hash-checked training WAVs, without model weights or inference. Phase 5B is complete for its declared scope.

- **Phase 6A training smoke passed:** eight frozen-train clips, two actual FP16 AMP optimizer updates including checkpoint/resume, finite gradients, nonzero weight changes, and saved optimizer/scheduler/scaler/RNG state. No validation CER was measured. See [aggregate Phase 6A results](PHASE6A_RESULTS.md).
- **Phase 6B P1 completed:** 581 train clips / 1.002867 hours, three epochs, all 732 validation clips each epoch. CER improved from 159.66% to 107.26%; B0 remains better at 103.13%. Insertions remain the dominant Epoch 3 error class.
- **Phase 6C passed:** batch-4 and batch-8 validation matched single-example token/text/normalization and aggregate S/D/I results on the deterministic 12-clip audit. Batch 8 is selected; it was 2.14× faster than single-example generation with 941.7 MiB peak allocated memory. The private diagnostics show repeated sequences, high insertions and one cap-hit/no-EOS example.

- **Phase 7 F1 completed:** six epochs, patience-3 early stopping, best Epoch 3 **51.34634% CER** (32,932 / 64,137), all 732 validation clips. All 3,629 frozen training clips were used; the model started from the pinned pretrained Base. Summed epoch training/validation time was 38.47 minutes; the exported 25.48-minute wall field covers only the last resumed session.
- **F1 validation decoding audit:** explicit attention mask/task reproduced original predictions on all 732 clips. Repetition penalty 1.05 reached **48.54920% CER** (31,138 / 64,137); no-repeat 3-gram worsened CER to 53.00%. These remain validation-only results. See [F1 report](PHASE7_F1_RESULTS.md) and [aggregates](PHASE7_F1_RESULTS_AGGREGATE.json).

## Next: matched comparison and final selection

1. Run the planned LoRA follow-up independently from the pinned pretrained Base on the same frozen train/validation sets. Record training budget, decoding configuration and measured compute.
2. Select model/checkpoint and decoding settings using validation only. Distinguish F1 primary greedy CER from the separately tuned repetition-penalty candidate.
3. Predeclare the final comparison set, including B0, then prepare the private 683-clip test bundle and evaluate the selected configurations. Do not tune after opening test results.

Preserve unresolved identity/label uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication. LoRA and final-test results remain unmeasured. No test evaluation was performed during this repository sync.
