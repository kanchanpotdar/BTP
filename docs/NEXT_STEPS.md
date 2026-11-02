# Narration: next steps

Updated 2 November 2026 after review of the completed Phase 6B P1 export.

## Completed evidence

- Phase 3.5B is frozen: 3,629 train, 732 validation, 683 test clips; membership remains private.
- Phase 4 processor/tokenizer/model engineering and Phase 5A private validation preparation passed their recorded checks.
- **Phase 5B full-validation B0 measurement is complete:** 732 clips / 1.881808 hours, 64,137 reference code points, S/D/I 36,780 / 4,171 / 25,193; **103.12924% corpus CER**. No failed/empty predictions; 130 cap hits without EOS, 52 U+FFFD outputs, all retained in CER.
- Stale preflight plots/runtime fields were corrected from saved predictions without model inference. All six validation plots cover 732 clips; four separate processor illustrations use two training-only examples. See [Phase 5B results](PHASE5B_RESULTS.md).
- The supplied executed notebook confirms checksum/scorer/FP32/processor gates and full B0 completion. Its four processor figures were recreated anonymously from hash-checked training WAVs, without model weights or inference. Phase 5B is complete for its declared scope.

- **Phase 6A training smoke passed:** eight frozen-train clips, two actual FP16 AMP optimizer updates including checkpoint/resume, finite gradients, nonzero weight changes, and saved optimizer/scheduler/scaler/RNG state. No validation CER was measured. See [aggregate Phase 6A results](PHASE6A_RESULTS.md).
- **Phase 6B P1 completed:** 581 train clips / 1.002867 hours, three epochs, all 732 validation clips each epoch. CER improved from 149.92% to 109.10%; B0 remains better at 103.13%. Insertions remain the dominant Epoch 3 error class.
- **Phase 6C passed:** batch-4 and batch-8 validation matched single-example token/text/normalization and aggregate S/D/I results on the deterministic 12-clip audit. Batch 8 is selected; it was 2.14× faster than single-example generation with 941.7 MiB peak allocated memory. The private diagnostics show repeated sequences, high insertions and one cap-hit/no-EOS example.

## Next: validation optimization, then F1

1. Run F1 independently on all 3,629 frozen training clips with validation batch 8. Keep test sealed and preserve the same CER/normalization contract.
2. Select the F1 checkpoint by minimum corpus validation CER and inspect its private error diagnostics.
3. Run LoRA only as the matched follow-up comparison; it is not the primary fix for validation runtime.

Preserve unresolved identity/label uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication. F1, LoRA and final-test results remain unmeasured.
