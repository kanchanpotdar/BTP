# Narration: next steps

Updated 1 November 2026 after verification of the Phase 6A Kaggle export and preparation of the Phase 6B pilot.

## Completed evidence

- Phase 3.5B is frozen: 3,629 train, 732 validation, 683 test clips; membership remains private.
- Phase 4 processor/tokenizer/model engineering and Phase 5A private validation preparation passed their recorded checks.
- **Phase 5B full-validation B0 measurement is complete:** 732 clips / 1.881808 hours, 64,137 reference code points, S/D/I 36,780 / 4,171 / 25,193; **103.12924% corpus CER**. No failed/empty predictions; 130 cap hits without EOS, 52 U+FFFD outputs, all retained in CER.
- Stale preflight plots/runtime fields were corrected from saved predictions without model inference. All six validation plots cover 732 clips; four separate processor illustrations use two training-only examples. See [Phase 5B results](PHASE5B_RESULTS.md).
- The supplied executed notebook confirms checksum/scorer/FP32/processor gates and full B0 completion. Its four processor figures were recreated anonymously from hash-checked training WAVs, without model weights or inference. Phase 5B is complete for its declared scope.

- **Phase 6A training smoke passed:** eight frozen-train clips, two actual FP16 AMP optimizer updates including checkpoint/resume, finite gradients, nonzero weight changes, and saved optimizer/scheduler/scaler/RNG state. No validation CER was measured. See [aggregate Phase 6A results](PHASE6A_RESULTS.md).
- **Phase 6B is prepared, not run:** fixed 581-clip / 1.002867-hour train subset across all 15 training components, source notebook and [private Kaggle run guide](PHASE6B_KAGGLE_RUN.md). All 732 validation clips remain the evaluation set; test remains sealed.

## Next: controlled pilot

1. Upload the private Phase 6B train ZIP and existing private Phase 5A validation ZIP to Kaggle as two private datasets. Follow the guide's hash, label, model and policy checks.
2. Train from pinned pretrained Base using the declared pilot recipe; evaluate every epoch on full validation in FP32 with the B0 greedy generation policy. The Phase 6A saved default forced-decoder configuration must not silently change validation decoding.
3. Select by corpus CER, report S/D/I and all denominators, inspect private error samples, and measure complete runtime and memory. Do not infer pilot performance from training loss alone.
4. If stable, start F1 independently on all 3,629 frozen training clips, then compare LoRA as planned. Keep the 683 test clips untouched until final selection.

Preserve unresolved identity/label uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication. No Phase 6B, F1, LoRA or final-test result has been measured.
