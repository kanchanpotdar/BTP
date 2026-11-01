# Narration: next steps

Updated 1 November 2026 after Phase 5B B0 saved-result verification.

## Completed evidence

- Phase 3.5B is frozen: 3,629 train, 732 validation, 683 test clips; membership remains private.
- Phase 4 processor/tokenizer/model engineering and Phase 5A private validation preparation passed their recorded checks.
- **Phase 5B full-validation B0 measurement is complete:** 732 clips / 1.881808 hours, 64,137 reference code points, S/D/I 36,780 / 4,171 / 25,193; **103.12924% corpus CER**. No failed/empty predictions; 130 cap hits without EOS, 52 U+FFFD outputs, all retained in CER.
- Stale preflight plots/runtime fields were corrected from saved predictions without model inference. All six validation plots cover 732 clips; four separate processor illustrations use two training-only examples. See [Phase 5B results](PHASE5B_RESULTS.md).
- The supplied executed notebook confirms checksum/scorer/FP32/processor gates and full B0 completion. Its four processor figures were recreated anonymously from hash-checked training WAVs, without model weights or inference. Phase 5B is complete for its declared scope.

## Next: supervised training smoke and pilot

1. Keep the B0 checkpoint/configuration, frozen membership and scoring fixed. Review selected high/low-CER, capped and Unicode-error outputs privately against audio and references, preserving label uncertainty.
2. Preserve the supplied executed notebook privately and use the corrected anonymous publication figures. Private references, predictions, audio, manifests, IDs and source provenance remain excluded from publication.
3. Before Phase 6A, specify decoder-prefix, label shifting and padding-mask policy without inventing a Nyishi token. Any changed decoding policy is a separate declared validation experiment.
4. Complete Phase 6A: finite FP16-AMP loss and gradients, a verified optimizer update, and checkpoint/resume including optimizer/scaler state. These checks are not run yet.
5. Then run a component-aware 1–2 hour Phase 6B training pilot, select by validation CER and measure training speed/memory. Full fine-tuning and matched LoRA follow only after a stable pilot; improvement and resource fit remain unmeasured.
6. Keep all 683 test clips untouched until final selection. Preserve unresolved identity/label uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication.

See the local sprint plan for provisional training settings and [the protocol](PHASE4_ZERO_SHOT_PROTOCOL.md) for the evaluation contract. No Nyishi training or final test evaluation has run.
