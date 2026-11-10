# Public research decisions — Nyishi IPA-ASR

This is a shareable summary of the working decision register. It omits source-row identifiers and frozen membership. The private `DECISIONS.md` remains the detailed historical register in the local workspace. A status below describes the first Narration baseline, not a universal rule for later studies.

| Decision | Status in this study | Reason and evidence |
|---|---|---|
| Use Dataset A Narration for the first benchmark | Applied | Controlled speech scope; Dataset B YouTube has different content and needs rights, speech/song and alignment review. |
| Preserve source recordings, text and provenance | Applied | Derived views and frozen manifests were versioned; original records were not rewritten. |
| Use NFD, outer trim and repeated ordinary-space collapse | Applied | Mechanical formatting policy was fixed before model comparison; IPA marks and spaces remain in CER. |
| Automatically rewrite IPA symbols or remove diacritics | Rejected for this study | No expert Nyishi adjudication supports such changes. |
| Exclude missing references and unusable/empty audio from supervised examples | Applied | Each exclusion was recorded while the source row remained preserved. |
| Omit >30-second clips from the initial Whisper training view | Applied | The standard processor window is 30 seconds; no unaligned truncation was used. |
| Keep related speaker/session and duplicate-audio components together | Applied with documented identity uncertainty | This controls known structural leakage; it does not prove every speaker identity. |
| Freeze the accepted `phase3_5b_v1` split | Applied | 3,629 train, 732 validation and 683 test clips; no test-driven resplitting. |
| Use corpus-level NFD code-point CER as primary score | Applied | Sum per-clip S/D/I over summed reference code points, including spaces and IPA marks. |
| Start from pinned `neurlang/ipa-whisper-base` | Applied | IPA-output initialization was tested; no Nyishi ability was assumed from the model card. |
| Select checkpoint, learning rate and repetition penalty on validation | Applied | Final selected full fine-tuning: LR `2e-5`, Epoch 5, penalty `1.08`. |
| Compare LoRA under the same frozen evaluation contract | Applied | Best LoRA validation CER was 60.68416%; it was not selected over full fine-tuning at 47.23950%. |
| Open the frozen test only for the declared B0/F1 comparison | Applied | Both configurations were scored on 683 clips; no test-based tuning followed. |
| Compare larger or different models and optionally Dataset B | Deferred | Requires a newly declared matched protocol and, for Dataset B, separate provenance/rights and label review. |

The [sprint plan](../SPRINT_PLAN.md) records the experimental design and [end-to-end handoff](REPORT_PRESENTATION_HANDOFF.md) records the completed sequence. The [final test report](PHASE8_RESULTS.md) is the authoritative aggregate outcome.
