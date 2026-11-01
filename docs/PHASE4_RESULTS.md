# Phase 4: measured engineering results and remaining diagnostic correction

Reviewed 1 November 2026. This publication contains aggregate results and plots; private audio, transcripts, predictions, raw tokens and input manifests remain local.

## Status

The v5 Kaggle run passed the core engineering checks. Loaded and installed Transformers both report 4.57.1; this resolves the mixed-version earlier run. **The v5 EOS/length-limit flags are invalid because the notebook measured postprocessed content tokens as full decoder sequences.** The source notebook is corrected as v6, but the corrected generation route has not been executed on Kaggle. No zero-shot validation/test CER or Nyishi fine-tuning result exists.

## Measured results

| Check | Result |
|---|---|
| Model and processor | neurlang/ipa-whisper-base, revision 949f7c4e4ee374b5202bc5886f6e4552e5ae2371 |
| Frozen targets | 5,044; zero round-trip, unknown-token or target-length failures |
| Set-aside references | 42; all representable, still excluded pending label review |
| Maximum reference content tokens | Frozen 358; set-aside 361; special-token-inclusive checks pass capacity 448 |
| Audio/features | 12 training-only clips, 0.75–28.82 s; finite [1,80,3000] features |
| Repeat generation/save-reload | First clip token IDs identical in both checks |
| Compute | Tesla T4; FP32; greedy batch 1; seed 0; Transformers 4.57.1; PyTorch 2.11.0+cu128 |
| Generation-loop time | 18.8245 s for 128.2169 s audio; includes repeated first inference, processing and incremental exports |
| Peak PyTorch allocated/reserved | 339.26 / 362 MiB; excludes save/reload and does not measure training memory |
| Accuracy | Not scored; poor qualitative alignment and extended repetition |

All 12 decoded predictions match the earlier mixed-version run. Tokenizer representability is not evidence of acoustic recognition competence. Set-aside labels are not reinstated by passing tokenizer checks. Repeatability here concerns one clip on one runtime. The overall Kaggle environment still has recorded dependency conflicts for unrelated installed applications; this does not claim a globally clean environment.

## Plots

![Training-only inputs and latency](../assets/phase4/inputs_timing.png)

The first clip includes warm-up effects. Several clips have similar generation times despite different audio lengths; their output token lengths are similar.

![Content-token lengths and consecutive repetition](../assets/phase4/output_lengths_repetition.png)

Six outputs contain 445 content tokens. Long identical-token runs reach 419, 376 and 211; a run of one does not rule out repeated multi-token phrases. These are output diagnostics, not CER. Sample numbers are anonymous plotting positions.

![Measured inference memory](../assets/phase4/inference_memory.png)

This establishes batch-1 inference feasibility for the measured setup, not fine-tuning fit or full-run speed.

## Stopping diagnostic correction

Whisper v4.57.1's default returned tensor excludes decoder input tokens and EOS. Therefore the earlier plot's zero EOS endings and zero cap hits cannot be interpreted as generation termination results. Six 445-content-token outputs strongly suggest the 448-token limit was reached with a three-token prefix, but full decoder sequences must confirm it. One output contains U+FFFD; preserve it in scoring and inspect its raw termination.

The v6 source requests `return_dict_in_generate=True` and reads `.sequences` for short-form generation, which the pinned implementation documents as retaining decoder prefix and EOS. This instrumentation route still needs a small equivalence/termination check on the same training examples before baseline use. [Pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/generation_whisper.py).

## Next steps

1. Run the corrected stopping instrumentation on the same training examples, verify decoded text against v5 and regenerate EOS/cap plots. Keep poor predictions and the original run evidence.
2. Verify CER on known pairs, combining marks and empty predictions. Keep NFD + trim + repeated ordinary-space collapse; retain spaces and IPA marks.
3. Prepare/hash-check the frozen validation bundle: 732 clips / 1.881808 hours. Run a small preflight, then all validation clips with resumable outputs and corpus code-point CER, S/D/I, reference characters, failures, cap hits and timing. No test-driven tuning.
4. Conduct a 1–2 hour supervised training pilot; select by validation CER and measure training memory before expanding. Reserve the 683 test clips for final evaluation.

The Phase 3.5B split was accepted and frozen locally on 1 November 2026: train 3,629 / 10.851637 h, validation 732 / 1.881808 h, test 683 / 1.891057 h. Manifests remain private. No new split or linguistic normalization is justified by this smoke run.
