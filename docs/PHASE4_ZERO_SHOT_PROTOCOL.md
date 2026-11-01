# Phase 4 and zero-shot evaluation protocol

Accepted and recorded on 1 November 2026. This document fixes the initial evaluation rules before model predictions are inspected. This protocol was recorded before the model run. The 1 November Phase 4 export has since been reviewed; see `PHASE4_RESULTS.md` for measured checks. No validation/test baseline or fine-tuning has run.

## Frozen split

The accepted split is `phase3_5b_v1`. Private frozen manifests and their hashes are under `manifests/frozen_A_narration_phase3_5b_v1/`:

- `train.parquet`: 3,629 clips / 10.851637 hours / 15 components.
- `validation.parquet`: 732 clips / 1.881808 hours / 16 components.
- `test.parquet`: 683 clips / 1.891057 hours / 13 components.
- `frozen_all.parquet`: complete 5,044-row frozen table; `split_membership.csv` is the small membership index.
- `freeze_record.json` records the source proposal hash, frozen manifest hashes and structural audit. `SHA256SUMS.txt` lists the hashes for the five data manifests.

The source Phase 3.5B proposal SHA-256 is `7f7de424dedb8bbfefc9ae98520807d65170fc3a008475dc3bf1334a27c3fee4`. No known component, speaker, session or exact-audio key crosses splits, and no train-only rows occur in validation or test. One complete normalized reference string is shared between train and validation on three distinct audio clips. Record this prompt overlap; do not treat it as duplicate audio. The split is fixed for subsequent model comparisons. Any later correction requires a new version and documented rationale; do not silently overwrite this one.

## Model and zero-shot decoding

- Checkpoint: `neurlang/ipa-whisper-base` at Hub commit `949f7c4e4ee374b5202bc5886f6e4552e5ae2371`, observed on 1 November 2026. Pass this full revision to both model and processor loading; do not use mutable `main` for a reported run.
- Processor: the matching `WhisperProcessor` at the same revision. It creates Whisper log-Mel features and decodes output token IDs.
- Precision: FP32 for the first comparable zero-shot run; no quantization or mixed precision.
- Set model to evaluation mode and run inference without gradient tracking.
- Greedy decoding: `do_sample=False`, `num_beams=1`. Do not tune decoding on test.
- Use an explicit copy of the checkpoint GenerationConfig: `forced_decoder_ids=None`, `suppress_tokens=[]`, `language=None`, `task=None`, `do_sample=False`, `num_beams=1`; `_from_model_config=False`. Pass a copy into generation rather than mutating legacy model.config generation fields. Retain and record inherited `begin_suppress_tokens=[220,50257]` and length defaults. This is the measured implementation after fixing save/reload behavior; do not claim that no suppression exists at all.
- For short-form termination diagnostics, use `return_dict_in_generate=True` and `.sequences` to retain prefix/EOS. Default tensor output excludes them. The reviewed run confirms decoded-text equivalence on all 12 smoke examples. Record full token count, EOS, cap hits and U+FFFD; never drop these predictions from scoring.
- `language=None` leaves automatic language selection active. The 12 smoke outputs selected the English prefix; this is an observation, not an explicit English or Nyishi forcing decision. Log actual prefix tokens during baseline inference.
- Use the pinned checkpoint's generation length defaults; record the resolved values. Check for outputs that reach the maximum length and inspect them as possible truncations.
- Use single-clip inference for the initial smoke test and validation run. Do not request timestamps or use long-form chunking: this frozen pool is restricted to 0.5–30 seconds. Verify clip durations before feature extraction.
- Retain the original 16 kHz mono waveform and let the matched processor create model inputs. Apply no denoising, VAD trimming, pitch/level changes or other corpus audio transforms.
- Decode with special tokens skipped. Save the original decoded string as well as the normalized prediction.

The model card's recommended IPA usage leaves language/task context unforced and disables token suppression. The card describes IPA adaptation from phonemizer-generated multilingual Common Voice, which is a useful initialization but does not establish Nyishi recognition. [Pinned model card revision](https://huggingface.co/neurlang/ipa-whisper-base/commit/949f7c4e4ee374b5202bc5886f6e4552e5ae2371), [Whisper processor documentation](https://huggingface.co/docs/transformers/model_doc/whisper).

## Text and CER policy

1. Reference: `transcript_phase2` from the frozen manifest. Keep `transcript_raw` unchanged for traceability.
2. Apply Unicode NFD to the reference and prediction, trim outer whitespace, and collapse repeated ordinary U+0020 spaces to one. Apply the same function to both sides.
3. Keep spaces in the comparison. Preserve IPA letters, punctuation, dental/tone/length/aspiration marks, combining marks and speaker-specific forms. Do not fold symbols or remove diacritics.
4. Primary metric: corpus-level **code-point CER**. For each clip, calculate Levenshtein substitutions, deletions and insertions over the normalized Unicode code-point sequences. Sum S, D and I across clips, then divide by the total number of reference code points: `CER = (sum(S) + sum(D) + sum(I)) / sum(reference code points)`. Report as both a fraction and percentage, with all counts.
5. Do not concatenate clips before computing edit distance; score each reference/prediction pair so errors cannot shift across clip boundaries. A failed decode or empty prediction remains in the evaluation: count its reference characters as deletions and include the clip in the denominator. Report failures separately.
6. WER is not the headline metric because word boundaries and spaces may not be consistently linguistic units in these labels. Any space-stripped or convention-tolerant result must be a separately named diagnostic, never a replacement for the primary CER.
7. If output inspection shows the checkpoint is emitting non-IPA or otherwise incompatible text, label CER as diagnostic and do not rank it as a valid IPA score.

## Run order and test protection

1. Resolve and record the exact Python, PyTorch, Transformers, Datasets, Evaluate/jiwer, CUDA and GPU versions on the selected runtime. The model revision and all generation settings above are fixed; software versions must be recorded with the run because they can affect generation behavior.
2. Verify processor loading, audio features, full-label tokenization and decode round trips on the 5,044 frozen labels. Also check the 42 set-aside symbol-review labels for representability, but do not add them to the frozen training or evaluation pool. Check each complete label against the model's target-length limit; no truncation or silent token suppression is allowed.
3. Smoke test on 12 varied clips drawn from **training only**. Check audio loading, generated output, runtime, GPU memory and repeatability. This is pipeline debugging, not a score.
4. Once the processor and output format pass, run zero-shot on all validation clips. Save one row per sample with reference, raw hypothesis, normalized hypothesis, edit counts, status and runtime. Inspect errors and examples; any decoding-policy changes are recorded and validation may be rerun.
5. Keep test predictions unopened during processor debugging, training and checkpoint selection. At final evaluation, run both the pinned zero-shot checkpoint and the selected fine-tuned checkpoint on the frozen test set with the same text normalization, CER code and declared generation settings. Do not use test results to tune or choose between settings.

Report sample count, reference code points, S/D/I, CER, evaluated audio hours, empty/decode failures, runtime and real-time factor, plus model/processor revision, software versions, hardware, manifest hashes and configuration. Store predictions and run artifacts locally; do not publish private manifests, audio or personal identifiers.
