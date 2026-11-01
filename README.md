# Nyishi Narration ASR research

**Dataset A Narration · IPA transcription · reproducible speaker/session splits**

This project builds a Nyishi speech-to-IPA baseline, then measures whether supervised adaptation improves it. The first checkpoint is **`neurlang/ipa-whisper-base`**. Dataset B YouTube is deferred.

## Current status

| Work | Status |
|---|---|
| Inventory, eligibility and minimal transcript formatting | Completed through Phase 2; source data preserved |
| Targeted label/speaker review | Conservative exclusions recorded; expert adjudication remains pending |
| Phase 3.5B split | Accepted, frozen and hashed locally |
| Phase 4 processor/model smoke test | **Engineering checks complete**, including corrected EOS/cap diagnostics |
| Phase 5A validation-input preparation | **Completed and independently verified**; private bundle verified and used for B0 |
| Phase 5B B0 validation CER | **732/732 scored; 103.13% CER; reporting corrected** |
| Phase 6A training smoke | **Passed** on eight frozen-train clips: two verified optimizer updates across save/reload; no CER |
| Phase 6B P1 pilot | **Prepared, not run**: 581 train clips / 1.002867 h; full 732-clip validation planned |
| Full Nyishi fine-tuning and final test evaluation | **Not run** |

**Latest finding:** the full B0 zero-shot validation measurement is complete: **66,144 edits / 64,137 NFD reference code points = 103.13% CER** over **732 clips / 1.881808 hours**. This is poor recognition under the declared IPA policy. **130 outputs hit the 448-token cap without EOS; 52 contain U+FFFD.** All remain scored. Corrected metrics and six full-validation plots were regenerated from saved predictions without model inference. The supplied executed notebook confirms the processor/scorer/runtime gates; four anonymous processor illustrations are now included. Phase 6A has since passed its training engineering gate, while the Phase 6B accuracy pilot remains unrun.

### Navigate

[Phase 0](#phase-0-inventory-and-local-setup) · [Phase 1](#phase-1-eligibility-and-audio-checks) · [Phase 2](#phase-2-transcript-formatting) · [Phase 2.5](#phase-25-targeted-review) · [Phase 3](#phase-3-first-group-split-proposal) · [Phase 3.5](#phase-35-category-aware-revision) · [Phase 3.5B](#phase-35b-accepted-and-frozen-split) · [Phase 4](#phase-4-processor-and-model-smoke-test) · [Phase 5A](#phase-5a-validation-input-preparation) · [Phase 5B](#phase-5b-b0-full-validation-zero-shot-baseline) · [Phase 6A](#phase-6a-supervised-training-smoke) · [Phase 6B](#phase-6b-controlled-p1-pilot-prepared) · [Next steps](#next-steps-after-the-pilot)

<a href="notebooks/Phase-4B-Results-Public.ipynb"><ins>Open the Phase 4 results notebook with saved figures</ins></a> · <a href="docs/PHASE4_RESULTS.md"><ins>Read the full Phase 4 report</ins></a>

## Research rules

- **Scope:** Narration only; preserve raw audio, transcripts, IDs and provenance.
- **References:** Unicode NFD, outer whitespace trim and collapse repeated ordinary spaces. Preserve IPA letters, punctuation, tone/length/dental/aspiration marks and speaker-specific forms.
- **Splits:** whole speaker/session/duplicate-audio components; uncertain identities excluded from held-out data under the recorded conservative policy.
- **Evaluation:** corpus-level NFD **code-point CER**, including spaces, with substitutions/deletions/insertions and all denominators. Validation selects settings/checkpoints; test is reserved for final evaluation.
- **Evidence:** distinguish historical analysis, proposals, verified results and work not run. Tokenizer support does not prove Nyishi recognition.

## Phase 0: inventory and local setup

<a href="notebooks/Phase-0.ipynb"><ins>Open Phase 0 notebook and saved outputs</ins></a>

### What was established

- **6,718 Narration rows**, 59 Parquet shards and a **54-column** starting master (`master_A_v1_1.parquet`).
- Initial flags: 5,651 rows with references, 1,067 missing, 6,714 marked audio-OK, three zero-duration clips and 708 clips over 30 seconds.
- Combining initial labelled/audio-OK/0.5–30 s conditions yielded **5,086 candidates**, before final review and splits.
- Local setup at that time: Python 3.11.1, NumPy 2.3.5, pandas 2.2.3, PyArrow 26.0.0 and SoundFile 0.13.1. Model packages were not yet installed locally. The later Kaggle environment is recorded under Phase 4.

### Figure: inventory counts

![Narration inventory counts](assets/narration_inventory.svg)

**Meaning:** the master is a preserved source table. Initial flags and candidate counts are not a frozen training-set definition.

## Phase 1: eligibility and audio checks

<a href="notebooks/Phase-1.ipynb"><ins>Open Phase 1 notebook and saved outputs</ins></a>

### Filtering waterfall

The 6,718 master rows joined one-to-one with full-decode facts. The audit recorded no full-decode failures or sample-rate/channel/decoded-duration disagreements.

| First applicable outcome | Clips |
|---|---:|
| Eligible 0.5–30 s candidates | **5,086** |
| Missing/empty reference | 1,067 |
| Outside initial duration range | 561 |
| Zero audio frames | 3 |
| Duration discrepancy held for review | 1 |

The candidates contain **14.826 decoded hours**. Duration exclusions comprise 528 labelled clips over 30 s and 33 labelled, nonempty clips below 0.5 s. The separate zero-frame and duration-discrepancy exclusions avoid double counting. **967 candidate rows** are provisionally train-only because of pair linkage, multiple speakers or uncertain identity. Source hashes were unchanged.

### Figures: eligibility and duration

![Phase 1 eligibility reasons](assets/phase1_eligibility_reasons.png)

**Meaning:** missing references and duration restrictions account for most exclusions. Source records remain preserved.

![Phase 1 duration distribution](assets/phase1_duration_distribution.png)

**Meaning:** duration varies substantially; first-pass Whisper examples are limited to 0.5–30 s. This figure was separately generated from the saved Phase 1 audit, rather than an executed notebook plot cell.

### Figures: speaker coverage and identity status

![Phase 1 speaker hours](assets/phase1_speaker_hours.png)

**Meaning:** unequal speaker hours make row counts alone insufficient for split balance.

![Phase 1 speaker status](assets/phase1_speaker_status.png)

**Meaning:** uncertain or linked identities require conservative grouping; their flags do not mean the recordings are silent or unusable for training.

## Phase 2: transcript formatting

<a href="notebooks/Phase-2.ipynb"><ins>Open Phase 2 notebook and saved outputs</ins></a>

### Policy and measured changes

The derived `transcript_phase2` retains `transcript_raw` and applies **NFD + outer trim + repeated ordinary-space collapse**.

- **497 rows changed (9.77%)**, entirely through repeated-space removal.
- Character total: **522,368 → 521,747**, a reduction of 621 ordinary spaces.
- NFD and outer trimming changed zero rows in this run; the policy was idempotent.
- Both candidate views contain 102 distinct code points. Source hashes were unchanged.

### Figures: formatting changes and label lengths

![Phase 2 formatting and label lengths](assets/phase2_normalization_and_length.png)

**Meaning:** these are mechanical formatting changes. Rare marks are retained because frequency does not determine linguistic importance.

![Phase 2 audio duration versus label length](assets/phase2_duration_vs_label_length.png)

**Meaning:** reference length and audio duration are useful diagnostics; they do not establish transcript correctness or model accuracy.

Plain `g`/IPA `ɡ`, dental marks, nasalization, length marks and rare accents are not automatically merged or deleted. Historical A_v1/A_v2 replacements remain provisional alternatives.

## Phase 2.5: targeted review

<a href="notebooks/Phase-2.5.ipynb"><ins>Open Phase 2.5 notebook and saved outputs</ins></a>

### Conservative decisions

- **42 clips / 0.201 hours** were set aside: 41 contain `:`, `;` or `[`, and one contains combining acute accents.
- The review also lists five speaker-merge proposals, 967 provisional train-only rows across nine groups and one session connecting two proposed groups.
- Expert review remains pending; original records and user review fields are preserved.

### Figure: review workload

![Phase 2.5 review queue](assets/phase2_5_review_workload.png)

**Meaning:** the queue identifies uncertainty to review, not symbols to delete. Phase 4 subsequently confirmed that all 42 set-aside labels are tokenizable; that does not resolve their linguistic correctness.

## Phase 3: first group split proposal

<a href="notebooks/Phase-3.ipynb"><ins>Open Phase 3 notebook and saved outputs</ins></a>

### Grouping and proposal

Setting aside 42 clips left **5,044 clips / approximately 14.625 hours**. Connections through proposed speaker groups, speaker keys, sessions and exact decoded-waveform hashes produced **44 indivisible components**.

| Proposed partition | Clips | Hours |
|---|---:|---:|
| Train | 2,876 | 10.50 |
| Validation | 1,428 | 2.19 |
| Test | 740 | 1.93 |

Known speaker/session/exact-audio overlap was zero, and no train-only rows were held out. One full reference string occurred across partitions on different audio: **prompt overlap**, not duplicate audio.

### Figures: split size and category balance

![Phase 3 split sizes](assets/phase3_split_sizes.png)

![Phase 3 category balance](assets/phase3_category_balance.png)

**Decision:** this proposal was not frozen. Validation was **52.9% under 3 s**; test was **96.4% November 2024** by hours. Short-clip balance and recording-condition shifts needed review.

## Phase 3.5: category-aware revision

<a href="notebooks/Phase-3.5.ipynb"><ins>Open Phase 3.5 notebook and saved outputs</ins></a>

### What changed

The search considered whole components, hours, month, age, duration and gender. It proposed **3,055 train / 542 validation / 1,447 test** clips, approximately **10.23 / 2.20 / 2.19 hours**. Known structural overlap remained zero.

### Figures: revised balance

![Phase 3.5 split comparison](assets/phase3_5_split_comparison.png)

![Phase 3.5 category comparison](assets/phase3_5_category_comparison.png)

**Decision:** not frozen. One component supplied **90.5% of validation hours**, validation was **98.8% female** by hours, and **52.3% of test clips were under 3 s**. A better internal search score did not make this a better evaluation design.

## Phase 3.5B: accepted and frozen split

<a href="notebooks/Phase-3.5B.ipynb"><ins>Open Phase 3.5B notebook</ins></a> · <a href="scripts/phase3_5b_proposal.py"><ins>View the proposal code</ins></a>

### Final partition

The proposal kept train-only and dominant components in training, then balanced held-out hours, gender, component influence and short clips. **Accepted and frozen locally on 1 November 2026.** Membership and SHA-256 hashes are retained privately.

| Partition | Clips | Hours | Components | Female hours | Clips under 3 s | Largest component's hours |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 3,629 | 10.851637 | 15 | 59.7% | 24.0% | 33.7% |
| **Validation** | 732 | 1.881808 | 16 | 50.9% | 19.9% | 24.1% |
| **Test** | 683 | 1.891057 | 13 | 50.9% | 13.8% | 33.7% |

### Figure: comparison of split strategies

![Comparison of the three split strategies](assets/phase3_5b_proposal_comparison.png)

**Why selected:** a more usable balance than the earlier proposals while preserving known component/speaker/session/exact-audio separation. No train-only rows enter validation/test.

**Remaining limitations:** November represents 75.6% of validation and 67.8% of test hours versus 10.6% of training. One complete reference string overlaps train/validation on distinct audio. Proposed identities lack full expert adjudication. Report these limitations; do not redesign the test set after seeing scores.

## Phase 4: processor and model smoke test

<a href="notebooks/Phase-4B-Kaggle-Smoke-Test.ipynb"><ins>Open the Kaggle source notebook</ins></a> · <a href="notebooks/Phase-4B-Results-Public.ipynb"><ins>Open saved public results and plots</ins></a> · <a href="docs/PHASE4_RESULTS.md"><ins>Read the full review</ins></a>

### Compatibility and reproducibility: passed

- Pinned model/processor: **`neurlang/ipa-whisper-base`**, revision `949f7c4e4ee374b5202bc5886f6e4552e5ae2371`.
- **5,044 frozen + 42 set-aside references:** zero round-trip, unknown-token or target-length failures. The 42 remain excluded pending review.
- **12 training-only audio clips:** valid finite `[1,80,3000]` features; repeated first-clip generation and checkpoint reload return identical tokens.
- **Stack:** Transformers 4.57.1, PyTorch 2.11.0+cu128, Tesla T4, FP32, greedy batch 1, seed 0.

### Generation quality: limitations confirmed

| Diagnostic | Measured result |
|---|---:|
| Ended with EOS | **6/12** |
| Reached 448-token limit without EOS | **6/12** |
| Contains Unicode replacement character | **1/12** |
| Longest identical-token runs | **419, 376, 211** |
| Generation-loop wall time | **19.41 s** for 128.22 s audio |
| Peak PyTorch allocated/reserved memory | **380.57 / 408 MiB** |
| Validation/test CER or Nyishi training | **Not run** |

The corrected full-sequence diagnostics are now verified. The earlier zero EOS/cap flags were an instrumentation error. All 12 decoded hypotheses remain unchanged after the correction: the model still shows omissions, substitutions and extended repetition. These smoke examples do not establish a population error rate.

### Figure: input lengths and inference time

![Phase 4 inputs and timing](assets/phase4/01_inputs_and_timing.png)

**Meaning:** clips from 0.75–28.82 s load successfully. Similar long outputs create similar decoding times; the first clip includes warm-up. Loop time includes repeated first inference and exports, not model loading or save/reload.

### Figure: reference capacity and full generated lengths

![Phase 4 target and generation lengths](assets/phase4/02_token_lengths.png)

**Meaning:** references fit the model. Six predictions hit the full 448-token cap. Target-token counts exclude special tokens; generation counts include the prefix/EOS, so the two panels have different denominators.

### Figure: output growth and repetition

![Phase 4 text lengths and repetition](assets/phase4/03_output_growth_and_repetition.png)

**Meaning:** excessively long output and repeated tokens explain some visible failures. A single-token run of one does not rule out phrase loops. Text length is not CER.

### Figure: corrected stopping and Unicode flags

![Phase 4 EOS, cap and Unicode flags](assets/phase4/04_stopping_and_unicode_flags.png)

**Meaning:** six clips terminate with EOS; six reach the limit without EOS. Preserve those outputs and the one U+FFFD occurrence in baseline scoring.

### Figure: inference memory

![Phase 4 inference memory](assets/phase4/05_inference_memory.png)

**Meaning:** this inference setup fits on the measured T4. Fine-tuning memory and speed remain unmeasured.

### What this means for the baseline

**Phase 4 is complete for its engineering scope.** Poor zero-shot output is a baseline finding to measure, not a reason to delete difficult clips or remove IPA marks. All 12 sequences automatically selected an English prefix despite `language=None`; record this behavior without inventing a Nyishi language token. The broader Kaggle image still reports conflicts in unrelated installed applications.

The export labels the measured run **v6**; the prepared source is **v7**, which removes an extra restart guard while retaining the corrected inference path. The exact executed Kaggle notebook has not been supplied locally. This repository provides the source and a separate report notebook with preserved, privacy-safe measured figures.

<a href="docs/PHASE4_RESULTS_AGGREGATE.json"><ins>Aggregate metrics</ins></a> · <a href="docs/PHASE4_SMOKE_DIAGNOSTICS.json"><ins>Anonymous plot inputs</ins></a> · <a href="scripts/render_phase4_results.py"><ins>Rebuild the five public figures</ins></a>

## Phase 5A: validation-input preparation

<a href="notebooks/Phase-5A-Results-Public.ipynb"><ins>Open the Phase 5A results notebook with saved figures</ins></a> · <a href="docs/PHASE5A_RESULTS.md"><ins>Read the Phase 5A report</ins></a> · <a href="docs/PHASE5A_RESULTS_AGGREGATE.json"><ins>View aggregate results</ins></a>

### What was prepared and checked

- A private, validation-only package contains **732 frozen validation clips** (**1.881808 hours**) across **16 existing components**, with **64,137 NFD reference code points** including spaces.
- All **732 WAVs** decoded and matched the frozen waveform checks. There were **zero extraction errors**. The copied references and membership match the frozen validation manifest.
- All **737 payload-file checksums** passed; the ZIP has 738 entries including the checksum list. The archive is **138.64 MiB**. Frozen manifests and all 33 source-shard hashes remained unchanged.
- The output contains validation audio only. Stable group aliases preserve the frozen component grouping; they do not certify individual speaker identities.
- Durations range from **0.50 to 29.99 seconds** (median **7.46 seconds**); **146 clips (19.95%)** are under three seconds. The largest component contributes **24.14%** of validation hours, and **75.59%** of hours are from November 2024. Keep these skews in later error analysis.

### Figure: duration and reference length

![Phase 5A validation duration and reference length](assets/phase5a/01_validation_duration_and_reference_length.png)

### Figure: validation hours by anonymous component

![Phase 5A validation component hours](assets/phase5a/02_validation_component_hours.png)

### Figure: validation metadata coverage

![Phase 5A validation metadata coverage](assets/phase5a/03_validation_metadata_coverage.png)

**Interpretation:** Phase 5A confirms that the intended validation inputs were copied and packaged intact. It does not check whether the references are linguistically correct and contains no model predictions; **Phase 5A itself performs no inference, CER scoring or training**. The ZIP, source locator map and private provenance records are not published here.

## Phase 5B: B0 full-validation zero-shot baseline

[Corrected source notebook](notebooks/Phase-5B-Zero-Shot-Validation.ipynb) · [Saved aggregate results notebook](notebooks/Phase-5B-Results-Public.ipynb) · [Full report](docs/PHASE5B_RESULTS.md) · [Aggregate JSON](docs/PHASE5B_RESULTS_AGGREGATE.json)

### Methods and verified coverage

The pinned `neurlang/ipa-whisper-base` revision `949f7c4e4ee374b5202bc5886f6e4552e5ae2371` ran on a Tesla T4 with PyTorch 2.11.0+cu128, Transformers 4.57.1, seed 0, **FP32, batch 1, greedy decoding and a 448-token full-sequence cap**. Language/task and decoder prefix were not forced; inherited begin-suppression `[220,50257]` remained. All 732 sequences automatically selected the English/transcribe prefix; this is an observation, not evidence of Nyishi support.

Both references and hypotheses use **NFD + outer trim + repeated ordinary-space collapse**. Spaces, punctuation and every IPA mark remain. Each clip is scored separately; corpus CER sums S/D/I across clips and divides by total reference code points. It is not mean clip CER or phoneme error rate. Failed/empty hypotheses remain in the denominator; there were none. WER and alternate/raw-policy CER were not calculated.

Private saved predictions were checked against all **732 frozen validation members and references**, across **16 components**; all per-clip S/D/I were independently recalculated. The full pass reused 12 preflight predictions and generated the remaining 720. No test data was needed or scored; Nyishi fine-tuning has not run.

| Measurement | Verified B0 result |
|---|---:|
| Clips / evaluated audio | **732 / 1.881808 hours** |
| Reference code points, including spaces | **64,137** |
| Substitutions / deletions / insertions | **36,780 / 4,171 / 25,193** |
| Total edits / corpus CER | **66,144 / 103.12924%** |
| Decode errors / empty predictions | **0 / 0** |
| EOS terminated / capped without EOS | **602 (82.2%) / 130 (17.8%)** |
| Outputs containing U+FFFD | **52 (7.1%)** |
| Sum / mean per-clip elapsed time | **707.3469 s / 0.966321 s** |
| Observed real-time factor | **0.104413** |
| Full pass, 720 newly processed clips | **698.5552 s** |
| Full-pass peak PyTorch allocated / reserved | **426.21 / 460 MiB** |

Per-clip time includes processing/decoding work, excludes model loading and reporting, and differs from pass wall time (709.7663 s across both passes). Memory is measured PyTorch usage, not total GPU occupancy or a training-fit claim.

### Reporting correction

The original export's CER/edit totals were correct, but its plots, mean runtime and last-pass telemetry still described the **12-clip preflight**. The save cell now reloads saved predictions and pass telemetry and regenerates every metric and all six validation plots. A regression check reproduced a stale preflight report followed by a full result and verified that running the save cell alone repairs the export. **No inference was rerun.** The six validation plots each summarize all 732 clips; subgroup totals recover the full denominator/edit count. The review prose's earlier memory values mixed MB and MiB; the corrected figures above use byte telemetry.

### Figures and interpretation

![Phase 5B CER by duration, all 732 validation clips](assets/phase5b/01_cer_by_duration.png)

Duration scores are descriptive, without a monotonic pattern. Bins are right-closed: the first includes 147 clips at or below 3 s, whereas Phase 5A reported 146 strictly below 3 s.

![Phase 5B CER by anonymous component, all 732 validation clips](assets/phase5b/02_cer_by_component.png)

Existing components range from about 90.4% to 159.4% CER with unequal sample counts. Numbers are anonymous plot positions sorted by CER; no mapping to frozen aliases or speakers is published. These components are not expert-verified speaker identities, and small groups do not support broad conclusions.

![Phase 5B edit counts, all 732 validation clips](assets/phase5b/03_edit_counts.png)

Substitutions contribute **55.6%** of edits, insertions **38.1%**, deletions **6.3%**. CER exceeds 100% because insertions can raise edit counts above reference length. This is a weak zero-shot baseline and a fixed comparison point for supervised adaptation; it does not establish model superiority or linguistic competence.

![Phase 5B generation diagnostics, all 732 validation clips](assets/phase5b/04_generation_flags.png)

The 130 cap hits and 52 U+FFFD outputs remain in CER. Unicode flags can overlap stopping flags. Their presence motivates private error analysis; changing decoding or removing labels cannot be assumed to improve recognition.

![Phase 5B CER by stopping outcome, all 732 clips](assets/phase5b/05_cer_by_stopping.png)

The 130 cap-limited outputs have **153.8% CER**, compared with **75.4%** for the 602 EOS-terminated outputs. This is an association between observed outcomes; it does not establish that increasing the cap would improve accuracy.

![Phase 5B generated-token distribution, all 732 clips](assets/phase5b/06_generation_length_distribution.png)

Counts include the prefix/EOS and sum to 732. The last bin isolates the 130 outputs at the 448-token cap without EOS. Unequal-width bins show counts, not a probability density.

### Processor illustrations: training-only examples

The executed notebook contains four processor figures and finite FP32 `[1,80,3000]` checks for short (0.750 s) and long (28.820 s) training-smoke audio. For publication these figures were recreated from the same hash-checked private WAVs using the pinned **Transformers 4.57.1 NumPy CPU feature extractor** and anonymous labels. No model weights were loaded and no inference was run. This is a fresh processor calculation; bitwise equality to the saved Kaggle tensors is not claimed.

![Short training example, waveform and full padded features](assets/phase5b/processor_short_full.png)

![Short training example, feature close view](assets/phase5b/processor_short_zoom.png)

![Long training example, waveform and full padded features](assets/phase5b/processor_long_full.png)

![Long training example, feature close view](assets/phase5b/processor_long_zoom.png)

The 30-second window includes zero padding after real audio ends; the close views make the real-audio features readable. Colors are scaled log-Mel values, not calibrated dB. FFT windows overlap near the boundary. These are **two training illustrations**, separate from the six plots covering all 732 validation clips; they do not establish audio/reference alignment or recognition quality.

**Completion scope:** Phase 5B is complete for the declared processor/scorer checks and full B0 validation measurement. The executed notebook verifies all 737 payload checksums, the scorer checks, processor shapes/finiteness, FP32 model parameters and full validation completion. Its old preflight report is superseded by the corrected results here. The original executed notebook remains private; the public source is corrected and outputs cleared, and the public report notebook contains all ten safe figures.

## Phase 6A: supervised training smoke

The [source notebook](notebooks/Phase-6A-Kaggle-Training-Smoke.ipynb) and [Kaggle instructions](docs/PHASE6A_KAGGLE_RUN.md) check the Phase 6A training path on **eight distinct frozen-training clips**. The supplied private Kaggle export and executed notebook were reviewed; the [aggregate results](docs/PHASE6A_RESULTS.md) and [JSON](docs/PHASE6A_RESULTS_AGGREGATE.json) contain the privacy-safe findings.

| Measured gate | Result |
|---|---:|
| Training clips / optimizer updates | 8 / 2 (the second after checkpoint reload) |
| Target lengths including EOS | 6–164 tokens; EOS and padding mask checked |
| First / resumed maximum probed weight change | ~1.003e-5 / ~1.004e-5 |
| First / resumed measured update loop | 0.618 s / 0.439 s |
| First / resumed peak allocated GPU memory | 1,955 / 3,035 MiB |
| Validation CER / test | Not computed / untouched |

The FP16 AMP forward/backward, finite gradients, actual weight changes and save/reload of optimizer, scheduler, scaler and RNG state passed. Both updates reused the same eight clips, so lower second-pass loss is **not evidence of generalization**. These timings exclude setup, model load and checkpoint I/O. The saved model generation default differs from B0's explicit unforced decoding; Phase 6B must override that default for comparable validation CER. The executed notebook, training-state ZIP and weights remain private.

## Phase 6B: controlled P1 pilot prepared

The [pilot notebook](notebooks/Phase-6B-Kaggle-Pilot.ipynb), [private-bundle builder](scripts/prepare_phase6b_pilot.py) and [Kaggle run guide](docs/PHASE6B_KAGGLE_RUN.md) are prepared; **no Phase 6B training or CER result exists yet**. The deterministic nested pilot subset contains **581 full frozen-train clips / 1.002867 hours / all 15 training components**. It is checked against the private frozen manifest and hashed source WAVs. The existing B0 validation bundle supplies **all 732 clips / 1.881808 hours / 64,137 reference code points**; the test set is excluded.

The declared pilot begins independently from pinned `neurlang/ipa-whisper-base`, uses FP16 AMP training with a provisional three-epoch maximum, evaluates each epoch in FP32 under B0's exact greedy decoding/text policy, and saves resumable checkpoints. Select the minimum full-validation corpus CER, then inspect errors and measured runtime/memory before deciding on the full 10.851637-hour F1 run. No improvement or GPU runtime is claimed in advance.

## Next steps: after the pilot

1. Upload the two **private** ZIPs named in the Phase 6B guide to Kaggle and run the hash/target preflight, then the pilot. Keep predictions, transcripts, IDs, audio, manifests and weights outside GitHub.
2. Review the full-validation CER curve, S/D/I counts, decode failures, capped outputs and a small private sample of errors against B0's 103.13% CER. Preserve label uncertainty and difficult examples.
3. If the recipe and resource cost are stable, run F1 on all **3,629 train clips / 10.851637 h** from the original pinned pretrained checkpoint, then the matched LoRA comparison. Reserve **683 frozen test clips** for final evaluation after validation selection.

Unresolved label/identity uncertainty, month shift, prompt overlap and the lack of expert Nyishi adjudication remain limitations.

## Historical Narration findings

### Why the original splits were replaced

- **57/60 public speaker IDs** appear in multiple original row splits, and **49/60** appear in all three.
- Earlier grouping proposed 55 groups from 60 IDs, but name-based merges remain unverified.
- Historical identity states: 5,712 `ok`, 622 `linked_to_pair`, 208 `multi_person`, 176 `uncertain_identity`.

### Audio and transcript caveats

- Source audio is 16 kHz mono PCM16. Three zero-frame clips and one duration-discrepancy clip remain preserved with exclusion/review reasons.
- **708 clips exceed 30 s**, including 528 labelled clips. They remain in the master for possible aligned segmentation; do not truncate audio and retain a full reference.
- Raw labels contain 107 code points in the wider labelled inventory. The Phase 2 candidate subset contains 102; these are different scopes.
- Repeated ordinary spaces occur in 640 raw references. Historical A_v1/A_v2 labels differ in 51 strings; fewer flags do not prove correct linguistic conventions.
- The signal audit found no decode failures, three empty clips and no nonempty all-zero clips. **104 recordings below −50 dBFS RMS** are level-review candidates, not automatic exclusions.

### Preserved analysis evidence

The local recovered notebooks preserve original analyses and outputs for thesis reporting. Public historical copies retain saved non-audio figures/tables; embedded audio remains local. Some historical public outputs contain small displayed transcript tables, but bulk datasets and private review evidence are not distributed. This Phase 5B release adds no transcript-bearing results.

## Repository guide

| Location | Purpose |
|---|---|
| `notebooks/Phase-0.ipynb` through `Phase-3.5B.ipynb` | Historical staged analyses and saved results |
| `notebooks/Phase-4B-Kaggle-Smoke-Test.ipynb` | Current Kaggle engineering code |
| `notebooks/Phase-4B-Results-Public.ipynb`, `notebooks/Phase-5A-Results-Public.ipynb` | Public report notebooks with saved figures and privacy-safe aggregate results |
| `docs/PHASE4_RESULTS.md`, `docs/PHASE5A_RESULTS.md` | Phase reports, provenance, limitations and next actions |
| `docs/PHASE4_RESULTS_AGGREGATE.json` | Machine-readable engineering findings |
| `docs/PHASE5A_RESULTS_AGGREGATE.json` | Machine-readable bundle verification and coverage results |
| `notebooks/Phase-5B-Zero-Shot-Validation.ipynb` | Corrected inference/reporting source; private inputs required, outputs cleared |
| `notebooks/Phase-5B-Results-Public.ipynb` | Saved aggregate summary and six full-validation plots and four processor illustrations |
| `notebooks/Phase-6A-Kaggle-Training-Smoke.ipynb`, `docs/PHASE6A_RESULTS.md` | Training-smoke source and verified aggregate review; no private checkpoint |
| `notebooks/Phase-6B-Kaggle-Pilot.ipynb`, `docs/PHASE6B_KAGGLE_RUN.md` | Prepared P1 source and private Kaggle upload/run instructions |
| `docs/PHASE5B_RESULTS.md`, `docs/PHASE5B_RESULTS_AGGREGATE.json` | Verified B0 method, results, limitations and public numeric plot inputs |
| `assets/phase5b/`, `scripts/render_phase5b_results.py` | Full-validation plots and renderer using public aggregates only |
| `scripts/render_phase5b_processor.py`, `docs/PHASE5B_PROCESSOR_ILLUSTRATIONS.json` | Recreate anonymous processor figures from private training smoke inputs; no model inference |
| `scripts/audit_phase5b_saved_results.py` | Recompute metrics from authorized private saved predictions; no inference |
| `assets/phase4/`, `assets/phase5a/` | Public plots; no audio is embedded |
| `scripts/render_phase4_results.py` | Rebuild plots from public numeric diagnostics |
| `scripts/phase3_5b_proposal.py` | Historical deterministic split search; private inputs required |
| `src/nyishi_asr/`, `run_preprocessing.py` | Historical reconstruction/integrity code |
| `tests/` | Historical checks, some requiring private fixtures |

### Rebuild Phase 4 and Phase 5B figures

```bash
python scripts/render_phase4_results.py
python scripts/render_phase5b_results.py
```

Requires NumPy and Matplotlib. Reads only public JSON metrics and anonymous numeric diagnostics; performs no inference.

### Running historical workflows

Historical reconstruction code uses Python 3.11 and the root `requirements.txt`. That file describes the reconstruction environment, **not the Kaggle model stack**. Older workflows expect private inputs in their former layout; a raw download alone does not recreate every fixture. Saved outputs are historical evidence, not a claim that every old workflow runs unchanged today. Dataset-dependent tests have not been rerun in this publication.

## Data access and limitations

Audio, bulk transcripts, private manifests, speaker/reviewer metadata, workbooks, local inventories, raw generation tokens, result ZIPs and weights remain outside this release. The dataset is <a href="https://huggingface.co/datasets/Eries1506/nyishi-speed-tb-phase1"><ins>Eries1506/nyishi-speed-tb-phase1</ins></a>; observe its license and attribution terms. This repository grants no dataset rights.

**Scientific limitations:** unresolved label/identity uncertainty, no expert Nyishi adjudication, recording-month distribution shift, prompt overlap on distinct audio, and poor zero-shot validation accuracy; final test performance is unmeasured. Engineering checks do not establish tone recognition, linguistic correctness or model superiority.
