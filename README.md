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
| Zero-shot validation CER | **Next — not run** |
| Nyishi fine-tuning and final test evaluation | **Not run** |

**Latest finding:** the pipeline works on a Tesla T4, but current predictions contain omissions and repetition. In the 12 training-only smoke examples, **6 ended with EOS and 6 reached the 448-token cap without EOS**. This is a measured engineering diagnostic, not a held-out accuracy result.

### Navigate

[Phase 0](#phase-0-inventory-and-local-setup) · [Phase 1](#phase-1-eligibility-and-audio-checks) · [Phase 2](#phase-2-transcript-formatting) · [Phase 2.5](#phase-25-targeted-review) · [Phase 3](#phase-3-first-group-split-proposal) · [Phase 3.5](#phase-35-category-aware-revision) · [Phase 3.5B](#phase-35b-accepted-and-frozen-split) · [Phase 4](#phase-4-processor-and-model-smoke-test) · [Next steps](#next-steps-zero-shot-validation-baseline)

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

## Next steps: zero-shot validation baseline

1. **Prepare inputs:** hash-check the frozen validation manifest and package its **732 clips / 1.881808 hours** privately for Kaggle. Verify audio/reference alignment.
2. **Verify scoring:** check CER on known pairs, combining marks and empty predictions. Preserve spaces, marks and every example in the denominator.
3. **Run a small validation preflight**, then all validation clips with resumable per-sample exports and the same pinned settings.
4. **Report corpus CER:** S/D/I, reference-character denominator, clips/hours, decode failures, EOS/cap flags, runtime and memory. Repetition and U+FFFD count as output errors; CER can exceed 100%.
5. **Begin supervised adaptation:** use a 1–2 hour training pilot; select checkpoints by validation CER and measure training memory before expanding.
6. **Final evaluation:** reserve the 683 test clips until the selected recipe is ready. Any decoding-policy alternative is a documented validation experiment.

No further general cleaning or split search is required to begin the baseline. Save the executed Kaggle notebook locally for provenance when available.

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

The local recovered notebooks preserve original analyses and outputs for thesis reporting. Public historical copies retain saved non-audio figures/tables; embedded audio remains local. Some historical public outputs contain small displayed transcript tables, but bulk datasets and private review evidence are not distributed. This Phase 4 release adds no transcript-bearing results.

## Repository guide

| Location | Purpose |
|---|---|
| `notebooks/Phase-0.ipynb` through `Phase-3.5B.ipynb` | Historical staged analyses and saved results |
| `notebooks/Phase-4B-Kaggle-Smoke-Test.ipynb` | Current Kaggle engineering code |
| `notebooks/Phase-4B-Results-Public.ipynb` | Public report with saved figures derived from the reviewed export |
| `docs/PHASE4_RESULTS.md` | Full results, provenance, limitations and next actions |
| `docs/PHASE4_RESULTS_AGGREGATE.json` | Machine-readable engineering findings |
| `assets/phase4/` | Five current public plots, with anonymous example positions |
| `scripts/render_phase4_results.py` | Rebuild plots from public numeric diagnostics |
| `scripts/phase3_5b_proposal.py` | Historical deterministic split search; private inputs required |
| `src/nyishi_asr/`, `run_preprocessing.py` | Historical reconstruction/integrity code |
| `tests/` | Historical checks, some requiring private fixtures |

### Rebuild Phase 4 figures

```bash
python scripts/render_phase4_results.py
```

Requires NumPy and Matplotlib. Reads only public JSON metrics and anonymous numeric diagnostics; performs no inference.

### Running historical workflows

Historical reconstruction code uses Python 3.11 and the root `requirements.txt`. That file describes the reconstruction environment, **not the Kaggle model stack**. Older workflows expect private inputs in their former layout; a raw download alone does not recreate every fixture. Saved outputs are historical evidence, not a claim that every old workflow runs unchanged today. Dataset-dependent tests have not been rerun in this publication.

## Data access and limitations

Audio, bulk transcripts, private manifests, speaker/reviewer metadata, workbooks, local inventories, raw generation tokens, result ZIPs and weights remain outside this release. The dataset is <a href="https://huggingface.co/datasets/Eries1506/nyishi-speed-tb-phase1"><ins>Eries1506/nyishi-speed-tb-phase1</ins></a>; observe its license and attribution terms. This repository grants no dataset rights.

**Scientific limitations:** unresolved label/identity uncertainty, no expert Nyishi adjudication, recording-month distribution shift, prompt overlap on distinct audio, and no held-out model accuracy result yet. Engineering checks do not establish tone recognition, linguistic correctness or model superiority.
