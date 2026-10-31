# Nyishi Narration ASR research

Source code and analysis notebooks with saved plots and tables for a Nyishi IPA speech-recognition research project. Current scope is Dataset A Narration; YouTube is deferred.

## Status

This initial snapshot preserves historical data exploration, staged metadata/IPA-policy reconstruction, and experimental curation code. No final speaker-disjoint split, trained baseline or Nyishi accuracy result is supplied. The newer working decision is minimal formatting normalization; the historical A_v1/A_v2 phonetic substitutions remain provisional research alternatives.

## Phase 0: local inventory

The [Phase 0 notebook](notebooks/Phase-0.ipynb) records the 31 October 2026 local environment and data inventory, with its saved cell outputs. All required local inputs were found. The 59 Narration Parquet shards and the starting `master_A_v1_1.parquet` each contain 6,718 rows; the master has 54 columns. The local Python 3.11.1 environment had NumPy 2.3.5, pandas 2.2.3, PyArrow 26.0.0 and SoundFile 0.13.1. PyTorch and the model-training packages were not installed there. Kaggle GPU type, memory and quota have not been recorded.

The first-pass manifest counts are 5,651 rows flagged with transcripts, 1,067 flagged missing, 6,714 marked `audio_ok`, three with zero decoded duration, and 708 over 30 seconds. Combining labelled, audio-OK and 0.5–30-second conditions gives 5,086 candidate rows. These conditions have not yet undergone a final QC waterfall or speaker/session split; the candidate count is not a training-set size.

![Narration inventory counts](assets/narration_inventory.svg)

## Phase 1: eligibility and audio checks

The [Phase 1 notebook](notebooks/Phase-1.ipynb) now includes its saved output tables and plots. Its 6,718 master rows joined one-to-one with the full-decode facts. There were no full-decode failures, sample-rate disagreements, channel disagreements or disagreements between decoded-duration fields. The filtering waterfall is:

| First applicable outcome | Clips |
|---|---:|
| Initial 0.5–30-second candidates | 5,086 |
| Missing or empty reference | 1,067 |
| Outside initial duration range | 561 |
| Zero audio frames | 3 |
| Duration discrepancy held for review | 1 |

The candidates contain **14.826 decoded hours** before any speaker/session split. The 561 duration exclusions comprise 528 labelled clips above 30 seconds and 33 labelled, nonempty clips below 0.5 seconds. Another three labelled clips below 0.5 seconds have zero frames and are counted under that earlier exclusion. One duration-mismatch clip, `A_004972`, is held for review. Of the candidates, 967 are provisionally train-only because of pair linkage, multiple speakers or uncertain identity. The source master and decode-facts hashes did not change during the Phase 1 run.

![Phase 1 eligibility reasons](assets/phase1_eligibility_reasons.png)

![Phase 1 duration distribution](assets/phase1_duration_distribution.png)

![Phase 1 speaker hours](assets/phase1_speaker_hours.png)

![Phase 1 speaker status and train-only flags](assets/phase1_speaker_status.png)

The eligibility, speaker-hours and speaker-status images come from saved Phase 1 notebook outputs. The duration image was generated separately from the same saved Phase 1 audit because its notebook plot cell had not been executed. No train/validation/test split or model baseline is implied by these figures.

## Phase 2: transcript formatting policy

The [Phase 2 notebook](notebooks/Phase-2.ipynb) includes its saved tables and plots. It reads the 5,086 first-pass candidates and retains every `transcript_raw`. Its derived `transcript_phase2` applies Unicode NFD, trims outer whitespace and collapses repeated ordinary spaces. The run changed **497 rows (9.77%)**, entirely because of repeated spaces; NFD and outer trimming changed zero rows. The character total fell from 522,368 to 521,747, a difference of 621 spaces. Both versions contain 102 distinct code points in this candidate subset. The source hash was unchanged and the policy was idempotent.

![Phase 2 formatting changes and transcript lengths](assets/phase2_normalization_and_length.png)

![Phase 2 audio duration versus label length](assets/phase2_duration_vs_label_length.png)

The plots describe label formatting and length, not transcription quality or model accuracy. The Unicode inventory includes dental combining marks, nasalization, length marks and rare combining accents. Their frequency does not determine whether they are linguistically meaningful. The first baseline preserves these marks and does not automatically convert `g`/`ɡ` or punctuation-like symbols.

The targeted Phase 2.5 review queue is the **41 distinct candidate clips** containing `:`, `;` or `[`, approximately **0.195 decoded hours**. Their meanings are unresolved; keep them out of primary held-out references until reviewed. A single clip contains two combining acute accents and should also be reviewed without deleting the marks. Phase 2.5 also checks proposed speaker merges, pair-linked and multi-person train-only groups, one recording session spanning two proposed groups, and exact-audio overlap before Phase 3 freezes splits. The pinned IPA-Whisper processor's tokenizer round-trip and target-length checks remain a model-compatibility gate; no processor or weights were downloaded in Phase 2.

## Findings from the recovered Narration analyses

The local recovered notebooks retain complete originals for thesis reporting. The three published analysis notebooks now include their saved non-audio plots, tables and textual results. Embedded WAV playback outputs are omitted from the public copies; the originals remain in the local BTP directory. The aggregate findings are summarized here; bulk datasets and review workbooks are not published.

- The original row-level splits overlap heavily in speakers: 57 of 60 public speaker IDs appear in multiple splits and 49 in all three. A new speaker/session-aware split is needed before a held-out evaluation.
- Earlier proposed grouping reduces 60 public speaker IDs to 55 groups, but name-based merges are unverified. The current status distribution is 5,712 `ok`, 622 `linked_to_pair`, 208 `multi_person`, and 176 `uncertain_identity`.
- Narration audio is 16 kHz, mono PCM16. Three IDs have zero frames: `A_004973`, `A_006614`, `A_006615`. `A_004972` is 2.531 seconds decoded versus 3.89 seconds stated and needs boundary review. There are 708 clips above 30 seconds; 528 have transcript references. Long clips stay in the master for possible later aligned segmentation.
- Raw references contain 107 distinct code points including space and are already in Unicode NFD. Repeated ordinary spaces occur in 640 references. Plain `g` and IPA `ɡ` both occur frequently. The old A_v1 and A_v2 conventions differ in 51 transcript strings; fewer unresolved flags under A_v2 do not validate its linguistic choices.
- The audio signal audit found no decoding failures, three empty recordings and no nonempty all-zero recordings. It flagged 104 clips below −50 dBFS RMS. Those level measurements are review signals, not evidence that the speech is unusable.

The next research step uses `transcript_raw`, `transcript_missing`, `audio_ok`, `audio_issue`, `duration_decoded`, and stable `sample_id`/`audio_path` fields for a derived eligibility manifest. Speaker and session fields are needed for splits. In the starting master, `transcript_normalized`, `duplicate_group_id`, and `final_split` are unfilled; no final target policy or split can be inferred from them.

## Contents

- `notebooks/`: Phase 0, Phase 1 and Phase 2 notebooks with saved results, plus three adapted historical analysis notebooks with saved plots/tables. Complete recovered originals, including embedded audio outputs, remain local for thesis reporting.
- `src/nyishi_asr/` and `run_preprocessing.py`: historical staged reconstruction and integrity checks.
- `scripts/prepare_narration_baseline.py`: earlier candidate-pool, duplicate-waveform and grouping analysis; its IPA-policy assumptions are not the selected minimal-normalization baseline.
- `tests/`: normalization, integrity, source-recovery and dataset-dependent checks.
- `requirements.txt`: exact package versions from the existing reconstruction environment, not a model-training environment.

## Data is not included

Audio, transcripts, manifests, speaker/reviewer metadata, review workbooks, local inventories, recovery archives and model weights are intentionally excluded. The source dataset is [Eries1506/nyishi-speed-tb-phase1](https://huggingface.co/datasets/Eries1506/nyishi-speed-tb-phase1); observe its license and attribution requirements. This repository does not grant a license to the dataset.

## Running the historical code

Use Python 3.11 and the listed dependencies. This snapshot is not a standalone training package: historical recipes expect privately supplied files in their former layout, including `raw/A_narration`, `manifests`, `cleaning`, `metadata`, reconstruction evidence, and some original recovery inputs. A fresh raw download alone does not recreate those inputs. Source-recovery/dataset tests require those fixtures and should not be presented as passing without them. The local directory cleanup changed their locations; the historical code is retained here as a research record, not an already-integrated pipeline for that layout.

The normalization and integrity unit tests can run without dataset fixtures. Notebooks also contain staged historical assumptions and must be inspected before execution. Their saved outputs are historical evidence; they do not make the older code directly runnable in the current streamlined layout. Some notebook outputs include small displayed transcript tables, while bulk datasets, audio payloads and review files remain outside this repository.

## Next experiment

1. Implement a new minimal-cleaning workflow from original reference text, preserving IPA marks and pronunciation differences.
2. Exclude missing-label/empty-audio examples in a derived view; retain source data.
3. Freeze speaker/session-aware splits and one evaluation policy.
4. Validate IPA tokenization and run an IPA-Whisper Base pilot before substantive comparisons.

Do not infer linguistic correctness, tone recognition or model superiority from code checks or candidate membership.
