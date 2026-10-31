# Nyishi Narration ASR research

Source code and output-free analysis notebooks for a Nyishi IPA speech-recognition research project. Current scope is Dataset A Narration; YouTube is deferred.

## Status

This initial snapshot preserves historical data exploration, staged metadata/IPA-policy reconstruction, and experimental curation code. No final speaker-disjoint split, trained baseline or Nyishi accuracy result is supplied. The newer working decision is minimal formatting normalization; the historical A_v1/A_v2 phonetic substitutions remain provisional research alternatives.

## Contents

- `notebooks/`: three adapted analysis notebooks without saved outputs, execution counters or embedded media. Original notebooks with results remain private in the local workspace for thesis reporting.
- `src/nyishi_asr/` and `run_preprocessing.py`: historical staged reconstruction and integrity checks.
- `scripts/prepare_narration_baseline.py`: earlier candidate-pool, duplicate-waveform and grouping analysis; its IPA-policy assumptions are not the selected minimal-normalization baseline.
- `tests/`: normalization, integrity, source-recovery and dataset-dependent checks.
- `requirements.txt`: exact package versions from the existing reconstruction environment, not a model-training environment.

## Data is not included

Audio, transcripts, manifests, speaker/reviewer metadata, review workbooks, local inventories, recovery archives and model weights are intentionally excluded. The source dataset is [Eries1506/nyishi-speed-tb-phase1](https://huggingface.co/datasets/Eries1506/nyishi-speed-tb-phase1); observe its license and attribution requirements. This repository does not grant a license to the dataset.

## Running the historical code

Use Python 3.11 and the listed dependencies. This snapshot is not a standalone training package: historical recipes expect privately supplied files in their former layout, including `raw/A_narration`, `manifests`, `cleaning`, `metadata`, reconstruction evidence, and some original recovery inputs. A fresh raw download alone does not recreate those inputs. Source-recovery/dataset tests require those fixtures and should not be presented as passing without them. The local directory cleanup changed their locations; the historical code is retained here as a research record, not an already-integrated pipeline for that layout.

The normalization and integrity unit tests can run without dataset fixtures. Notebooks also contain staged historical assumptions and must be inspected before execution. Some notebook source cells include short linguistic examples; bulk transcripts and saved data outputs are absent.

## Next experiment

1. Implement a new minimal-cleaning workflow from original reference text, preserving IPA marks and pronunciation differences.
2. Exclude missing-label/empty-audio examples in a derived view; retain source data.
3. Freeze speaker/session-aware splits and one evaluation policy.
4. Validate IPA tokenization and run an IPA-Whisper Base pilot before substantive comparisons.

Do not infer linguistic correctness, tone recognition or model superiority from code checks or candidate membership.
