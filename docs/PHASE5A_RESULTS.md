# Phase 5A — validation input preparation

**Status: completed and independently verified · 1 November 2026**

Phase 5A assembled the frozen validation partition into a private Kaggle-ready bundle. The checks answer whether the correct validation clips and references were copied intact. They do **not** assess transcription quality.

## Verified results

| Check | Result |
|---|---:|
| Frozen validation clips | 732 |
| Audio duration | 1.881808 hours |
| Existing validation components | 16 |
| Reference denominator under the current NFD policy | 64,137 code points, including spaces |
| Audio decode / frame / waveform checks | 732 passed; 0 failures |
| Payload checksums | 737 passed (738 ZIP entries counting the checksum list) |
| Source Parquet shards rechecked | 33 unchanged |
| Frozen input manifests | unchanged |
| Private ZIP size | 145,378,360 bytes (138.64 MiB) |
| Private ZIP SHA-256 | `2a7cb19aac3707488733b9af8bd70b694f969583e1fbe03de2a9f39466954f52` |
| Inference / CER / training | not run |

The frozen IDs, raw and approved normalized references, and waveform identities in the ZIP were independently compared with the frozen validation manifest. Audio was extracted from Parquet without resampling or other audio transformations. The original frozen manifests and all 33 source shards were verified unchanged.

## Coverage observations

- Clip durations range from **0.50 to 29.99 seconds**; median **7.46 seconds**. **146 clips (19.95%)** are shorter than 3 seconds. These are valid members of the frozen validation set and were retained.
- The largest existing component accounts for **24.14%** of validation audio hours. The component labels in the bundle are stable aliases for grouping; they do not imply expert-verified individual speaker identities.
- By audio hours, recording-month coverage is **75.59% November 2024**, **23.68% April 2025**, and **0.73% June 2025**. Gender metadata is **50.88% female**, **46.33% male**, and **2.80% unknown**. These skews should be considered during later error analysis; they are not reasons to alter the frozen split now.

See the plots in [`assets/phase5a/`](../assets/phase5a/) and the saved-figure notebook [`Phase-5A-Results-Public.ipynb`](../notebooks/Phase-5A-Results-Public.ipynb).

## What the Parquet and grouping notes mean

The audio is stored inside large Parquet shard files. A shard is one container file; a row group is a block of rows inside that container. By collecting requested clips by shard and row group, the preparation notebook opens each relevant block efficiently instead of repeatedly scanning the same large file for every clip.

Only the 732 selected validation WAVs were written. The local locator/provenance map records where each clip came from in the source Parquet files, which is useful for audit and recovery. It was deliberately kept private and is not part of the Kaggle ZIP or this GitHub release.

The bundle replaces source component labels with stable aliases such as validation groups. This preserves the already-frozen grouping without exposing the original labels. A group is a connected component used to keep related data together; it is not proof that all its clips belong to one verified speaker.

The storage estimate was for planning disk capacity and included extracted WAVs, the compressed ZIP, and working buffer. The completed ZIP size above is the measured result. The estimate was not a request to create another buffer file.

## Gate and next step

**Gate: passed.** No code, data, transcript-normalization, or split changes are required based on Phase 5A. The ZIP is ready to upload as a private Kaggle dataset. The next work is Phase 5B: validate the CER implementation, illustrate processor features using the separate training-only smoke examples, run a small validation preflight, then run the full zero-shot validation baseline. Keep the frozen test data out of Kaggle until the final evaluation is predeclared.

The ZIP, WAVs, validation manifest, source locator map, source-shard hash records, and private provenance remain excluded from GitHub.
