# Phase 5B — B0 full-validation zero-shot baseline

**B0 status: complete and independently verified · 1 November 2026. Reporting corrected from saved predictions; no model inference rerun.**

## Method and scope

`neurlang/ipa-whisper-base` revision `949f7c4e4ee374b5202bc5886f6e4552e5ae2371`, seed 0, Tesla T4 (`cuda:0`), PyTorch 2.11.0+cu128, Transformers 4.57.1. The source pins Tokenizers 0.22.2, Hub 0.36.2 and Safetensors 0.8.0; the exported run configuration does not independently record those three installed versions. FP32, batch 1, greedy decoding (`num_beams=1`, `do_sample=False`), full sequence cap 448 tokens, no forced language/task/decoder prefix, no timestamps. `suppress_tokens=[]`; inherited `begin_suppress_tokens=[220,50257]` was retained. All 732 outputs selected the English/transcribe prefix automatically. This is observed model behavior, not a Nyishi language-token policy.

References and hypotheses use Unicode NFD, outer trim and collapse of repeated ordinary U+0020 spaces. Spaces, punctuation and IPA marks remain. Each clip is scored separately; corpus CER is summed S+D+I divided by summed reference code points. It is neither mean clip CER nor a grapheme/phoneme error rate. Empty or failed predictions remain in the reference denominator; none occurred here. WER and alternate/raw-policy CER were not computed.

The full run reused the 12 preflight predictions and processed the remaining 720 clips. No training or test inference is evidenced by this export; the inputs and results checked here contain validation only. Test remains reserved for final evaluation. No new master-data inventory or audio extraction was performed during this reporting review.

## Verified results

| Measurement | Full-validation result |
|---|---:|
| Clips / existing frozen components | **732 / 16** |
| Evaluated audio | **1.881808 hours** |
| Reference code points, including spaces | **64,137** |
| Substitutions / deletions / insertions | **36,780 / 4,171 / 25,193** |
| Total edits / corpus CER | **66,144 / 103.12924%** |
| Decode errors / empty outputs | **0 / 0** |
| EOS terminated | **602 (82.2%)** |
| 448-token cap without EOS | **130 (17.8%)** |
| Outputs containing U+FFFD | **52 (7.1%)** |
| Automatically selected English prefix | **732 / 732** |
| Sum / mean per-clip elapsed time | **707.3469 s / 0.966321 s** |
| Observed real-time factor | **0.104413** |
| Two inference-pass wall times combined | **709.7663 s** |
| Full pass wall time, 720 newly processed clips | **698.5552 s** |
| Full-pass peak PyTorch allocated / reserved | **426.21 / 460 MiB** |

RTF uses summed per-clip elapsed time divided by audio seconds. These measurements include feature extraction/decoding work timed per clip; they exclude installation, model loading and reporting and are not a steady-state speed benchmark. Pass wall time has a different scope. PyTorch peaks are not total GPU occupancy or training-memory requirements. The private review prose's 446.9/482.3 values confused decimal MB with MiB; the byte telemetry supports the MiB figures above.

## Reporting correction and checks

The original ZIP's aggregate CER and S/D/I were correct, but its three plots still showed the 12-clip preflight. Its mean runtime and last-pass telemetry were also stale. The save cell refreshed only some metrics while reusing earlier reporting globals and plots.

The corrected notebook uses one reporting function. Both its report and save cells reload predictions and pass telemetry, verify references/grouping/durations against the validation bundle, and regenerate every validation plot and summary before export. A local regression check first generated a 12-row report, replaced the saved input with the full run, then executed only the save cell: all 732 rows, four plots, mean runtime and full-pass memory/timing updated correctly. Model imports/generation were not executed for this check.

Independent audit of the supplied results checked:

- Exactly 732 unique prediction IDs equal the private bundle and frozen validation membership; references, raw references, durations and waveform identities match the frozen records.
- Bundle-manifest checksum matches the saved run configuration; every record and pass has the matching run fingerprint. These private identifiers/checksums remain outside the public aggregate.
- CSV and JSONL prediction/reference/S/D/I fields agree. All 732 per-clip S/D/I decompositions were recalculated from saved text, including scorer checks for combining marks, spaces and empty hypotheses.
- Duration and anonymous-component aggregates each sum to 732 clips, 64,137 reference code points and 66,144 edits. The four published plots were regenerated and visually inspected.
- Source notebook code cells parse and carry no saved execution outputs. The report notebook contains only aggregate text/JSON and these four plots.

## Full-set plots and interpretation

![B0 CER by duration, all 732 clips](../assets/phase5b/01_cer_by_duration.png)

Duration bins are right-closed: `(0,3]`, `(3,5]`, `(5,10]`, `(10,15]`, `(15,20]`, `(20,30]` seconds. The first bin contains 147 clips because it includes a clip exactly 3 seconds long; Phase 5A's strictly-under-3-second count was 146. Bin CER is descriptive, without a consistent monotonic duration trend.

![B0 CER by anonymous component, all 732 clips](../assets/phase5b/02_cer_by_component.png)

Component numbers are plot positions sorted by CER, with no published mapping to frozen aliases, speakers or source provenance. These are existing evaluation components, not expert-verified speakers. Scores range from approximately 90.4% to 159.4%; group sizes differ substantially, including one with only four clips. No causal or statistical-significance claim follows from this plot.

![B0 substitutions, deletions and insertions, all 732 clips](../assets/phase5b/03_edit_counts.png)

Substitutions account for 55.6% of edits, insertions 38.1%, deletions 6.3%. CER can exceed 100% because insertions add errors beyond the reference length; 103.13% does not mean that percentage of clips or words is wrong. This is a poor zero-shot result under the declared IPA scoring convention and a usable fixed reference for supervised adaptation. It establishes neither Nyishi competence nor model superiority.

![B0 stopping and Unicode flags, all 732 clips](../assets/phase5b/04_generation_flags.png)

Cap hits and U+FFFD are retained in the score. U+FFFD overlaps stopping categories. These diagnostics motivate private error review; they do not prove that changing the cap, language prefix or normalization would improve recognition.

## Evidence limits and next steps

**The full-validation B0 measurement and its corrected aggregate publication are complete.** The local source notebook has no saved execution outputs, and the supplied ZIP omits the separate training-only processor illustrations saved outside the result folder. Their execution/visual review and the entire notebook checklist cannot be certified from this export. Retain the executed Kaggle notebook and those illustrations privately when available; this does not require rerunning B0 inference.

Recommended next work, following the existing sprint plan:

1. Preserve B0, frozen splits and scoring as the comparison reference. Privately inspect high/low-CER, capped and U+FFFD outputs against audio and references. Record uncertainty; do not alter labels or exclude clips merely to improve CER.
2. Specify supervised decoder-prefix, target shifting and padding-mask policy; do not invent a Nyishi token. Any changed decoding setup is a separate declared validation experiment.
3. Complete Phase 6A: finite FP16-AMP loss/gradients, a verified optimizer update and checkpoint/resume with optimizer/scaler state. Then run the component-aware 1–2 hour Phase 6B pilot and select by validation CER.
4. Measure training resources before full training or LoRA comparisons. Keep the 683 test clips untouched until final selection.

Unresolved label/identity uncertainty, no expert Nyishi adjudication, month-distribution shift and prompt overlap on distinct audio remain limitations. Supervised improvement, linguistic correctness and tone recognition remain unmeasured.

## Public artifacts and reproduction

- [Corrected source notebook](../notebooks/Phase-5B-Zero-Shot-Validation.ipynb): private frozen-hash expectations are supplied through environment variables; outputs are cleared.
- [Aggregate report notebook](../notebooks/Phase-5B-Results-Public.ipynb): saved figures and summary, with no prediction-level tables.
- [Allowlisted aggregate JSON](PHASE5B_RESULTS_AGGREGATE.json): numeric group totals and declared method only.
- [Audit code](../scripts/audit_phase5b_saved_results.py): recomputes from authorized private ZIP, bundle manifest and frozen manifest; see `--help`. Use a private output location until reviewed.
- [Plot renderer](../scripts/render_phase5b_results.py): `python scripts/render_phase5b_results.py` uses public JSON only (Matplotlib required).

The original ZIP, corrected private ZIP, prediction tables, transcripts, audio, manifests, sample IDs, source/speaker mappings and private provenance are excluded. Original private evidence is preserved unchanged. This report is derived from exported results; it is not an executed copy of the inference notebook.
