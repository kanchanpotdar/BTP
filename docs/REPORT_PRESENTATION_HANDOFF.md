# Nyishi IPA-ASR — report and presentation handoff

**Status:** completed first Narration baseline, including the one declared final-test comparison and post-hoc descriptive analysis. **Prepared:** 10 November 2026. This document is for a collaborator who will write the BTP report and slides from saved evidence **without running training or inference**. It separates measurements from interpretation and future proposals. The [README](../README.md) is the visual overview; the phase reports and executed notebooks linked below are the underlying evidence.

## 1. The result to communicate first

The study asked whether supervised adaptation of an IPA-oriented Whisper Base model improves Nyishi Narration transcription under a frozen, group-aware train/validation/test design. It did:

| Declared configuration | Partition | Clips | Reference code points | Corpus CER | Substitutions | Deletions | Insertions | Cap hits without EOS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0, pinned pretrained model, zero-shot | Test | 683 | 65,577 | **101.23214%** | 36,940 | 4,395 | 25,050 | 133 |
| Validation-selected full fine-tuning, LR `2e-5`, Epoch 5, repetition penalty `1.08` | Test | 683 | 65,577 | **44.57355%** | 13,525 | 6,158 | 9,547 | 0 |

The improvement is **56.66 absolute CER percentage points** on the same held-out clips. Substitutions and insertions decreased; deletions increased. The selected model ended all 683 generations with EOS and produced no U+FFFD outputs, but only **two** clips were exact full-transcript matches. A 44.57% CER is a meaningful improvement over this zero-shot baseline, yet remains a high transcription error rate. Call it a **reproducible first research baseline**, not a production-ready transcriber. The test result is in [Phase 8](PHASE8_RESULTS.md), its [aggregate JSON](PHASE8_RESULTS_AGGREGATE.json), the [executed Kaggle notebook](../notebooks/executed/phase8/final_sealed_test.ipynb), and the [dashboard](../assets/phase8/test_aggregate_dashboard.png). Do not present the test score as a validation-selected hyperparameter or tune anything from it.

## 2. What the model and target are

- **Task:** speech-to-text where the target is the project's IPA-like Nyishi transcription, not English translation. The dataset does not provide independently adjudicated phonetic ground truth.
- **Initialization:** [`neurlang/ipa-whisper-base`](https://huggingface.co/neurlang/ipa-whisper-base), pinned revision `949f7c4e4ee374b5202bc5886f6e4552e5ae2371`. It is a Whisper Base encoder-decoder Transformer previously adapted toward IPA outputs. Its model card reports 15,000 synthetic IPA-labelled Common Voice recordings from more than 70 languages. This is a useful initialization, **not evidence of prior Nyishi competence**.
- **Processor:** Whisper feature extraction converts 16 kHz mono audio into log-Mel inputs with a fixed 30-second feature window; the tokenizer handles IPA text. Round-trip checks on the actual labels verified that the target symbols and lengths fit the decoder budget. It does not align long audio to full transcripts automatically.
- **Hardware:** measured main runs used an NVIDIA T4. Training used FP16 automatic mixed precision; validation/test generation used FP32. The model and architecture were not changed during this first comparison.

## 3. Data lineage and frozen evaluation design

The source-of-truth Narration master contained **6,718 rows and 54 columns**; 59 source shards matched it row-for-row. There were **5,651** rows with usable-looking references, about **20.88** decoded labelled hours. Three recordings had zero frames. A first-pass reference/audio/duration waterfall led to **5,086 structurally eligible 0.5–30-second candidates**, about **14.83 hours**, before group-aware splitting. Long clips were preserved in the source rather than truncated with mismatched full transcripts. This is [Phase 0–1](../README.md#phase-0--inventory-and-local-setup) evidence, not 5,086 training clips.

Phase 2 applied only Unicode **NFD**, outer whitespace trimming and collapse of repeated ordinary spaces; 497 candidate rows changed because of repeated-space cleanup. IPA marks, punctuation, tone/length signs and raw source labels were retained. Phase 2.5 created review queues for unusual symbols and uncertain speaker/group assignments; it did not automatically rewrite labels or prove speaker identities. The first Phase 3 and revised Phase 3.5 group-split proposals were rejected for balance problems. Phase **3.5B** accepted and froze the split after known speaker/session/component/exact-audio cross-split leakage checks. The accepted distribution is:

| Frozen partition | Clips | Audio hours | Components | NFD reference code points |
|---|---:|---:|---:|---:|
| Train | **3,629** | **10.851637** | 15 | 383,474 |
| Validation | **732** | **1.881808** | 16 | 64,137 |
| Test | **683** | **1.891057** | 13 | 65,577 |

The first benchmark covers **Dataset A Narration only**. Dataset B YouTube was excluded. Its speech/song mix, transcript alignment, overlap and rights would require separate review before a future training study; there is no measured claim here that adding it helps or hurts. See the [frozen-split figure](../assets/phase3_5b_proposal_comparison.png), [dataset funnel](../assets/phase8/eda/01_dataset_funnel_and_duration.png), [public decisions](DECISIONS_PUBLIC.md), and [sprint plan](../SPRINT_PLAN.md).

### Evaluation contract

For each reference and prediction, apply NFD, trim ends, and collapse repeated ordinary U+0020 spaces. Keep spaces and IPA combining marks. Compute the Levenshtein substitutions `S`, deletions `D`, and insertions `I` **within each clip**; sum the three counts over all clips and divide by the sum of reference Unicode code points:

`corpus CER (%) = 100 × (ΣS + ΣD + ΣI) / Σ reference code points`.

Do not average clip-level CERs or concatenate clip boundaries. CER can exceed 100% when insertion output is long. “Cap hit” means the model reached the declared generation limit without EOS; those predictions were retained and scored. WER is not a headline metric because word boundaries in these labels are not independently validated. The [protocol](PHASE4_ZERO_SHOT_PROTOCOL.md) and [Phase 8 scorer](../scripts/phase8_final_test.py) are the implementation sources.

## 4. Completed phases, in chronological order

The tables below are **measured run facts** unless a row explicitly says proposal, preparation or diagnostic. “B0”, “P1”, “F1” and “L1” are experiment labels: zero-shot baseline, pilot, full fine-tuning, and LoRA comparison. They are not separate datasets or different language tasks.

| Phase | What was done | Verified outcome and meaning | Read and reuse |
|---|---|---|---|
| 0 | Inventoried source data, raw label and metadata fields. | 6,718 Narration rows; source preserved. No model score. | [README Phase 0](../README.md#phase-0--inventory-and-local-setup), [notebook](../notebooks/Phase-0.ipynb) |
| 1 | Decoded and screened eligibility with recorded reasons. | 5,086 short labelled candidates / 14.826 h; no full-decode failures. This is a candidate pool. | [README Phase 1](../README.md#phase-1--eligibility-and-audio-checks), [eligibility plot](../assets/phase1_eligibility_reasons.png), [notebook](../notebooks/Phase-1.ipynb) |
| 2 | Fixed the conservative text view. | 497 repeated-space-only changes; zero labels became empty; IPA distinctions retained. | [README Phase 2](../README.md#phase-2--transcript-formatting), [plot](../assets/phase2_normalization_and_length.png), [notebook](../notebooks/Phase-2.ipynb) |
| 2.5 | Recorded symbol/identity and group-review queues. | 42 symbol-queue clips and 967 train-only candidates; no automatic linguistic correction. | [README Phase 2.5](../README.md#phase-25--targeted-review), [plot](../assets/phase2_5_review_workload.png), [notebook](../notebooks/Phase-2.5.ipynb) |
| 3 | Proposed group-aware split. | Known structural leakage zero, but month/short-clip balance was poor; rejected. | [plot](../assets/phase3_category_balance.png), [notebook](../notebooks/Phase-3.ipynb) |
| 3.5 | Searched a category-aware revision. | Still concentrated validation in one component; rejected. | [plot](../assets/phase3_5_category_comparison.png), [notebook](../notebooks/Phase-3.5.ipynb) |
| 3.5B | Accepted a frozen, group-aware split. | 3,629/732/683 clips; known structural cross-split overlaps zero. | [plot](../assets/phase3_5b_proposal_comparison.png), [notebook](../notebooks/Phase-3.5B.ipynb) |
| 4 | Verified processor, tokenizer, generation and checkpoint reload on training-only examples. | 5,086 tokenizer checks; 12 audio examples; six of 12 hit 448-token cap without EOS; **no accuracy score**. | [report](PHASE4_RESULTS.md), [executed notebook](../notebooks/executed/phase4/processor_model_smoke.ipynb), [plots](../assets/phase4/) |
| 5A | Built and checksum-verified the private validation input bundle. | All 732 clips / 1.881808 h and 64,137 reference code points aligned. **No inference**. | [report](PHASE5A_RESULTS.md), [plots](../assets/phase5a/) |
| 5B / B0 | Ran pinned model zero-shot on full validation. | **103.12924% CER**, S/D/I = 36,780/4,171/25,193; 130 no-EOS cap hits and 52 U+FFFD outputs. Insertion-heavy poor starting point. | [report](PHASE5B_RESULTS.md), [executed notebook](../notebooks/executed/phase5b/zero_shot_validation.ipynb), [plots](../assets/phase5b/) |
| 6A | Tested real supervised updates and checkpoint resume on eight frozen-train clips. | Two finite AMP optimizer updates; saved/reloaded optimizer, scheduler, scaler and RNG. **No CER**. | [report](PHASE6A_RESULTS.md), [executed notebook](../notebooks/executed/phase6a/training_smoke.ipynb) |
| 6B / P1 | Independently trained a 581-clip / 1.002867 h pilot for three epochs; evaluated all 732 validation clips each time. | CER **159.66 → 129.72 → 107.26%**; best Epoch 3 remained worse than B0. Loss and cap hits fell, showing learning but inadequate one-hour coverage. | [report](PHASE6B_RESULTS.md), [executed notebook](../notebooks/executed/phase6b/pilot.ipynb), [plots](../assets/phase6b/) |
| 6C | Audited generation configuration and compared individual, batch-4 and batch-8 inference on a deterministic 12-clip validation subset. | Exact token, raw/normalized text and S/D/I equality. Batch 8: **5.14 s**, **941.7 MiB** peak, **2.14×** faster than individual generation. Repetition/insertion pathology remained. | [report](PHASE6C_RESULTS.md), [aggregate](PHASE6C_EQUIVALENCE_AGGREGATE.json) |

### Full-data experiments and final selection

| Run | Training and selection | Best complete-validation result | What it established |
|---|---|---|---|
| Initial F1, LR `1e-5` | All 3,629 train; six completed epochs; best Epoch 3 by 732-clip CER; maximum 10, patience 3. | **51.34634%** CER; S/D/I **15,940/4,323/12,669**; 730 EOS, 2 cap hits. | Full-data adaptation strongly beat B0. Training loss kept falling after Epoch 3 while validation CER worsened, so final epoch was not selected. [Report](PHASE7_F1_RESULTS.md), [executed notebook](../notebooks/executed/phase7/f1_full_finetuning.ipynb). |
| Validation configuration audit | Re-decoded Epoch 3 with explicit attention mask and `task="transcribe"`. | All **732 predictions identical** to original; no score change. | Earlier generation warnings did not invalidate the measured F1 primary score. [F1 report](PHASE7_F1_RESULTS.md). |
| F1 repetition-penalty sweep | Same saved Epoch 3 checkpoint; only penalty varied across `1.00, 1.02, 1.05, 1.08, 1.10` on validation. | Best `1.08`: **48.39172%**, S/D/I **14,542/5,739/10,756**, 0 cap hits. | A modest decoding control lowered insertions, but deletions rose as the penalty increased; `1.08` beat `1.05` by only 0.15748 points. This was **validation-only**. [Report](PHASE7_F1_REPETITION_PENALTY_SWEEP_RESULTS.md), [plot](../assets/phase7_f1_repetition_penalty/07_selection_dashboard.png). |
| Independent F1 learning-rate mini-sweep | Original pinned Base initialization for each new run; `5e-6` and `2e-5`, with `1e-5` as reference; all use penalty 1.08. | `5e-6` Epoch 3 **51.05477%**; `1e-5` Epoch 3 **48.39172%**; **`2e-5` Epoch 5 47.23950%**. | `2e-5` was the validation-selected checkpoint before test opening. It later worsened despite falling training loss. [Report](PHASE7_F1_LR_RESULTS.md), [executed notebook](../notebooks/executed/phase7/f1_learning_rate_sweep.ipynb), [dashboard](../assets/phase7_f1_learning_rate/lr_sweep_comparison_dashboard.png). |
| L1 LoRA | Three independent adapters from original Base on the same frozen train/validation data. | Rank 8/alpha 16/LR `1e-4`: **60.68416%** at Epoch 7; rank 8/LR `2e-5`: **65.79509%**; rank 16/alpha 32/LR `1e-4`: **61.12540%**. | Best LoRA was **13.44 points worse** than selected full fine-tuning on validation, although it used roughly **1,216 MiB** allocated versus roughly **2,190 MiB** for the full LR runs. LoRA was **not selected** and was not tested on the sealed test set. [Report](PHASE7_LORA_RESULTS.md), [executed notebook](../notebooks/executed/phase7/lora_sweep.ipynb), [plot](../assets/phase7_lora/lora_comparison.png). |
| Phase 8 | Declared B0 versus the selected `2e-5` Epoch 5 F1 checkpoint, penalty 1.08; all 683 frozen test clips. | **101.23214% versus 44.57355%** CER. | The first unseen test comparison supports a large improvement over B0. No test-based configuration change was made. [Report](PHASE8_RESULTS.md), [executed notebook](../notebooks/executed/phase8/final_sealed_test.ipynb). |
| Post-hoc EDA | Joined saved B0/F1 test predictions to frozen metadata, without model execution. | 683/683 rows for each model rechecked; duration, reference length, recorded gender/age, month and domain slices produced. | Describes where errors remain; cannot be used for model or threshold selection. [EDA report](PHASE8_DESCRIPTIVE_EDA.md), [executed public notebook](../notebooks/Phase-8-Descriptive-EDA-Public.ipynb). |

### Exact selected F1 training and decoding recipe

Start from the original pinned Base revision, not from P1 or the initial F1 Epoch 3. Use 3,629 training clips, physical batch **2**, gradient accumulation **4**, effective batch **8**, approximately **454 optimizer updates per epoch**, AdamW weight decay **0.01**, linear schedule with approximately 5% warm-up, seed **0**, FP16 AMP training, FP32 feature extraction and generation, validation batch **8**, validation and checkpoint once per epoch, maximum **10 epochs**, patience **3**, and minimum full-validation corpus CER checkpoint selection. The selected independent run used learning rate **`2e-5`** and best **Epoch 5**. Generation was greedy/one beam with `task="transcribe"`, automatic language detection, `forced_decoder_ids=None`, explicit attention mask, maximum sequence length 448, no n-gram ban, and validation-selected `repetition_penalty=1.08`. The implementation and pinned stack are in the [trainer](../scripts/phase7_f1_train.py), [learning-rate orchestrator](../scripts/phase7_f1_lr_sweep.py), [final-test code](../scripts/phase8_final_test.py), and executed notebooks. This paragraph describes the **completed run**; do not change these settings in the report to make the outcome appear better.

## 5. How to explain the test examples and EDA

The [README example table](../README.md#examples-of-individual-test-outcomes) quotes three selected saved test references and corresponding predictions without IDs, audio, speaker metadata or provenance. Their clip CERs are **15.56%, 39.80%, and 80.30%**. They show a relatively good phrase with IPA substitutions and spacing changes, a partly correct but repeated phrase, and a severe over-generation example. The examples were selected **after** the test run. Cite the all-683 corpus score for performance; do not infer prevalence from three examples. There is no English translation verified for them.

The EDA uses the **same saved predictions** and recomputes subgroup corpus CER, with denominators retained in [tables](phase8_eda_tables/) and [JSON](PHASE8_DESCRIPTIVE_EDA_AGGREGATE.json). It does not rerun inference. The principal observations are:

| Test slice | Measured F1 CER | Denominator and interpretation |
|---|---:|---|
| 0.5–3-second clips | 49.10% | 94 clips, 1,833 reference code points. |
| >3–6 seconds | 50.03% | 148 clips, 7,305 reference code points. |
| >20–30 seconds | 42.46% | 62 clips, 13,758 reference code points. Short-clip errors are higher **within this split**, but duration is confounded with other factors. |
| Reference length 1–40 code points | 60.71% | 146 clips; a small denominator amplifies each error. |
| Reference length >120 | 41.51% | 205 clips; this does not prove longer speech is intrinsically easier. |
| Recorded female / male | 44.36% / 44.86% | 375 clips in six components / 308 in seven. Similarity is descriptive, not demographic parity. |
| Recorded age 60+ | 48.98% | 97 clips but only **two** independent components; cannot attribute the difference to age. |

F1 made fewer edits than B0 on **624** matched test clips, the same number on **14**, and more on **45**. A paired component-resampling interval for the B0-minus-F1 CER reduction was **50.16 to 68.27 percentage points** across 5,000 exploratory replicates over 13 observed components. It is not a guarantee for new speakers or domains. Month/domain categories often contain only one component; keep those panels as diagnostics or an appendix. Recorded test education is entirely `Unknown`. The age 60+ label appears in 17 train clips (0.5%) versus 97 test clips (14.2%); this describes split composition, not a causal explanation of errors. The full EDA caveats are in the [EDA report](PHASE8_DESCRIPTIVE_EDA.md).

## 6. Figure package for the report and slides

All figures below are already saved as image files. Read their captions in the linked phase reports before rewording them. Do not redraw them from incomplete notebook output.

| Recommended placement | Figure | Point to explain |
|---|---|---|
| Introduction/data | [Narration inventory](../assets/narration_inventory.svg) and [eligibility waterfall](../assets/phase1_eligibility_reasons.png) | Source scope versus supervised candidate pool. |
| Data methods | [Split proposal comparison](../assets/phase3_5b_proposal_comparison.png), [frozen-split funnel and duration](../assets/phase8/eda/01_dataset_funnel_and_duration.png) | Why split membership was frozen; train/validation/test denominators. |
| Processor method, optional | [Input/feature view](../assets/phase4/01_inputs_and_timing.png), [processor illustration](../assets/phase5b/processor_short_zoom.png) | Whisper feature/token pipeline, not model quality. |
| B0 result | [B0 edits](../assets/phase5b/03_edit_counts.png), [generation flags](../assets/phase5b/04_generation_flags.png) | 103.13% validation CER and over-generation. |
| Pilot result | [P1 learning and validation](../assets/phase6b/phase6b_learning_and_validation.png) | Supervised pilot learns but remains above B0. |
| Full training | [F1 loss versus CER](../assets/phase7_f1/01_f1_cer_and_training_loss.png), [edit composition](../assets/phase7_f1/02_validation_edit_composition.png) | Best epoch selected by validation, not by falling training loss. |
| Decoding comparison | [Penalty selection dashboard](../assets/phase7_f1_repetition_penalty/07_selection_dashboard.png) | Modest penalty reduces insertions with deletion trade-off. |
| Experiment selection | [Learning-rate dashboard](../assets/phase7_f1_learning_rate/lr_sweep_comparison_dashboard.png), [LoRA comparison](../assets/phase7_lora/lora_comparison.png) | Why `2e-5` full fine-tuning was selected and LoRA was not. |
| Main final result | [Phase 8 test dashboard](../assets/phase8/test_aggregate_dashboard.png), [error composition](../assets/phase8/eda/03_test_overall_errors_and_generation.png) | All 683 test clips, 101.23% to 44.57%, S/D/I trade-off and stopping. |
| Error analysis | [CER versus duration](../assets/phase8/eda/04_cer_vs_duration.png), [reference length](../assets/phase8/eda/05_cer_vs_reference_length.png), [paired outcomes](../assets/phase8/eda/07_clip_cer_distribution_and_pairing.png) | Heterogeneous remaining errors; subgroup CER uses subgroup character totals. |
| Appendix/qualified discussion | [Gender](../assets/phase8/eda/06a_cer_by_gender.png), [age](../assets/phase8/eda/06b_cer_by_age.png), [month/domain](../assets/phase8/eda/08_month_and_domain_diagnostics.png) | Explicitly show component counts and avoid causal or fairness claims. |

## 7. Code, notebooks and artifacts: where a collaborator finds them

### Repository completeness audit

The GitHub `main` tree was enumerated before this handoff update. It contained **42 notebook files**, **15 phase scripts**, **12 package-source files**, **5 tests**, **62 figure/asset files**, and **19 indexed artifact files**. The Phase 4–8 executed Kaggle notebook index includes nine downloaded run notebooks with saved output cells; early Phase 0–3.5B notebooks, source and public-results notebooks are also present. The README's then-broken decision/model-research links have been replaced by the public decision summary and historical model-research copy in this update. The main phase code, aggregate reports, plots, final test outputs and selected full-fine-tuning checkpoint references are indexed below. This is an **inventory check**, not a claim that every old notebook executes in a fresh environment or that Git LFS objects were downloaded and validated from a new clone.

| Material needed by the report writer | GitHub status | Practical note |
|---|---|---|
| Phase chronology, methods, metrics and interpretation | Present in README, this handoff and phase reports | Use aggregate reports to resolve rounded figures. |
| Figure images and EDA tables | Present under `assets/` and `docs/phase8_eda_tables/` | Captions and small-component caveats are provided. |
| Source code and saved notebook outputs | Present under `scripts/`, `src/`, `tests/`, and `notebooks/` | No rerun is needed for writing the report. |
| Selected model archives and result exports | Indexed under `artifacts/` | Model ZIPs are Git LFS pointers; actual LFS download was not rechecked here. |
| Raw audio and frozen private manifests | Intentionally absent | The author can cite frozen counts and hashes from checked reports. |
| LoRA best-checkpoint archive | Not supplied in local downloads | Its measured result and notebook outputs are present; do not claim a saved adapter is available. |

- **Read-only report writing needs:** this handoff, [README](../README.md), the phase reports in `docs/`, selected images in `assets/`, and the [executed notebook index](../notebooks/executed/README.md). No Kaggle session is required.
- **Early pipeline:** [Phase 0–3.5B notebooks](../notebooks/); [preparation code](../scripts/prepare_narration_baseline.py), [split proposal](../scripts/phase3_5b_proposal.py) and [freeze code](../scripts/freeze_phase3_5b.py). Frozen manifests and audio are intentionally outside the public repository.
- **Baseline and audit:** [B0 audit code](../scripts/audit_phase5b_saved_results.py), [Phase 6C generation equivalence code](../scripts/phase6c_validate.py), [protocol](PHASE4_ZERO_SHOT_PROTOCOL.md).
- **Training and final test:** [full F1 trainer](../scripts/phase7_f1_train.py), [learning-rate orchestration](../scripts/phase7_f1_lr_sweep.py), [LoRA notebook builder](../scripts/build_phase7_lora_notebook.py), [final-test code](../scripts/phase8_final_test.py). The corresponding self-contained and executed notebooks are indexed in [notebooks](../notebooks/executed/README.md).
- **Numerical exports:** phase reports and machine-readable aggregates in `docs/`; public EDA CSVs in [`phase8_eda_tables/`](phase8_eda_tables/); available run ZIPs and selected checkpoint references in the [artifact index](../artifacts/README.md). Large model archives are Git LFS pointers in a normal clone; access to LFS objects must be checked separately if someone needs actual weights. The report writer does not need them.
- **What is not here:** source audio, the frozen private manifests and the full intermediate-epoch learning-rate archive. No LoRA checkpoint archive was supplied. The repository includes the three deidentified example pairs in the README as well as earlier owner-requested exported prediction ZIPs and some notebook outputs with text/IDs, as disclosed in the [artifact index](../artifacts/README.md). Check any further quoted examples carefully before using them in a report.

## 8. Ready-to-use report outline and slide sequence

The report can follow: **Abstract → problem and scope → related model rationale → data provenance and curation → split and scoring method → engineering gates → B0/P1/F1/decoding/LR/LoRA experiments → predeclared test comparison → descriptive error analysis → limitations → future work**. A compact methods table should identify the frozen counts, model revision, text policy, GPU/precision, selection rule, and selected hyperparameters. Keep validation and test results in distinct tables. Put detailed subgroup and phase plots in an appendix if the main report becomes crowded.

For a presentation of about 10–12 slides: (1) question and Nyishi IPA task, (2) source and why Dataset B was deferred, (3) 6,718 → 5,086 → frozen split, (4) model/processor and CER, (5) B0 and error mode, (6) supervised engineering/P1 gate, (7) full-data F1 curve, (8) repetition-penalty and learning-rate selection, (9) LoRA comparison, (10) final test result, (11) three example error patterns and selected EDA, (12) limitations and later work. Speak to the figures; do not try to show every diagnostic in the main deck.

### Claims checklist for the author and presenter

1. Say **test** only for Phase 8's two declared configurations. All earlier CER optimization is **validation**.
2. Use **corpus CER** with reference-character denominators. Explain why B0 exceeds 100%: insertions can exceed reference length.
3. Label P1 as a pilot and Phase 6A/6C as engineering checks; neither is a final accuracy benchmark.
4. Report the selected F1 checkpoint as **LR `2e-5`, Epoch 5, penalty 1.08**. The original F1 Epoch 3 and 48.39% decoding result are earlier validation milestones.
5. Explain that LoRA saved measured allocated GPU memory but had worse CER. It was not selected or evaluated on test.
6. Mention remaining IPA symbol, spacing, deletion and occasional over-generation errors. A normal EOS ending does not guarantee a correct transcript.
7. State that subgroup differences are descriptive and limited by small independent component counts, label uncertainty and recorded metadata quality.
8. Do not claim expert-validated Nyishi phonetic correctness, population fairness, robustness to songs/YouTube or that a larger model would definitely improve CER.
9. Treat future IPA-Whisper Small, matched multilingual Whisper Small, and an IPA-target XLS-R CTC model as **recommendations for later work**, not experiments completed here. The [model research note](MODEL_RESEARCH_PUBLIC.md) documents why they are candidates.
10. Verify any screenshot/table transcribed into the final report against the linked aggregate JSON or phase report, and include the model revision and frozen split name in methods.

## 9. Limits and next work

The core outcome is verified from saved prediction archives, with exact 683-row coverage and independent S/D/I/CER recomputation; model inference was **not** rerun during the publication audit. Remaining limitations include no expert Nyishi label adjudication, uncertain speaker identity beyond the available grouping, one documented prompt overlap on different train/validation audio, recording-month imbalance, Dataset A Narration-only scope, and no independent multi-seed or external-corpus test. The post-hoc test slices should guide discussion, **not model tuning**.

The immediate project work is to assemble the report and presentation from the saved evidence, check selected examples/figures for readability, and present measured outcomes with caveats. Later research may predeclare matched IPA-Whisper Small versus multilingual Whisper Small and IPA-target CTC comparisons, and separately review Dataset B. Those are recommendations and require a fresh evaluation design before new model selection. The living [next steps](NEXT_STEPS.md) and [sprint plan](../SPRINT_PLAN.md) record that boundary.
