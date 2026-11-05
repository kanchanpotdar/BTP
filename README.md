# Nyishi IPA-ASR: reproducible narration baseline

This repository documents a reproducible, Narration-only speech-recognition study for Nyishi transcribed into IPA. The first model is `neurlang/ipa-whisper-base`. The project is organized as a sequence of auditable phases: data inventory, curation, split design, processor/model checks, a zero-shot baseline, supervised pilots, validation optimization, and finally full training.

## Research question

> How much does supervised adaptation improve Nyishi IPA transcription over a fixed zero-shot reference, under a frozen speaker/session-disjoint evaluation design?

**Updated 5 November 2026.** See the [F1 report](docs/PHASE7_F1_RESULTS.md), [learning-rate comparison](docs/PHASE7_F1_LR_RESULTS.md), [repetition-penalty sweep](docs/PHASE7_F1_REPETITION_PENALTY_SWEEP_RESULTS.md), and [machine-readable aggregate results](docs/PHASE7_F1_LR_RESULTS_AGGREGATE.json).

The repository separates measured results from proposals and keeps the final test split sealed until model selection is complete.

## Current status at a glance

| Area | Status | Evidence |
|---|---|---|
| Dataset A Narration curation | Complete for the frozen scope | Phase 0–3.5B notebooks and manifests |
| Frozen split | Accepted and locally hashed | `phase3_5b_v1`: 3,629 / 732 / 683 clips |
| Processor/model engineering | Passed | Phase 4 reports |
| B0 zero-shot validation | Complete | 732/732 validation clips; 103.13% corpus CER |
| Phase 6A training smoke | Passed | Two finite AMP updates and checkpoint resume; no accuracy metric was produced |
| Phase 6B P1 pilot | Complete | Best Epoch 3 CER 107.26%; B0 remained better; insertion-heavy |
| Phase 6C validation optimization | Passed | Batch 8 exactly matched single-example outputs; validation speed improved 2.14× |
| Phase 7 F1 full fine-tuning | Complete | Best Epoch 3 CER 51.35%; validation-only repetition-penalty sweep selected 1.08 at 48.39% |
| Phase 7 F1 learning-rate mini-sweep | Complete | `2e-5`, Epoch 5: 47.24% validation CER; test remains sealed |
| LoRA comparison | Not run | Matched follow-up experiment remains |
| Final sealed test evaluation | Not run | Test set remains intentionally untouched |

## Start here

Public readers can inspect the chronological phase notebooks and saved plots without private data. For the current result, open the [learning-rate public results notebook](notebooks/Phase-7-F1-Learning-Rate-Results-Public.ipynb). To reproduce training with authorized private inputs, follow the [learning-rate Kaggle run guide](docs/PHASE7_F1_LR_KAGGLE_RUN.md) and [self-contained notebook](notebooks/Phase-7-F1-Learning-Rate-Mini-Sweep-Kaggle.ipynb). These instructions were inspected here; training and inference were not rerun during this documentation sync.

## Frozen data contract

- Scope: Dataset A Narration only; Dataset B YouTube is out of scope.
- Train: **3,629 clips / 10.851637 hours**.
- Validation: **732 clips / 1.881808 hours**.
- Test: **683 clips / 1.891057 hours**, sealed.
- Known component, speaker-group, session, and exact-audio overlaps across the accepted split: zero.
- Primary metric: corpus-level character error rate (CER) over NFD-normalized IPA strings, with substitutions, deletions, and insertions reported separately. Apply NFD, outer trim, and repeated ordinary-space collapse to each reference and hypothesis; retain spaces and IPA marks. Sum per-clip edits over summed reference code points; lower CER is better and insertions can make CER exceed 100%.
- Raw references and private manifests remain outside the public release.

## Phase chronology

### Phase 0 — inventory and local setup

- **Purpose:** establish what is available before any filtering, transcript editing, or model work.
- The source-of-truth Narration master contained **6,718 rows and 54 columns**, and matched the 59 raw Narration shards row-for-row.
- The inventory recorded raw audio locations, transcript fields, duration basis, speaker/session fields, review flags, and provenance without rewriting the source.
- Dataset B (YouTube) was explicitly kept out of the first baseline scope.
- The initial audit identified **5,651 rows with a non-empty reference**, **708 clips flagged longer than 30 seconds in the inventory**, and **three zero-frame recordings**. These are inventory facts, not deletion instructions.

| Inventory item | Measured value | Why it mattered |
|---|---:|---|
| Dataset A Narration master rows | 6,718 | Complete starting population |
| Master columns | 54 | Preserved metadata, transcript, review, and provenance fields |
| Rows with a usable-looking reference | 5,651 | Upper bound before audio/duration checks |
| Clips flagged over 30 seconds in the inventory | 708 | Must be segmented deliberately; never silently truncated |
| Zero-frame recordings | 3 | Require exclusion from the first supervised candidate view and retention in the audit |

![Narration source inventory](assets/narration_inventory.svg)

**Figure interpretation — source inventory.** This is a data-inventory illustration, not a model result. It shows the relationship between the preserved Narration source, derived audit tables, and later candidate/split views. The important design decision was to keep the master and its provenance intact while making versioned derived artifacts.

### Phase 1 — eligibility and audio checks

- **Purpose:** apply explicit first-pass rules to create a candidate pool for later grouping and split design.
- The master and full-decode audit joined one-to-one (**6,718 / 6,718**). There were no full-decode failures and no sample-rate or channel-count disagreements.
- Every row received a reason and boolean flags; source rows were not deleted. The first applicable reason was used for the mutually exclusive waterfall below.

| First-pass outcome | Clips | Decoded hours remaining | Interpretation |
|---|---:|---:|---|
| All master rows | 6,718 | 25.792 | Starting population |
| After missing/empty reference | 5,651 | 20.877 | 1,067 rows had no usable reference |
| After zero-frame check | 5,648 | 20.877 | Three labelled rows had no decoded frames |
| After audio-issue review hold | 5,647 | 20.876 | One duration-discrepancy record was held for review |
| After the 0.5–30 s rule | **5,086** | **14.826** | Initial structurally valid candidate pool |

The 5,086 candidates were not yet the train/validation/test split. They were the input to transcript normalization, review queues, and group-aware split construction. The duration rule excluded 561 rows in the mutually exclusive waterfall. The detailed Phase 1 review counted **528 reference-present clips over 30 seconds**; this is narrower than the Phase 0 inventory flag because Phase 1 applies the usable-reference predicate before describing labelled durations. Short-clip and other edge-case flags were retained in the all-row audit. Long recordings remained preserved for future alignment-aware segmentation.

![Phase 1 eligibility reasons](assets/phase1_eligibility_reasons.png)

![Phase 1 duration distribution](assets/phase1_duration_distribution.png)

![Phase 1 candidate hours by speaker](assets/phase1_speaker_hours.png)

![Phase 1 speaker status](assets/phase1_speaker_status.png)

**Figure interpretation — Phase 1 plots.** The eligibility chart explains why rows leave the first candidate view; it is not a quality score. The duration plot shows why the 30-second boundary was treated as a preparation rule rather than a truncation rule. The speaker-hour and speaker-status plots show that the 5,086 candidates were not automatically safe for random splitting: the largest public speaker IDs contributed a disproportionate share of hours, and **967 candidates were marked train-only**. That evidence motivated the group-aware split work in Phase 3.

### Phase 2 — transcript formatting

- **Purpose:** define a conservative, reproducible text representation for tokenizer checks and CER without making linguistic substitutions.
- Input: the 5,086 Phase 1 candidates. The approved policy was:

| Operation | Applied? | Effect |
|---|---|---|
| Unicode NFD | Yes | Makes combining-mark representation consistent |
| Trim outer whitespace | Yes | Removes accidental leading/trailing whitespace |
| Collapse repeated ordinary U+0020 spaces | Yes | Removes formatting-only repeated spaces |
| Rewrite IPA symbols, diacritics, punctuation, or tone marks | **No** | Linguistic meaning remains unchanged |
| Preserve `transcript_raw` beside the derived label | **Yes** | Every transformation remains auditable |

- **497 rows** changed under the final mechanical policy; all 497 changes were repeated ordinary-space cleanup. NFD alone changed zero rows, and no normalized label became empty.
- The inventory contained **102 distinct raw and normalized code points**. The normalized labels contained 521,747 code points, including IPA combining marks; these counts describe Unicode code points, not phonemes.
- The policy was idempotent: applying it a second time produced the same normalized label. No model tokenizer or split assignment was used in this phase.

![Phase 2 normalization and label length](assets/phase2_normalization_and_length.png)

![Phase 2 duration versus label length](assets/phase2_duration_vs_label_length.png)

**Figure interpretation — Phase 2 plots.** These figures describe the text representation before training. One compares raw and normalized label-length distributions; the other relates decoded duration to reference length. They are useful for spotting formatting changes and unusually long/short examples, but they are not evidence of recognition accuracy. The downstream CER contract uses the same NFD, outer-trim, and repeated-space policy for references and hypotheses.

### Phase 2.5 — targeted review

- **Purpose:** isolate small, high-impact uncertainties before constructing speaker/session-disjoint splits.
- A targeted symbol queue contained **42 clips (0.201 decoded hours)**: 3 with `:`, 20 with `;`, 19 with `[`, plus one combining-acute case. Categories overlap, so their bars must not be added together.
- A separate group-readiness review covered **five proposed speaker-ID merges**, **nine train-only/proposed groups**, **1,745 recording sessions**, and **398 repeated normalized transcript rows**. Repeated text was not treated as duplicate audio.
- The 967 train-only candidates represented **2.474 hours** and were kept out of automatic held-out assignment. Exact-audio duplicate grouping and final split assignment were deliberately deferred to Phase 3.
- Review queues preserved exact source locators privately, but no automatic linguistic correction, speaker merge, or split decision was applied here.

![Phase 2.5 review workload](assets/phase2_5_review_workload.png)

**Figure interpretation — review workload.** This plot is a queue-sizing and risk-management view. It shows which symbol classes and proposed group workloads needed human attention; it does not claim that those symbols were wrong or that proposed speaker merges were true identities. The practical outcome was a clean handoff to group-aware split construction with uncertainty explicitly recorded.

### What these four phases established

The early pipeline moved from **6,718 preserved source rows** to **5,086 mechanically eligible candidates**, while keeping the rejected/uncertain records and their reasons available for audit. Phase 0 measured the population, Phase 1 made audio/reference eligibility explicit, Phase 2 fixed only formatting-level transcript variation, and Phase 2.5 surfaced human-review and leakage risks. None of these phases produced a model accuracy result; they made the later frozen split and CER comparison scientifically interpretable.

### Phase 3 — first group split proposal

- Constructed indivisible groups from proposed speaker/session links and exact waveform duplicates.
- Found zero known structural leakage but rejected the first proposal because short-clip and recording-month balance was unsuitable.

![Phase 3 split sizes](assets/phase3_split_sizes.png)

**Figure interpretation — split sizes.** This figure compares the number of clips and hours assigned to the first train, validation, and test proposal. It is a planning diagnostic: a numerically large split is not automatically a representative split.

![Phase 3 category balance](assets/phase3_category_balance.png)

**Figure interpretation — category balance.** These panels show why the proposal was rejected. Short clips and recording-month composition were uneven across held-out sets, creating a risk that the model would be evaluated on a different recording regime from the one it learned.

### Phase 3.5 — category-aware revision

- Searched whole-component assignments while considering hours, month, age, duration, gender, and held-out balance.
- Rejected the revision because validation was dominated by one component and female hours, despite a better internal search score.

![Phase 3.5 split comparison](assets/phase3_5_split_comparison.png)

**Figure interpretation — revised split sizes.** The category-aware search moved whole components between partitions while preserving group integrity. The chart is useful for comparing candidate allocations, but the internal search objective alone was not sufficient for acceptance.

![Phase 3.5 category comparison](assets/phase3_5_category_comparison.png)

**Figure interpretation — revised categories.** The revised candidate improved some numerical balance measures but concentrated validation in one component and one gender distribution. It was therefore rejected in favor of a more usable design.

### Phase 3.5B — accepted frozen split

- Selected the more balanced split while preserving component/speaker/session/exact-audio separation.
- Frozen membership and hashes are retained privately.

![Phase 3.5B proposal comparison](assets/phase3_5b_proposal_comparison.png)

**Figure interpretation — accepted split comparison.** This is the final design comparison used to select `phase3_5b_v1`. It shows the trade-off between train capacity and held-out balance after enforcing component, speaker/session, and exact-audio separation. The split is frozen; it must not be redesigned after seeing model scores.

### Phase 4 — processor and model engineering

- Verified processor/tokenizer round trips, feature shape and finiteness, model loading, generation configuration, and a training-only smoke path.
- This phase established compatibility; it did not establish Nyishi accuracy.

![Phase 4 input and timing checks](assets/phase4/01_inputs_and_timing.png)

**Figure interpretation — processor/model smoke checks.** The figure summarizes input duration coverage and basic timing behavior during the engineering smoke test. It establishes that the processor and model can accept the intended waveform format; it does not measure transcription quality.

### Phase 5A — validation-input preparation

- Prepared and independently checked the private 732-clip validation bundle.
- Confirmed waveform hashes, audio decoding, references, and the 64,137-code-point denominator.

### Phase 5B — B0 full-validation zero-shot baseline

- Ran the pinned model on all 732 validation clips under the declared greedy FP32 policy.
- Result: **103.12924% corpus CER** with 66,144 edits over 64,137 reference code points.
- Counts: **36,780 substitutions, 4,171 deletions, 25,193 insertions**.
- Diagnostics: 130 generation-cap hits without EOS and 52 U+FFFD-containing outputs.
- Corrected plots were regenerated from saved predictions; inference was not rerun.

![B0 CER by duration](assets/phase5b/01_cer_by_duration.png)

**Figure interpretation — CER by duration.** Each duration bin reports corpus CER for the validation clips in that bin. This shows whether errors are concentrated in short or long recordings, but bins with different reference-character totals should not be compared as if they had equal statistical weight.

![B0 CER by component](assets/phase5b/02_cer_by_component.png)

**Figure interpretation — CER by anonymous component.** This plot checks whether the baseline behaves differently across recording components. Large differences may indicate acoustic, speaker/session, or data-distribution variation; they are diagnostic, not proof of a single cause.

![B0 edit counts](assets/phase5b/03_edit_counts.png)

**Figure interpretation — edit composition.** Substitutions change one character into another, deletions omit reference characters, and insertions add extra characters. Insertions are a particularly important B0 failure signal because they indicate over-generation rather than only IPA symbol confusion.

![B0 generation diagnostics](assets/phase5b/04_generation_flags.png)

**Figure interpretation — generation health.** These counts show how often decoding reached the token cap, produced replacement characters, or otherwise returned an unusual output. These outputs remain included in CER; they are not silently discarded.

![B0 CER by stopping behavior](assets/phase5b/05_cer_by_stopping.png)

**Figure interpretation — stopping behavior.** The plot compares CER for EOS-terminated and cap-hit predictions. A higher cap-hit CER would indicate that failing to stop is associated with severe over-generation, but this remains an association rather than a causal diagnosis.

![B0 generation length](assets/phase5b/06_generation_length_distribution.png)

**Figure interpretation — output lengths.** The length distribution reveals whether the decoder tends to produce short omissions, plausible-length strings, or unusually long continuations. The 448-token limit is an evaluation safeguard, not a claim that every clip should have that many tokens.

### Phase 6A — supervised training smoke

- Ran two real FP16 AMP optimizer updates on eight frozen training examples.
- Verified finite losses/gradients, nonzero parameter changes, and save/reload of optimizer, scheduler, scaler, and RNG state.
- This passed the engineering gate but produced no validation CER.

**Meaning of the result.** Phase 6A answered “can the supervised training loop update and resume safely?” It did not answer “does the model recognize Nyishi IPA?” The lower loss on the second update is therefore an engineering observation, not a generalization result.

**Plots.** No privacy-safe accuracy plot was produced for Phase 6A because it used eight training clips only and did not run validation. Its evidence is the aggregate engineering report and checkpoint-resume checks.

### Phase 6B — P1 controlled pilot

- Trained independently from the pinned pretrained checkpoint on 581 training clips for three epochs.
- Validation covered all 732 clips after every epoch.
- CER improved during the pilot from 159.66% to the latest verified Epoch 3 value of 107.26%, but did not beat B0's 103.13%.
- Epoch 3 remained insertion-heavy, so the pilot demonstrates learning but not a finished recipe.

**Figure and result interpretation.** The falling training loss and falling validation CER show that the model learned from the supervised subset. However, Epoch 3 still remained worse than B0 and had many insertions, so “learning occurred” must not be reported as “the model is accurate.”

![Phase 6B learning and validation](assets/phase6b/phase6b_learning_and_validation.png)

**Figure interpretation — learning and validation.** This plot shows the P1 training-loss curve beside validation CER and generation diagnostics. It demonstrates learning on the one-hour pilot, but the best P1 CER remained above B0, so it was a pilot for recipe validation rather than the final model.

![Phase 6B best CER by duration](assets/phase6b/phase6b_best_cer_by_duration.png)

**Figure interpretation — duration slices.** This plot shows how the selected P1 checkpoint behaved across duration bins. It is diagnostic only: the official selection criterion remains the full-validation corpus CER, not the best-looking subgroup.

### Phase 6C — validation optimization and F1 preparation

The audit compared single-example generation with batch 4 and batch 8 on a deterministic 12-clip duration-spread subset.

| Validation path | Runtime | Peak GPU memory | Equivalence |
|---|---:|---:|---|
| Single clip | 10.99 s | 380.6 MiB | Reference |
| Batch 4 | 5.49 s | 663.3 MiB | Exact match; 2.00× faster |
| Batch 8 | 5.14 s | 941.7 MiB | Exact match; 2.14× faster |

Batch 8 is the selected validation configuration. It preserved token IDs, decoded text, normalized text, and aggregate S/D/I counts. The private examples still show repeated sequences, high insertions, and one no-EOS cap hit; these are model-error signals, not batching failures.

**Figure interpretation.** The Phase 6C timing comparison is an engineering plot rather than an accuracy plot. Its purpose is to show that validation can be accelerated without changing predictions. The memory bars show the measured allocation on the T4 audit run; they are not a universal memory guarantee for every batch or software environment.

Evidence is in [`docs/PHASE6C_RESULTS.md`](docs/PHASE6C_RESULTS.md) and [`docs/PHASE6C_EQUIVALENCE_AGGREGATE.json`](docs/PHASE6C_EQUIVALENCE_AGGREGATE.json).

**Plots.** Phase 6C did not produce an accuracy plot because it was a 12-clip generation-equivalence and runtime audit. Its evidence is the timing/memory table and exact token/text/CER equality report; no plot was necessary to establish the gate.

## Phase 7 F1: full supervised fine-tuning

F1 started independently from `neurlang/ipa-whisper-base` revision `949f7c4e4ee374b5202bc5886f6e4552e5ae2371`, rather than from the P1 checkpoint.

- All 3,629 frozen training clips.
- Effective batch size 8: physical batch 2 with accumulation 4, with physical batch 1/accumulation 8 as the measured fallback.
- Learning rate `1e-5`; AdamW weight decay `0.01`.
- FP16 AMP training and FP32 validation.
- Batch-8 validation, once per epoch.
- Initial maximum of 10 epochs with patience-3 early stopping.
- Checkpoint every epoch; select minimum full-validation corpus CER.
- Save resumable optimizer/scheduler/scaler/RNG state and epoch telemetry.
- Keep all predictions, manifests, audio, IDs, and weights private.

The executed F1 run completed six epochs before patience-3 early stopping. The best checkpoint was **Epoch 3: 51.35% validation CER**, compared with B0's 103.13%. The model used all 3,629 training clips and all 732 validation clips; the test set remained sealed.

![F1 CER and training loss](assets/phase7_f1/01_f1_cer_and_training_loss.png)

**What this plot means.** Training loss falls steadily, but validation CER reaches its minimum at Epoch 3 and then oscillates. This is why the best validation checkpoint—not the final checkpoint—is selected.

![F1 validation edit composition](assets/phase7_f1/02_validation_edit_composition.png)

**What this plot means.** F1 reduces insertions strongly relative to B0, while substitutions remain the largest remaining error family at the selected epoch. The error breakdown is a diagnostic, not a change to the CER definition.

![F1 generation health](assets/phase7_f1/03_generation_health.png)

**What this plot means.** At Epoch 3, 730/732 validation clips ended with EOS, two reached the generation cap, and no replacement-character outputs were recorded. This indicates that stopping behavior is no longer the dominant failure mode.

![F1 runtime and GPU memory](assets/phase7_f1/04_runtime_and_gpu_memory.png)

**What this plot means.** The run stayed well within the T4 memory budget. Runtime includes training and batched validation, and is reported as an engineering measurement rather than a quality metric.

### Validation decoding ablation

After F1, a validation-only audit supplied an explicit attention mask and modern `task="transcribe"` configuration. It reproduced the original Epoch 3 predictions exactly on all 732 clips. A declared repetition-penalty ablation reached **48.55% CER**, with insertions falling from 12,669 to 11,117 and cap hits falling from 2 to 0. A no-repeat 3-gram rule worsened CER to 53.00% and was rejected.

Epoch 3 has **32,932 edits / 64,137 reference code points**: 15,940 substitutions, 4,323 deletions and 12,669 insertions. The repetition-penalty candidate has **31,138 edits / 64,137**, under a separately tuned decoding configuration.

The aggregate F1 report is [`docs/PHASE7_F1_RESULTS.md`](docs/PHASE7_F1_RESULTS.md), with the public results notebook at [`notebooks/Phase-7-F1-Results-Public.ipynb`](notebooks/Phase-7-F1-Results-Public.ipynb).

The six recorded epoch durations sum to **2,308.25 seconds (38.47 minutes)** on a T4, including training and validation but excluding setup, checkpoint writes and restart downtime. F1 resumed after Epoch 2. The exported wall-time field of 1,528.74 seconds (25.48 minutes) covers only the final session; it is not the full-run runtime.
### Phase 7 F1 repetition-penalty sweep

The selected F1 Epoch 3 checkpoint was re-evaluated on all 732 validation clips
with a predeclared sweep of `repetition_penalty` values `1.00`, `1.02`, `1.05`,
`1.08`, and `1.10`. All other generation settings, normalization and CER
scoring remained fixed; the test set was not used.

| Penalty | Validation CER | Substitutions | Deletions | Insertions | Cap hits |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 51.34634% | 15,940 | 4,323 | 12,669 | 2 |
| 1.02 | 50.88638% | 15,290 | 4,655 | 12,692 | 2 |
| 1.05 | 48.54920% | 14,785 | 5,236 | 11,117 | 0 |
| **1.08** | **48.39172%** | **14,542** | **5,739** | **10,756** | **0** |
| 1.10 | 48.48527% | 14,513 | 6,093 | 10,491 | 0 |

Penalty `1.08` is the validation-selected candidate. It reduces decoder
over-generation relative to `1.00`, but its margin over `1.05` is only 0.1575
absolute CER points. This is not a test-set result or a claim of statistically
significant generalization.

![Repetition penalty CER comparison](assets/phase7_f1_repetition_penalty/01_cer_comparison.png)

![Repetition penalty edit composition](assets/phase7_f1_repetition_penalty/03_error_composition_percent.png)

![Repetition penalty generation health](assets/phase7_f1_repetition_penalty/05_generation_health.png)

![Repetition penalty selection dashboard](assets/phase7_f1_repetition_penalty/07_selection_dashboard.png)

See the full public report at [`docs/PHASE7_F1_REPETITION_PENALTY_SWEEP_RESULTS.md`](docs/PHASE7_F1_REPETITION_PENALTY_SWEEP_RESULTS.md), aggregate data at [`docs/PHASE7_F1_REPETITION_PENALTY_SWEEP_AGGREGATE.json`](docs/PHASE7_F1_REPETITION_PENALTY_SWEEP_AGGREGATE.json), and the public notebook at [`notebooks/Phase-7-F1-Repetition-Penalty-Sweep-Public.ipynb`](notebooks/Phase-7-F1-Repetition-Penalty-Sweep-Public.ipynb).

### Phase 7 F1 learning-rate mini-sweep

Two independent full-data F1 runs were completed at learning rates `5e-6` and
`2e-5`, retaining the existing `1e-5` result as the reference. Both runs used
the frozen 3,629-clip training set and 732-clip validation set, effective batch
8, FP16 AMP training, batch-8 validation, and the validation-selected
`repetition_penalty=1.08` policy. The test set was not attached or evaluated.

| Configuration | Best epoch | Validation CER | S | D | I | EOS | Cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5e-6` | 3 | 51.05477% | 15,212 | 5,021 | 12,512 | 732 | 0 |
| `1e-5` reference | 3 | 48.39172% | 14,542 | 5,739 | 10,756 | 732 | 0 |
| **`2e-5`** | **5** | **47.23950%** | **14,503** | **5,583** | **10,212** | **732** | **0** |

The `2e-5` run is the current validation-selected candidate, improving the
`1e-5` reference by 1.15222 absolute CER points. Its validation CER worsened
after Epoch 5 while training loss continued to fall, so Epoch 5—not the final
Epoch 8 checkpoint—is selected. The lower-rate run peaked at Epoch 3 and then
oscillated. These results establish a validation preference, not unseen-test
performance.

![Learning-rate comparison dashboard](assets/phase7_f1_learning_rate/lr_sweep_comparison_dashboard.png)

![Learning-rate CER curves](assets/phase7_f1_learning_rate/cer_curves.png)

![Learning-rate training-loss curves](assets/phase7_f1_learning_rate/training_loss_curves.png)

**Figure interpretation — learning-rate comparison.** The higher rate reaches a
lower validation CER and remains stable through Epoch 5, but later epochs
degrade. The lower rate learns more slowly and does not beat the reference.
Training loss alone is not a selection criterion because it continues falling
after validation quality has begun to worsen.

Measured runtime was 36.83 minutes for `5e-6` and 48.00 minutes for `2e-5`;
peak allocated memory was approximately 2,190 MiB for both. Public files include
aggregate metrics, plots, and a results notebook only; private predictions and
checkpoints remain excluded.

See the [learning-rate report](docs/PHASE7_F1_LR_RESULTS.md),
[aggregate metrics](docs/PHASE7_F1_LR_RESULTS_AGGREGATE.json), and [public
results notebook](notebooks/Phase-7-F1-Learning-Rate-Results-Public.ipynb).

**How F1 is interpreted.** Training loss is diagnostic. The selected model is the epoch with the lowest full-validation CER. If CER stops improving for three validations, early stopping ends the run. The repetition-penalty result remains validation-only until the sealed test evaluation.

## Reproducibility and privacy

- Pin the model revision and compatible package versions in the Kaggle notebooks.
- Record seed, GPU, package versions, data/configuration hashes, optimizer steps, runtime, memory, and checkpoint paths.
- Public contents include source code, notebooks, aggregate metrics, and privacy-safe plots only.
- Private contents include audio, manifests, IDs, transcript-bearing predictions, checkpoints, provenance, and Kaggle output bundles.

## Repository map

- `notebooks/` — source and public-results notebooks in chronological phase order.
- `scripts/` — preprocessing, split, rendering, and audit code.
- `docs/` — phase reports, protocols, aggregate metrics, and next steps.
- `assets/` — privacy-safe aggregate plots and processor illustrations.
- `archive/kaggle_runs/` — local-only Kaggle notebooks and private run archives; not part of the public release.

## Limitations

Labels and proposed identities lack expert Nyishi adjudication. Recording-month shift and one overlapping prompt on distinct train/validation audio remain documented limitations. Validation improvements do not establish unseen-test performance or tone recognition. The 12-clip Phase 6C equivalence result is scoped to that checkpoint, sample and runtime.

## Next steps

1. Run the matched LoRA comparison independently from the original pinned checkpoint, using the same frozen data and validation contract.
2. Select between full fine-tuning and LoRA using validation CER, edit composition, generation health, runtime, and memory.
3. Predeclare the final comparison set and evaluate the selected configuration once on the sealed 683-clip test set.
3. Select the F1/LoRA configuration and decoding setting using validation only.
4. Prepare the private frozen 683-clip test bundle.
5. Evaluate B0 and the selected supervised model once on the sealed test set.

See [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) for the active handoff.
