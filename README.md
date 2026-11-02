# Nyishi IPA-ASR: reproducible narration baseline

This repository documents a reproducible, Narration-only speech-recognition study for Nyishi transcribed into IPA. The first model is `neurlang/ipa-whisper-base`. The project is organized as a sequence of auditable phases: data inventory, curation, split design, processor/model checks, a zero-shot baseline, supervised pilots, validation optimization, and finally full training.

## Research question

> How much does supervised adaptation improve Nyishi IPA transcription over a fixed zero-shot reference, under a frozen speaker/session-disjoint evaluation design?

**Updated 2 November 2026.** See the [F1 report](docs/PHASE7_F1_RESULTS.md) and [machine-readable aggregate results](docs/PHASE7_F1_RESULTS_AGGREGATE.json).

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
| Phase 7 F1 full fine-tuning | Complete | Best Epoch 3 CER 51.35%; validation-only repetition-penalty ablation reached 48.55% |
| LoRA comparison | Not run | Matched follow-up experiment remains |
| Final sealed test evaluation | Not run | Test set remains intentionally untouched |

## Start here

Public readers can inspect the chronological phase notebooks and saved plots without private data. For the current result, open the [F1 public results notebook](notebooks/Phase-7-F1-Results-Public.ipynb). To reproduce training with authorized private inputs, follow the [F1 Kaggle run guide](docs/PHASE7_F1_KAGGLE_RUN.md) and [self-contained notebook](notebooks/Phase-7-F1-Self-Contained-Kaggle.ipynb). These instructions were inspected here; training and inference were not rerun during this documentation sync.

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

- Audited the master Dataset A source and identified labelled, missing, long, and undecodable records.
- Preserved raw audio, IDs, transcripts, review fields, and provenance.

### Phase 1 — eligibility and audio checks

- Built the structurally valid labelled candidate view.
- Recorded duration, frame, decoding, and transcript eligibility facts without deleting source records.

### Phase 2 — transcript formatting

- Created a derived normalized view using Unicode NFD, outer trimming, and repeated ordinary-space collapse.
- Retained raw strings beside normalized strings.

### Phase 2.5 — targeted review

- Produced review queues for symbols, audio quality, speaker/session uncertainty, and possible curation decisions.
- Review queues were treated as uncertainty records, not automatic deletion rules.

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

1. Run the matched LoRA comparison from the original pinned checkpoint.
2. Select the F1/LoRA configuration using validation only.
3. Prepare the private frozen 683-clip test bundle.
4. Evaluate B0 and the selected supervised model once on the sealed test set.

See [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) for the active handoff.
