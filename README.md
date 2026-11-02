# Nyishi IPA-ASR: reproducible narration baseline

This repository documents a reproducible, Narration-only speech-recognition study for Nyishi transcribed into IPA. The first model is `neurlang/ipa-whisper-base`. The project is organized as a sequence of auditable phases: data inventory, curation, split design, processor/model checks, a zero-shot baseline, supervised pilots, validation optimization, and finally full training.

## Research question

> How much does supervised adaptation improve Nyishi IPA transcription over a fixed zero-shot reference, under a frozen speaker/session-disjoint evaluation design?

The repository separates measured results from proposals and keeps the final test split sealed until model selection is complete.

## Current status at a glance

| Area | Status | Evidence |
|---|---|---|
| Dataset A Narration curation | Complete for the frozen scope | Phase 0–3.5B notebooks and manifests |
| Frozen split | Accepted and locally hashed | `phase3_5b_v1`: 3,629 / 732 / 683 clips |
| Processor/model engineering | Passed | Phase 4 reports |
| B0 zero-shot validation | Complete | 732/732 validation clips; 103.13% corpus CER |
| Phase 6A training smoke | Passed | Two finite AMP updates and checkpoint resume |
| Phase 6B P1 pilot | Complete | Best Epoch 3 CER 109.10%; B0 remains better |
| Phase 6C validation optimization | Passed | Batch 8 exactly matches single-example outputs on the audit subset |
| F1 full fine-tuning | Not run | Next required experiment |
| LoRA comparison and final test | Not run | Later phases |

## Frozen data contract

- Scope: Dataset A Narration only; Dataset B YouTube is out of scope.
- Train: **3,629 clips / 10.851637 hours**.
- Validation: **732 clips / 1.881808 hours**.
- Test: **683 clips / 1.891057 hours**, sealed.
- Known component, speaker-group, session, and exact-audio overlaps across the accepted split: zero.
- Primary metric: corpus-level character error rate (CER) over NFD-normalized IPA strings, with substitutions, deletions, and insertions reported separately.
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

![Phase 3 category balance](assets/phase3_category_balance.png)

### Phase 3.5 — category-aware revision

- Searched whole-component assignments while considering hours, month, age, duration, gender, and held-out balance.
- Rejected the revision because validation was dominated by one component and female hours, despite a better internal search score.

![Phase 3.5 split comparison](assets/phase3_5_split_comparison.png)

![Phase 3.5 category comparison](assets/phase3_5_category_comparison.png)

### Phase 3.5B — accepted frozen split

- Selected the more balanced split while preserving component/speaker/session/exact-audio separation.
- Frozen membership and hashes are retained privately.

![Phase 3.5B proposal comparison](assets/phase3_5b_proposal_comparison.png)

### Phase 4 — processor and model engineering

- Verified processor/tokenizer round trips, feature shape and finiteness, model loading, generation configuration, and a training-only smoke path.
- This phase established compatibility; it did not establish Nyishi accuracy.

![Phase 4 input and timing checks](assets/phase4/01_inputs_and_timing.png)

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

![B0 CER by component](assets/phase5b/02_cer_by_component.png)

![B0 edit counts](assets/phase5b/03_edit_counts.png)

![B0 generation diagnostics](assets/phase5b/04_generation_flags.png)

![B0 CER by stopping behavior](assets/phase5b/05_cer_by_stopping.png)

![B0 generation length](assets/phase5b/06_generation_length_distribution.png)

### Phase 6A — supervised training smoke

- Ran two real FP16 AMP optimizer updates on eight frozen training examples.
- Verified finite losses/gradients, nonzero parameter changes, and save/reload of optimizer, scheduler, scaler, and RNG state.
- This passed the engineering gate but produced no validation CER.

### Phase 6B — P1 controlled pilot

- Trained independently from the pinned pretrained checkpoint on 581 training clips for three epochs.
- Validation covered all 732 clips after every epoch.
- CER improved during the pilot from 149.92% to 109.10%, but did not beat B0's 103.13%.
- Epoch 3 remained insertion-heavy, so the pilot demonstrates learning but not a finished recipe.

### Phase 6C — validation optimization and F1 preparation

The audit compared single-example generation with batch 4 and batch 8 on a deterministic 12-clip duration-spread subset.

| Validation path | Runtime | Peak GPU memory | Equivalence |
|---|---:|---:|---|
| Single clip | 10.99 s | 380.6 MiB | Reference |
| Batch 4 | 5.49 s | 663.3 MiB | Exact match; 2.00× faster |
| Batch 8 | 5.14 s | 941.7 MiB | Exact match; 2.14× faster |

Batch 8 is the selected validation configuration. It preserved token IDs, decoded text, normalized text, and aggregate S/D/I counts. The private examples still show repeated sequences, high insertions, and one no-EOS cap hit; these are model-error signals, not batching failures.

Evidence is in [`docs/PHASE6C_RESULTS.md`](docs/PHASE6C_RESULTS.md) and [`docs/PHASE6C_EQUIVALENCE_AGGREGATE.json`](docs/PHASE6C_EQUIVALENCE_AGGREGATE.json).

## F1: next supervised experiment

F1 should start independently from the original pinned pretrained checkpoint.

- All 3,629 frozen training clips.
- Effective batch size 8: physical batch 2 with accumulation 4, with physical batch 1/accumulation 8 as the measured fallback.
- Learning rate `1e-5`; AdamW weight decay `0.01`.
- FP16 AMP training and FP32 validation.
- Batch-8 validation, once per epoch.
- Initial maximum of 10 epochs with patience-3 early stopping.
- Checkpoint every epoch; select minimum full-validation corpus CER.
- Save resumable optimizer/scheduler/scaler/RNG state and epoch telemetry.
- Keep all predictions, manifests, audio, IDs, and weights private.

The measured estimate is approximately **30–40 minutes for three epochs** on a comparable T4 after batched validation optimization. A ten-epoch run may take roughly **1.5–2.5 hours**, depending on clip lengths and checkpoint overhead.

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

## Next steps

1. Run F1 with the validated batch-8 evaluator and the frozen 3,629-clip training set.
2. Select the best F1 checkpoint using validation CER only.
3. Run the matched LoRA comparison as a separate experiment.
4. Evaluate the preselected final comparison on the sealed 683-clip test set only after all validation decisions are complete.

See [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) for the active handoff.
