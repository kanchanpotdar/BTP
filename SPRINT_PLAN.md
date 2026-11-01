# SPRINT_PLAN.md — Nyishi IPA baseline and efficient fine-tuning

**Updated:** 1 November 2026. **Scope:** Dataset A Narration only. **First checkpoint:** `neurlang/ipa-whisper-base`.

## Objective and evidence status

Build a reproducible zero-shot reference, then measure how much supervised Nyishi IPA adaptation improves validation CER. Measure memory and training time alongside quality; do not assume FP16, LoRA or quantization preserves accuracy without comparison.

The user's attached “Nyishi ASR: IPA-Whisper Baseline and Fine-Tuning Plan” is a proposal reviewed against local evidence. Its general sequence is adopted below; rounded/proposed data totals and unverified performance statements are replaced with the existing frozen-split facts. Subsequent checked artifacts now establish B0 and the Phase 6A training smoke; Phase 6B remains prepared and unrun.

| Item | Verified current state |
|---|---|
| Working directory | Repository root in the private working copy |
| Source master | `manifests/master_A_v1_1.parquet`; preserve raw audio, labels, IDs and provenance |
| Frozen split | `phase3_5b_v1`; membership and SHA-256 records remain private |
| Train | **3,629 clips / 10.851637 hours / 15 components** |
| Validation | **732 clips / 1.881808 hours / 16 components** |
| Test | **683 clips / 1.891057 hours / 13 components** |
| Reference policy | NFD, outer trim, repeated ordinary-space collapse; preserve spaces and IPA marks |
| Set-aside labels | 42 remain excluded pending linguistic review, despite successful tokenization |
| Phase 4 | Engineering complete: 5,086 tokenizer checks; 12 training-only audio/features; repeat and reload checks |
| Measured generation limitations | 6/12 EOS endings; 6/12 full 448-token cap hits without EOS; one U+FFFD output |
| Model results | B0 full validation: 732 clips, 103.12924% corpus CER; test untouched |
| Phase 6A | Two finite FP16 AMP updates on 8 frozen-train clips, including checkpoint/resume; no CER |
| Phase 6B | Deterministic 581-clip / 1.002867 h subset prepared; pilot not run |

About 10.85 training hours justify a controlled transfer-learning experiment. Improvement, unseen-speaker generalization and tone recognition remain empirical questions. Preserve the frozen allocation rather than rounding away 0.85 hours or generating a new split. Known structural leakage checks pass; identity/label uncertainty, month shift and prompt overlap remain limitations.

## Phases and exit gates

Phases stay in sequential order. Completed evidence is reused; do not reopen Phase 4 simply to add illustrative figures.

| Phase | Work | Exit evidence | Status / planning effort |
|---|---|---|---|
| 0. Inventory/setup | Record sources, row/column counts, local environment and availability | Inventory and exact input paths/revisions | Completed historical phase |
| 1. Eligibility/QC | Versioned exclusions for missing labels, invalid/empty audio, duration and anomalies | Filtering waterfall, decode outcomes, preserved source hashes | Completed; retain review limitations |
| 2. Reference policy | Mechanical NFD/whitespace view; retain raw strings and all IPA marks | Counts, idempotence and symbol inventory | Completed |
| 2.5. Targeted review | Preserve symbol/identity review queues and conservative exclusions | Queue and unresolved decisions, not invented adjudication | Recorded; expert review remains pending |
| 3. Grouping/split proposal | Connect speaker/session/exact-audio components; inspect prompt overlap | Structural leakage and balance diagnostics | Historical proposal, superseded |
| 3.5. Category-aware proposal | Consider hours, month, age, gender and durations | Comparison with first proposal | Historical proposal, superseded |
| 3.5B. Accepted split freeze | Freeze membership and hash the accepted component assignment | Counts/hours/components and no known cross-split structural leakage | Completed; no new split search |
| 4. Processor/model smoke | Pin model/processor; check labels, audio/features, generation, memory, reload | Reviewed exports with full-sequence EOS/cap flags | Engineering complete; see `docs/PHASE4_RESULTS.md` |
| 5A. Baseline input preparation | Verify/package frozen validation data, references and hashes privately | Exactly 732 validation IDs, aligned audio/references; no test audio | Completed: 732 clips, 1.881808 h, verified private ZIP |
| 5B. Processor figures + B0 | Plot existing training smoke inputs; verify scoring; validation preflight and full zero-shot | Feature figures, tested CER logic, all validation outcomes with denominators and resumable exports | Completed: 732/732 validation clips; 103.12924% corpus CER |
| 6A. Supervised training smoke | Verify collator/prefixes, FP16 forward/backward/update and true checkpoint resume | Finite loss/gradients, verified updates, valid targets, resume state | Core engineering gate passed; 8 clips, 2 updates, no CER |
| 6B. P1 supervised pilot | Train a component-aware 1–2 h subset | Validation CER curve, sample decodes, measured VRAM/speed and stable checkpoints | Prepared: 581 clips / 1.002867 h / 15 components; not run |
| 7A. F1 full fine-tuning | Train all 10.851637 h using validated recipe | Best checkpoint by validation CER, full config and resumable history | Essential; runtime estimated only after pilot |
| 7B. L1 LoRA comparison | Same full training membership and evaluation contract; explicit adapter scope | CER/resource comparison and learning curves; recorded budgets | Recommended after F1; feasibility measured first |
| 7C. D1 data scaling | Optional nested 1, 2.5, 5 and full 10.851637 h training subsets | Hashed memberships, data-efficiency curve, compute/exposure totals | Optional; do not delay first complete baseline |
| 7D. S1 model scale | Conditional IPA-Whisper Small comparison | Newly pinned processor/model checks; matched data/scoring and documented budget | Conditional after Base evidence; Medium deferred |
| 8. Final test/error analysis | Predeclare selected model(s); evaluate on frozen test once per declared configuration | Test corpus CER, edit counts, group/duration/symbol analysis and uncertainty | Not run; 3–6 h analysis plus measured inference |
| 9. Handoff/thesis evidence | Consolidate reports, configurations, plots, limitations and provenance | Evidence-backed status; privacy-safe publication | Ongoing; 2–4 h final consolidation |

Effort ranges are human preparation/review estimates, not GPU runtime promises or a guaranteed 72-hour finish. Record actual hardware and pilot throughput before scheduling long runs.

## Experiment matrix and priority

| ID | Model and method | Training audio | Evaluation | Priority |
|---|---|---|---|---|
| B0 | Pinned IPA-Whisper Base, zero-shot | None | Full frozen validation, FP32 | Essential first reference |
| P1 | Base full fine-tuning, FP16 AMP | Fixed nested 1–2 h subset | Same full validation, FP32 | Essential pipeline/learning pilot |
| F1 | Base full fine-tuning, FP16 AMP | All **10.851637 h** | Same validation; final test only after selection | Essential supervised reference |
| L1 | Base LoRA, FP16 AMP | Same **10.851637 h** | Same validation and scoring | Recommended memory/quality comparison |
| D1 | Base full fine-tuning, nested subsets | 1, 2.5, 5, **10.851637 h** | Identical validation | Optional thesis study |
| S1 | `neurlang/ipa-whisper-small`, initially full AMP | Same full training set | Identical split/scoring; pin and verify independently | Conditional; no claim it is better |

Order: **B0 → training smoke → P1 → F1 → L1 → optional D1/S1 → preselected final test**. A short LoRA engineering check may precede L1, but a brief pilot does not establish its final quality ranking.

Each independent F1/L1/D1 comparison starts from the same pinned pretrained Base checkpoint. Do not silently warm-start F1 from P1 or larger D1 subsets from smaller-subset fine-tuned weights. Resume within an existing run is allowed and logged; warm-start studies are separate experiments.

## Precision and memory policy

- **Feature extraction:** use the matched processor in FP32, preserving original 16 kHz mono audio. No corpus denoising, pitch shifting, aggressive normalization, or audio/reference truncation.
- **B0 and comparable checkpoint evaluation:** retain FP32, deterministic greedy decoding and the existing CER policy. Mixed-precision training does not require mixed-precision evaluation.
- **Training:** begin with FP16 automatic mixed precision and gradient scaling. Do not simply cast every trainable parameter and optimizer state to FP16. Monitor finite loss/gradients and numerical skips/overflow behavior where exposed by the stack.
- **T4:** use FP16 AMP; do not assume native BF16. Hardware memory reported by the runtime, not an unverified quota, determines fit.
- **Memory controls:** start at micro-batch 2, accumulation 4 on one GPU. If needed, use micro-batch 1/accumulation 8. Verify peak memory through optimizer-state initialization, evaluation/generation and checkpoint resume, not just a forward pass.
- **Gradient checkpointing:** off initially for profiling. Enable if memory saving is needed or valuable; record the speed trade-off and check gradient flow. Cache configuration must be compatible with training/checkpointing and restored appropriately for evaluation.
- **Attention:** retain the measured SDPA route initially. Do not assume an unsupported FlashAttention implementation runs on T4.
- **LoRA:** measure actual trainable parameters and optimizer/activation memory; fewer trainable parameters do not eliminate activation memory.
- **Quantization:** QLoRA, 4/8-bit weights and optimizer changes are deferred unless ordinary AMP/memory controls cannot meet the measured budget. They add experimental variables and do not guarantee equal CER.
- Any FP16 inference alternative requires a separately recorded validation comparison before changing the declared FP32 evaluation reference. Preserve the Phase 4 FP32 results.

These settings aim for lower memory at acceptable measured quality; they do not guarantee either a specific VRAM budget or accuracy equivalence.

## Starting training configuration — provisional, not optimized

All values below are hypotheses to validate in 6A/P1. Final run files must record the actual configuration. Seed **0** retains the established project seed; the attachment's suggested 42 is not required.

| Setting | Full fine-tuning | LoRA comparison |
|---|---|---|
| Base initialization | Pinned `neurlang/ipa-whisper-base` revision from protocol | Same revision |
| Training/evaluation precision | FP16 AMP / FP32 | FP16 AMP / FP32 |
| Per-device batch / accumulation | 2 / 4; fallback 1 / 8 | Initially match |
| Effective batch, one GPU | 8 examples per full update; handle final partial batch explicitly | Initially match |
| Learning rate | `1e-5` | `1e-4` |
| Optimizer | AdamW | AdamW on declared trainable adapter parameters |
| Weight decay | 0.01 on eligible decay parameters; declare exclusions | 0.01 on eligible adapter parameters; declare exclusions |
| Schedule / warmup | Linear decay / 5% of planned optimizer updates | Same rule |
| Gradient clipping | 1.0 after AMP unscaling | Same rule |
| Full-run maximum | Up to 10 epochs; pilot has its own declared smaller step budget | Same maximum exposure budget initially |
| Evaluation / save | Initially every epoch; save enough state to resume | Initially match |
| Best checkpoint | Minimum validation corpus CER | Minimum validation corpus CER |
| Early stopping | 3 consecutive validation checks without improvement; threshold 0 initially | Same rule |
| Gradient checkpointing | Off initially; enable only with recorded reason/profile | Same rule; verify trainable gradient path |
| LoRA hypothesis | Not applicable | Rank 8; alpha 16; dropout 0.05; candidate `q_proj`/`v_proj` modules |
| Random seed | 0 | 0 |

The minimum-CER/zero-threshold checkpoint rule is an operational selection rule, not statistical significance. Preserve the declared maximum update budget when early stopping shortens a run; save actual updates and exposures.

At 3,629 examples and effective batch 8 on one GPU, an epoch is approximately **454 optimizer updates**, depending on sampler, accumulation remainder and data-loader behavior. Record actual counts. Do not confuse examples, micro-batches, optimizer updates, epochs and audio hours.

For LoRA, inspect actual encoder/decoder module names first. Log exact targets, rank, alpha, dropout, frozen/trainable counts and whether any biases, embeddings or heads are also trained. Verify nonzero adapter gradients, updates, save/load and inference before L1. Pin a compatible PEFT stack in a recorded environment; do not repeatedly change packages within an active kernel.

If loss fails to improve, inspect labels, prefix/shift handling, padding masks, optimizer membership and gradient flow before tuning learning rates. Only after those gates pass consider a small declared comparison such as `5e-6`, `1e-5`, `2e-5` for full fine-tuning.

## Processor visualization and supervised-target gates

### Small visual check in Phase 5B

Use a few existing **training smoke examples**, including short and near-30-second clips. Keep their figures separate from validation scoring.

1. Plot waveform amplitude against real time.
2. Plot the **exact processor input-feature tensor** as a heatmap, with time and Mel-band index axes. Label colors as scaled log-Mel feature values, not calibrated dB.
3. Mark the original-duration/padding boundary and show a cropped real-audio feature view. Document FFT window overlap near the boundary; padding need not appear as a perfectly sharp visual edge.
4. Record sample rate, channels, duration, feature shape/dtype and finite-value checks. Verify processor configuration (sample rate, FFT/hop, feature bins and padding window).
5. Do not interpret a plausible image as proof of transcript alignment or Nyishi correctness. Existing `[1,80,3000]` numerical checks already passed; this check documents the transformation rather than reopening Phase 4.

### Required training smoke in Phase 6A

- Build features with the pinned processor. Retain raw and approved normalized text views; reject empty references from supervised batches without deleting source rows.
- Pad audio features and target tokens independently. Mask label padding with **`-100`**, using the tokenizer padding mask rather than indiscriminately masking every EOS-valued token.
- Inspect actual tokenized labels, decoder-start handling and label shifting. Avoid duplicate start tokens; preserve the intended EOS training target.
- Explicitly resolve training prefix versus generation prefix. The smoke run's automatic English prefix is an observation, not approval to force English labels or invent Nyishi tokens. Save illustrative token IDs and the chosen policy before a learning run.
- Verify special-token-inclusive target lengths against decoder capacity, label round trips and absence of silent truncation. Recheck after the training prefix policy is implemented.
- Verify finite forward loss, gradients on intended parameters, and a successful scaled optimizer update. Check selected weights actually change; distinguish a gradient-scaler skipped update from a real update.
- Test saving **and resuming** optimizer, scheduler, AMP scaler and RNG/trainer state where supported. An inference-only model export is not a resumable training checkpoint.
- Log generation configuration before and after save/reload. Retain full-sequence EOS/cap checks from Phase 4. Never remove a poor example to make a gate appear to pass.

These gates were checked in Phase 6A on eight frozen-train clips. This establishes the training engineering path, not validation CER or generalization.

## Frozen data, nested subsets and fair comparisons

- Reverify the frozen hashes before every run. Preserve all 3,629 training rows for F1/L1; **10.851637 h** is the full training budget, not an instruction to select exactly 10 h.
- Build the pilot and D1 subsets only from frozen training IDs. Use deterministic, nested membership: `1 h ⊂ 2.5 h ⊂ 5 h ⊂ full train`. A 1–2 h pilot should be compatible with that ordering when feasible.
- Select full clips with speaker/component diversity and audit duration/category/IPA coverage. Exact hour totals may overshoot slightly because clips remain intact; record actual clips, hours and group coverage. Do not describe unresolved identities as verified speakers.
- Keep the validation set and reference/scoring policy identical across runs. Do not select training examples by looking at test labels or predictions.
- For F1/L1, predeclare a common maximum optimizer-update/exposure budget with matched effective batch and data order as far as practical. Learning rates can differ. If early stopping differs, show actual updates, learning curves, audio exposures and GPU-hours.
- **Equal optimizer updates do not equal compute:** report time/memory and model-selection budget. Do not call this a compute-matched experiment without measured matching.
- For D1, predeclare the budget convention. A fixed maximum epoch count gives different update/compute totals as data grows; record that confound. A fixed-update study is an optional separate control, not evidence that every curve isolates data quantity alone.
- Repeat the most consequential close comparison with additional seeds if feasible. Short pilots and one data-efficiency curve cannot by themselves establish the ultimate winning method or the cause of a plateau.

## Evaluation and decisions

Use the contract in `docs/PHASE4_ZERO_SHOT_PROTOCOL.md`: corpus **code-point CER** over per-clip NFD/trim/collapsed-ordinary-space strings, including spaces and all IPA marks. Sum per-clip S/D/I and divide by summed reference characters; do not average clip CERs or concatenate across clip boundaries.

- Retain raw/normalized hypotheses, failed/empty decodes, repeated outputs, U+FFFD and cap hits. Failed/empty predictions contribute deletions and remain in the denominator. CER can exceed 100%.
- Report clips, hours, reference characters, S/D/I, CER, failures, runtime/RTF with defined timing boundaries and GPU allocated/reserved peaks. WER is secondary only if word boundaries are defensible.
- Diagnose omissions/empty output, repetitive insertions, IPA-mark substitutions, per-group variation and train/validation loss divergence. A signal pattern suggests investigations; it does not prove a cause.
- Compute per-component/speaker results with character denominators. Bootstrap independent speaker/session components where feasible, recomputing corpus CER; report interval limitations from the small number of held-out groups.
- Do not treat **1–2 absolute CER percentage points** as a universal significance or equivalence threshold. If a non-inferiority tolerance is desired for LoRA, predeclare it before comparing results and consider uncertainty and seed variation. Without it, report the observed trade-off rather than claiming equivalent accuracy.
- Select configurations with validation only. At Phase 8, predeclare the final comparison set (at least pinned B0 and selected supervised checkpoint); run those on the frozen test with the same scoring policy. Do not use test scores to revise training, decoding or split membership.
- No arbitrary target CER, promised improvement or Nyishi tone capability is claimed. An honest negative result still satisfies reproducible engineering/evaluation deliverables; scientific improvement requires measured evidence.

## Immediate next action

Run the prepared [Phase 6B Kaggle pilot](docs/PHASE6B_KAGGLE_RUN.md) using the private fixed one-hour frozen-train bundle and the existing 732-clip Phase 5A validation bundle. The [source notebook](notebooks/Phase-6B-Kaggle-Pilot.ipynb) verifies hashes/targets, trains from the pinned pretrained Base, evaluates every epoch with the exact B0 policy, and checkpoints. The Phase 6A [aggregate review](docs/PHASE6A_RESULTS.md) passed its training smoke gate but produced no CER. Do not use test or claim pilot quality until an executed export is checked.

## Reproducibility, scope and handoff

For every actual run, save manifests/subset hashes, source/notebook hash, model/processor revision, compatible package versions (including PEFT when used), seed, GPU, full configuration, training and decoding precision, exact commands, steps/epochs/audio exposures, timing boundaries, memory, learning curves, checkpoints, predictions and gate status. Preserve original Kaggle exports and the executed notebook when available. Public GitHub copies exclude audio, private manifests, transcript-bearing predictions, raw generation IDs, personal metadata, secrets and model weights.

**Deferred scope:** Dataset B YouTube, new split searches, semantic IPA rewrites, blanket denoising, unaligned >30 s training, automatic quantization and mandatory larger-model comparisons. These would change the research question or add unvalidated variables before the first complete result.

### References and status history

- Existing evidence: `docs/PHASE4_RESULTS.md`, `docs/PHASE4_RESULTS_AGGREGATE.json`, frozen `freeze_record.json` and `docs/NEXT_STEPS.md`.
- [Hugging Face Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper): collator/label-padding principles; adapt historical examples to the pinned stack and IPA target policy.
- [PyTorch 2.11 AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html): autocast, gradient scaling and accumulation mechanics.
- [Transformers v4.57.1 GPU efficiency guide](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_train_gpu_one.md): mixed precision, accumulation/checkpointing and their trade-offs.
- 1 November 2026: revised with user-authorized experiment matrix, AMP/memory controls, processor figures, supervised-target/resume gates, full-vs-LoRA comparison, optional nested data scaling and group-aware uncertainty. Frozen splits/reference/evaluation policy are retained; no code, training or baseline was run by this plan edit.
