# Nyishi IPA baseline model research — public copy

Research date: 28 October 2026. This is a **historical pre-experiment shortlist**; for completed B0, fine-tuning, LoRA and final-test evidence use the [current handoff](REPORT_PRESENTATION_HANDOFF.md). This public copy omits source-row identifiers; the original local research note is preserved separately. Sources are primary model cards, authors' repositories, Hugging Face documentation and official Kaggle announcements. At the time of this research note, no candidate weights had been downloaded, no training packages installed and no model inference/fine-tuning run. GPU feasibility below was an engineering estimate, not a measured Nyishi benchmark or guaranteed memory budget.

## Decision

Start with a minimally screened, unenhanced Dataset A baseline. Use the existing reproducible labels; avoid blanket denoising or amplitude filtering. Compare a small general multilingual ASR starting point against an IPA-adapted checkpoint of the same family and a CTC model with explicit IPA targets. Include an existing compact speech-to-IPA model as an optional fourth comparator.

Suggested initial set:

1. `neurlang/ipa-whisper-base` for the quickest IPA-pilot and training-system check.
2. `openai/whisper-small` versus `neurlang/ipa-whisper-small` with matched Nyishi training data and adaptation settings; this tests whether IPA intermediate training helps.
3. `facebook/wav2vec2-xls-r-300m` with a Nyishi IPA CTC head/vocabulary, for an architecture comparison.
4. Optional: `nikhilnavas/branchshine-cr`, a much smaller already-IPA model with an explicit fine-tuning CLI. Its training conventions/coverage require scrutiny.

Do not train all candidates immediately. First validate one small recipe on a reproducible short subset and the intended target symbols, then expand. Report failed attempts as such; no model has demonstrated Nyishi accuracy in this project yet.

## Shortlist and evidence

| HF checkpoint | Existing output/initialization | Adaptation route | Practical assessment |
|---|---|---|---|
| [neurlang/ipa-whisper-base](https://huggingface.co/neurlang/ipa-whisper-base) | Already fine-tuned toward IPA; base-size architecture, roughly 73–74M parameters | Further supervised fine-tuning; Whisper-style LoRA is an architecture-compatible route requiring integration validation | Best small pilot; Kaggle first; local M1 smoke tests subject to memory/speed |
| [neurlang/ipa-whisper-small](https://huggingface.co/neurlang/ipa-whisper-small) | Already IPA-adapted Whisper-small, roughly 244M architecture | LoRA or full tuning after memory testing | Strong first substantive IPA candidate on one T4; not proven best for Nyishi |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | General multilingual orthographic ASR, not ready-made Nyishi IPA recognition | Supervise on IPA labels using the appropriate processor; full fine-tuning or LoRA | Useful matched control for IPA-Whisper-small; do not call orthographic zero-shot output an IPA benchmark |
| [neurlang/ipa-whisper-medium](https://huggingface.co/neurlang/ipa-whisper-medium) | IPA-adapted medium-size Whisper, roughly 769M architecture | LoRA; 8-bit LoRA or 4-bit QLoRA candidates if required and compatible | Later scale comparison; more compute/activation memory than small; no measured T4/local fit here |
| [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | Multilingual self-supervised encoder; requires an ASR output head | Train a new CTC IPA head; freeze some encoder parameters or test architecture-specific LoRA | Good controlled CTC baseline. It is adaptable to IPA, not already a Nyishi IPA transcriber |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | Multilingual CTC ASR with language adapters | Native MMS adapter + suitable new output vocabulary/head | Useful later parameter-efficient comparator; native adapters are not LoRA; 1B forward/activation memory still matters |
| [nikhilnavas/branchshine-cr](https://huggingface.co/nikhilnavas/branchshine-cr) | About 25M parameters, already speech-to-IPA CTC | Author supplies full/partial/head fine-tuning, vocabulary extension and NFD option | Attractive compact baseline without needing quantization; author reports memory figures, but we have not reproduced them |

The IPA-Whisper cards describe multilingual Common Voice audio paired with phonemizer-derived IPA labels, rather than manually verified Nyishi transcriptions. Base/small describe 15,000 recordings and medium 75,000. Their inherited model-card descriptions and placeholder-looking evaluation entries do not provide credible Nyishi CER evidence. Treat these as testable pretrained starting points. Their upstream reported 24GB training GPU does not establish a minimum fine-tuning requirement or prove a low-memory recipe.

BranchShine-CR explicitly supports a fine-tuning manifest, vocabulary extension and NFD; author code provides full and partial freezing options, rather than an established PEFT-LoRA wrapper. The card warns that G2P-derived conventions omit tone/stress/allophonic detail and that tightly cropped clips may be under-transcribed. These limitations are relevant for Nyishi. Its custom package and claimed memory/accuracy need independent verification before relying on it. [Author card and recipe](https://huggingface.co/nikhilnavas/branchshine-cr).

Additional already-phonetic candidates: [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) has a phoneme-recognition checkpoint; [changelinglab/PhoneticXeus](https://huggingface.co/changelinglab/PhoneticXeus) provides direct IPA phones and a roughly 0.6B encoder-based model with custom training code. PhoneticXeus's author repository records a September 2026 inference correction, so pin the tested revision. Its HF card metadata and author-repository LICENSE differ; reconcile before selecting it. Neither has been validated for this project's target conventions. [PhoneticXeus authors' repository](https://github.com/changelinglab/PhoneticXeus).

[facebook/omniASR-CTC-300M](https://huggingface.co/facebook/omniASR-CTC-300M) remains an optional later model: authors supply a fairseq2 fine-tuning recipe, but the card does not establish exact IPA coverage or a plug-and-play PEFT LoRA/QLoRA path. Its ~2GiB card figure is inference memory, not a fine-tuning budget. Do not equate broad language coverage with our target-symbol support.

## LoRA, quantized LoRA and evidence level

Hugging Face has an explicit [8-bit Whisper + LoRA ASR guide](https://huggingface.co/docs/peft/main/task_guides/int8-asr), demonstrated for Whisper-large-v2/Marathi. This establishes a supported Whisper-family technique; it does not establish Nyishi performance, a tested IPA-Whisper recipe or memory fit for this project. The guide includes legacy API names: use mutually compatible pinned stable packages when implementing, not a blind copy of its snippets.

LoRA trains low-rank adapters while freezing most base weights. Eight-bit LoRA uses an 8-bit base. QLoRA refers here to 4-bit base-weight quantization plus LoRA; it is not another name for any adapter method. PEFT documents [quantized adaptation](https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization), and Transformers documents [NF4/4-bit configuration](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/bitsandbytes.md). Generic QLoRA support does not establish that every custom speech model can use it without adaptation.

For base/small Whisper, begin with ordinary LoRA if it fits. Quantization adds variables and does not remove audio activation memory. For medium, evaluate 8-bit LoRA before making a 4-bit recipe the default. For CTC with a new IPA head, ensure the head is actually trainable and saved with adapters; a frozen randomly initialized output head cannot learn the new labels. MMS's [documented adapter recipe](https://huggingface.co/blog/mms_adapters) is a distinct PEFT method and a simpler initial path than inventing MMS LoRA.

Do not assume model-card inference examples demonstrate successful fine-tuning. Record checkpoint revision, trainable parameter counts, loss movement, peak memory, processor configuration and decoded IPA output in an actual smoke test before calling a candidate operational.

## GPU and local feasibility

Current local machine, inspected without changing configuration: MacBookAir10,1, Apple M1 GPU with eight cores, 8GB unified memory. This is not the old NVIDIA laptop and its 8GB is shared system memory, not dedicated CUDA VRAM. Use it for metadata/signal analysis and small inference/training experiments if the runtime permits; a complete training benchmark on Kaggle is more practical. No PyTorch/MPS benchmark was run here.

Kaggle's [September 2026 P100-retirement announcement](https://www.kaggle.com/product-announcements/735239) says P100 was retired on 15 September, with T4×2 remaining. Each T4 has 16GB; two cards are not an automatically pooled 32GB GPU. Some older Kaggle documentation still describes P100, so prefer the newer announcement and inspect the actual allocated hardware/quota. [Official notebook environment documentation](https://www.kaggle.com/docs/notebooks).

A single T4 is a sensible initial target for Whisper base/small LoRA and a short-clip XLS-R experiment; medium/MMS require more conservative batches and explicit profiling. This is a feasibility assessment, not a guarantee. Begin with micro-batch 1, gradient accumulation/checkpointing and bounded audio lengths. Select FP16 on T4 rather than assuming native BF16. Checkpoint/resume to survive session limits. DDP may increase throughput but usually replicates the model rather than solving an individual-model memory limit.

Current [bitsandbytes installation documentation](https://huggingface.co/docs/bitsandbytes/main/en/installation) includes Apple Silicon/macOS support, including CPU wheels. That is not proof that the particular CUDA Whisper QLoRA training path runs on this M1's MPS backend. Do not assume equivalence; prefer an initially tested CUDA recipe for the main baseline.

## What IPA support must mean in this project

Separate three cases: already IPA-adapted checkpoints; generic models that can be supervised toward IPA; and unverified language coverage claims. None is proof of correct Nyishi tone/length/dental realization.

Before training, perform a processor/tokenizer round trip on the actual NFD reference inventory and representative complete labels. Check ɛ, ə, ɨ, ŋ, ɡ, ʔ, ː and all combining marks and rare flagged symbols; ensure no unknown/suppressed targets or target-length truncation. Whisper byte-level representations can encode text without implying learned phonetic competence. Do not invent a Whisper language='nyishi' or an IPA language token. Review decoder prefix/generation settings for each checkpoint; ordinary language forcing can bias orthographic output.

For character CTC, use an explicit documented IPA vocabulary and trainable head, preserving NFD combining marks. If using a phoneme tokenizer on references that are already IPA, do not run English phonemization over them; Hugging Face's [Wav2Vec2Phoneme tokenizer docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme) explicitly distinguish already-phonemic inputs. Fix vocabulary policy independently of test-set labels and report out-of-vocabulary held-out symbols.

Phone recognizers may omit spaces or use phone separators; these are not automatically equivalent to our space-bearing character references. Report primary NFD CER under a documented convention; a boundary-insensitive score or PER is supplementary and requires a valid phone segmentation policy. Do not silently strip marks/spaces to make a model look better.

## Baseline without extensive cleaning

Yes: start with unenhanced audio and minimal defensible screening. There is no need to finish denoising every clip or perfect every training transcript before a controlled pilot. There is also no useful literal zero-preprocessing recipe: features, labels, duration handling and evaluation design still matter.

Minimum conditions:

1. Exclude missing reference labels and zero-frame recordings from supervised batches; keep originals and record exclusion counts.
2. Hold the known duration-mismatch clip for review. Use an initial model-appropriate short-clip pool, for example 0.5–30s for a controlled first Whisper recipe. Do not truncate a long waveform while retaining its full transcript.
3. Choose and document one label policy (e.g. provisional A_v2) and preserve A_v1/raw for sensitivity checks. Review unclear held-out labels; expert sign-off remains absent.
4. Create one fixed speaker/session-aware validation/test design with reliable references before reporting benchmark scores. Do not use published leaking random row splits. Do not compare models on different pools/splits.
5. Build an initial 1–2h training subset only after defining the held-out data. Use it to test pipeline/memory/IPA outputs, then expand based on validation results. Avoid selecting subsets or noise thresholds from test results.
6. Apply the model's required feature preparation but no blanket corpus denoising, pitch shifting or aggressive normalization. Keep matched raw-audio versus any later enhancement experiments separate.

The project is ready to prepare a baseline recipe, not to claim baseline accuracy. No split/training was created in this research task.

## Fresh audio signal audit

`reports/audio_signal_audit_2026-10-28/` contains a read-only audit script, full sample metrics, JSON summary and locator-based listening-candidate queue. All 6,718 Narration clips were read; the master file's SHA-256 was unchanged.

- Three zero-frame source rows were identified and retained in the private audit. No additional nonempty clip was entirely zero-valued.
- Zero read/decode errors in this pass.
- 104 nonempty recordings have absolute RMS below -50dBFS.
- 320 nonempty recordings have at least half their measured 20ms frames below -50dBFS.
- Zero recordings meet the provisional screen of at least 1% of samples at absolute amplitude >=0.999. This is not a guarantee that all audio is distortion-free.

These are transparent signal measurements, not calibrated noise, SNR, speech-duration or intelligibility judgments. Very quiet speech can trigger absolute-level flags. Steady background noise and steady speech can both produce a small level spread. No row was removed or marked high-noise, and no heuristic SNR was revived as an exclusion rule. Four synthetic measurement checks (empty/zero/known RMS/saturation) passed.

For a stronger later noise review, [Microsoft DNSMOS P.835](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/README.md) estimates speech/background/overall perceptual quality without clean reference recordings. It still needs validation on Nyishi, quiet recordings and short clips; it is not a transcript checker. A VAD plus quality-estimator and stratified listening review can prioritize clips, but weights/dependencies were not installed or downloaded here. Human listening remains necessary for reliable noise and label decisions.
