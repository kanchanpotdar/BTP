# Narration: next steps

Updated 1 November 2026 after review of the latest Phase 4 export. **Phase 4 engineering checks are complete.** No zero-shot validation/test CER or Nyishi fine-tuning has run.

## Completed evidence

- Minimal transcript formatting preserves raw strings and IPA marks.
- Phase 3.5B is accepted, frozen and hashed: 3,629 train, 732 validation, 683 test clips. Manifests remain private.
- 5,044 frozen and 42 set-aside labels pass tokenizer representation/length checks. The 42 remain excluded pending review.
- Twelve training clips pass audio/features; repeated first-clip inference and checkpoint save/reload match.
- Corrected full-token diagnostics confirm six EOS endings and six cap hits without EOS. All 12 decoded predictions match the earlier route.
- Actual inference environment and memory/time are recorded in [Phase 4 results](PHASE4_RESULTS.md). This does not establish transcription accuracy or training fit.

## Next: Phase 5 zero-shot validation baseline

1. Save the executed Kaggle notebook locally when available; its ZIP reports v6 while current prepared source is v7.
2. Prepare a private hash-checked bundle of the frozen validation split: **732 clips / 1.881808 hours**. Verify IDs, references, decoding and audio duration.
3. Verify code-point CER on known pairs, NFD combining marks and empty predictions. Keep ordinary spaces and all IPA marks.
4. Run a small validation preflight, then all validation rows with resumable per-sample outputs. Preserve raw and normalized hypotheses, full-token termination flags, S/D/I and failure statuses. Failed/empty predictions remain in the denominator.
5. Keep the pinned checkpoint/software, FP32 and greedy decoding. Record automatic language-prefix selection and inherited begin-suppression settings. Do not invent a Nyishi token or tune on test.
6. Report corpus CER, reference characters, sample/hour counts, failures, EOS/cap flags, runtime and memory. CER may exceed 100% due to repeated insertions.

## Then: supervised pilot and final evaluation

- Train a speaker-aware 1–2 hour subset from training only; select checkpoints by validation CER and measure training speed/memory.
- Expand only after the pilot is stable, using the same frozen references and scoring policy.
- Reserve the frozen test split for final evaluation of the pinned zero-shot and selected fine-tuned checkpoints. No test-driven selection.
- Preserve label/identity review limitations and the lack of expert Nyishi adjudication.

See [the protocol](PHASE4_ZERO_SHOT_PROTOCOL.md) for the CER contract. Public publication excludes audio, private manifests, transcript-bearing predictions and raw token sequences.
