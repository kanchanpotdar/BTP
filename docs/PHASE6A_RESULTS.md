# Phase 6A verified aggregate results — 1 November 2026

**Assessment:** the core supervised-training smoke gates passed on the supplied export. The result is engineering readiness for a controlled Phase 6B pilot, not evidence of Nyishi transcription accuracy or of a suitable full-training schedule.

## Evidence checked

- The private ZIP contains model weights, model/generation configuration, the training-state checkpoint and report. All five ZIP entries passed CRC verification. Its report bundle checksum matches the local versioned training-only Phase 6A bundle.
- The executed notebook's 12 core cells exactly match the prepared source. Input/model/target/collator, first update, resume and final report cells finished without errors. The setup cell retains an earlier package-install/restart `SystemExit`; the subsequent recorded runtime used the pinned versions.
- Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.57.1, Tokenizers 0.22.2, Hub 0.36.2, Safetensors 0.8.0, Tesla T4, seed 0.
- Eight distinct frozen-training smoke clips, with target lengths including EOS of 6–164 tokens, passed audio/features and label checks. The collator preserved EOS and masked only padding. The chosen supervised target was IPA text tokens plus EOS; the model inserted the decoder-start token.
- First optimizer update: four finite losses (7.7091, 6.7228, 8.1757, 5.6173), finite preclip gradient norm 86.1486, one AMP attempt at scale 256, nonzero probed weight change about 1.003e-5. Measured update loop 0.618 s, peak allocated 1,955 MiB.
- After saving and reloading model, optimizer, scheduler, scaler and RNG state, a second update completed on the same eight clips. Four finite losses (5.7745, 5.7119, 6.8619, 4.4859), one AMP attempt, nonzero probed weight change about 1.004e-5. Measured resumed update loop 0.439 s, peak allocated 3,035 MiB. These timings exclude setup, model load and checkpoint I/O and must not be extrapolated to a pilot.
- No validation CER or test result was produced. Both updates used the same eight clips; the lower second-pass losses cannot establish generalization.

## Configuration issue to close before Phase 6B validation

The saved model's `generation_config.json` includes `forced_decoder_ids=[[1,null],[2,50359]]` and no explicit `language`/`task` fields. B0's measured generation setup explicitly had `forced_decoder_ids=None`, `language=None`, `task=None`, `do_sample=False`, `num_beams=1`, `suppress_tokens=[]` and inherited `begin_suppress_tokens=[220,50257]`. Phase 6A did not generate validation predictions, so this mismatch did not affect the training smoke. Before any Phase 6B validation CER, make an explicit generation-config copy with the B0 settings and verify/log it after checkpoint reload. Avoid silently using the saved model default or changing the B0 evaluation policy.

The sprint plan also asks for illustrative target token IDs and a before/after reload generation-config log. The executed notebook records target lengths and policy but not those complete details. Record them privately as a pre-pilot addendum; do not rerun 6A training solely for documentation.

## Next step at the time of this report

The contemporaneous handoff was to run the prepared [Phase 6B Kaggle pilot](PHASE6B_KAGGLE_RUN.md) from the original pinned pretrained Base checkpoint on the deterministic one-hour frozen-training subset. That pilot, Phase 6C, and the subsequent Phase 7 F1 run are now complete; see [Phase 6B results](PHASE6B_RESULTS.md), [Phase 6C results](PHASE6C_RESULTS.md), and [Phase 7 F1 results](PHASE7_F1_RESULTS.md). The final test remains sealed.

Private ZIP, executed notebook, token IDs, source mapping, weights and training state remain local. This report contains aggregate evidence only.
