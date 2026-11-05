# Phase 7 F1 learning-rate mini-sweep (Kaggle)

This is the next declared experiment after the completed F1 repetition-penalty
sweep and before the matched LoRA comparison. It runs two independent full-data
fine-tuning jobs at `5e-6` and `2e-5`. The already measured `1e-5` F1 result is
the reference; it does not need to be rerun.

## Upload exactly these inputs

Attach two private Kaggle datasets (or ZIP files) containing the bundle markers:

1. `phase7_f1_train_v1`: the full frozen training bundle, **3,629 clips** and
   **10.851637 hours**, with its `bundle_record.json` and checksums.
2. `phase5_validation_baseline`: the frozen validation bundle, **732 clips**,
   with references, audio, generation configuration, and checksums.

The recommended route is the self-contained notebook:

- `notebooks/Phase-7-F1-Learning-Rate-Mini-Sweep-Kaggle.ipynb`

It contains both runs, resume handling, comparison plots, and ZIP packaging, so
no separate trainer file is required. If using the command-line route instead,
upload or paste these two code files into the Kaggle session:

- `scripts/phase7_f1_train.py`
- `scripts/phase7_f1_lr_sweep.py`

The trainer downloads `neurlang/ipa-whisper-base` at the pinned revision using
Kaggle internet access. Do **not** attach the test bundle, F1 checkpoint,
prediction exports, master manifest, or raw project data.

Both new runs validate with the selected `repetition_penalty=1.08` policy and
the corrected explicit attention mask / `task="transcribe"` configuration.
They remain separate model initializations even though they run in one notebook.

## Kaggle setup cell

```python
# Confirm the GPU and required code files before spending compute.
from pathlib import Path
import torch

assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator first."
print(torch.cuda.get_device_name(0))
assert Path("/kaggle/working/phase7_f1_train.py").is_file()
assert Path("/kaggle/working/phase7_f1_lr_sweep.py").is_file()

# The Kaggle image normally provides these. Fail early before a long run if a
# required package is absent; do not mix unpinned Transformers versions into
# an active experiment.
import importlib.util
for package in ("transformers", "soundfile", "tqdm"):
    assert importlib.util.find_spec(package), f"Missing Kaggle package: {package}"
```

If the files are under a different working subdirectory, change the two paths
in the run cell below; do not change the trainer configuration.

## Run cell

```python
# Runs 5e-6 and 2e-5 independently. Each run starts from the original pinned
# pretrained Base model and writes private checkpoints/predictions separately.
import runpy
import sys

sys.argv = [
    "phase7_f1_lr_sweep.py",
    "--trainer", "/kaggle/working/phase7_f1_train.py",
    "--output-root", "/kaggle/working/phase7_F1_LR_private",
]
runpy.run_path("/kaggle/working/phase7_f1_lr_sweep.py", run_name="__main__")
```

The progress bars show bundle discovery, training updates, validation batches,
checkpointing, and the two-rate sweep. Each run uses physical batch 2,
gradient accumulation 4 (effective batch 8), validation batch 8, FP16 AMP
training, FP32 feature extraction/inference, a 10-epoch ceiling, patience 3,
weight decay 0.01, seed 0, and the validated Phase 6C generation policy.

## Expected outputs and checks

The private output tree contains one directory per rate:

```text
/kaggle/working/phase7_F1_LR_private/
  lr_5em06/F1_report_private.json
  lr_2em05/F1_report_private.json
```

Each report must show 3,629 training clips, 732 validation clips, test data
unused, and a valid best checkpoint. Compare each best validation CER against
the existing F1 reference (`1e-5`, best Epoch 3, 51.34634%) and retain the
complete epoch histories, S/D/I counts, EOS/cap counts, runtimes, and peak
memory. Do not select a winner from training loss alone.

Zip the private output directory for download, but keep it out of GitHub. The
next experiment is the independently initialized LoRA comparison; only after
validation selection should the predeclared configurations be evaluated on the
sealed 683-clip test set.

## Downloadable private ZIP cell

```python
# Package both LR runs for download from Kaggle. This archive is private and
# must not be committed to the public repository.
from pathlib import Path
import shutil

source = Path("/kaggle/working/phase7_F1_LR_private")
assert (source / "lr_5em06" / "F1_report_private.json").is_file()
assert (source / "lr_2em05" / "F1_report_private.json").is_file()
archive = shutil.make_archive(
    "/kaggle/working/phase7_F1_LR_private",
    "zip",
    root_dir=source.parent,
    base_dir=source.name,
)
print(archive)
```
