"""Recreate anonymous training-only waveform/log-Mel illustrations; no model weights/inference.

Private smoke bundle required. Uses pinned Transformers 4.57.1's NumPy CPU feature
extractor and downloads only the public pinned preprocessor configuration if uncached.
Do not publish the input bundle, source notebook, WAVs or intermediate feature arrays.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import transformers
from transformers import WhisperFeatureExtractor

MODEL_ID = "neurlang/ipa-whisper-base"
REVISION = "949f7c4e4ee374b5202bc5886f6e4552e5ae2371"


def render(smoke_root, repository_root, cache_dir):
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Recreate these illustrations with Transformers 4.57.1.")
    smoke_root, repository_root = Path(smoke_root), Path(repository_root)
    record = json.loads((smoke_root / "bundle_record.json").read_text())
    if record.get("contains_validation_or_test_audio") is not False:
        raise ValueError("Expected the private training-only smoke bundle.")
    with (smoke_root / "smoke_manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 12:
        raise ValueError("Expected twelve training-only smoke rows.")
    rows.sort(key=lambda r:(float(r["duration_seconds"]),r["sample_id"]))
    extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache_dir)
    assert extractor.sampling_rate == 16000 and extractor.feature_size == 80
    assert extractor.n_fft == 400 and extractor.hop_length == 160
    output = repository_root / "assets/phase5b"
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "scope": "two training-only smoke illustrations; not validation plots",
        "transformers_version": transformers.__version__, "numpy_version": np.__version__,
        "backend": "NumPy CPU; no model weights or inference",
        "sampling_rate": 16000, "mel_bins": 80, "n_fft": 400, "hop_length": 160,
        "input_window_seconds": 30, "model_revision": REVISION,
        "bitwise_equality_to_saved_kaggle_tensors_claimed": False,
        "examples": [],
    }
    for alias,row in [("short",rows[0]),("long",rows[-1])]:
        path = (smoke_root / row["audio_file"]).resolve()
        path.relative_to(smoke_root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["audio_sha256"]
        waveform, sr = sf.read(path, dtype="float32", always_2d=True)
        assert sr == 16000 and waveform.shape[1] == 1
        waveform = waveform[:,0]
        duration = len(waveform)/sr
        assert abs(duration-float(row["duration_seconds"])) < 1e-9
        features = extractor(waveform, sampling_rate=sr, return_tensors="np").input_features
        assert features.shape == (1,80,3000) and np.isfinite(features).all()
        assert features.dtype == np.float32
        data = features[0]
        label = f"Training example: {alias} · {duration:.3f} s"
        fig,axes = plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
        axes[0].plot(np.arange(len(waveform))/sr,waveform,linewidth=.45,color="#376fa3")
        axes[0].axvspan(duration,30,color="#ddd",alpha=.3,label="Zero-padding interval")
        axes[0].axvline(duration,color="#c44e52",linestyle="--")
        axes[0].set(xlim=(0,30),xlabel="Time (seconds)",ylabel="Waveform amplitude",title=label)
        axes[0].legend(loc="upper right")
        im=axes[1].imshow(data,origin="lower",aspect="auto",interpolation="nearest",extent=(0,30,0,80),cmap="magma")
        axes[1].axvline(duration,color="cyan",linestyle="--",label="Real audio ends; padding follows")
        axes[1].set(xlabel="Feature time (seconds; full 30-second input)",ylabel="Mel-band index",title="WhisperFeatureExtractor input · 80 × 3000 · FP32 · NumPy CPU recreation")
        axes[1].legend(loc="upper right")
        fig.colorbar(im,ax=axes[1],label="Scaled log-Mel value (not calibrated dB)")
        fig.savefig(output/f"processor_{alias}_full.png",dpi=140)
        plt.close(fig)
        real_frames=min(3000,int(np.ceil(duration/.01))+2)
        fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True)
        im=ax.imshow(data[:,:real_frames],origin="lower",aspect="auto",interpolation="nearest",extent=(0,real_frames*.01,0,80),cmap="magma",vmin=float(data.min()),vmax=float(data.max()))
        ax.axvline(duration,color="cyan",linestyle="--",label="Real audio end")
        ax.set(xlim=(0,min(30,max(duration+.15,1))),xlabel="Feature time (seconds)",ylabel="Mel-band index",title=label+" · close view of real-audio feature region")
        ax.legend(loc="upper right")
        fig.colorbar(im,ax=ax,label="Scaled log-Mel value")
        fig.savefig(output/f"processor_{alias}_zoom.png",dpi=140)
        plt.close(fig)
        aggregate["examples"].append({"example":alias,"duration_seconds":duration,"frames":len(waveform),"feature_shape":[1,80,3000],"feature_dtype":"float32","finite":True,"padding_seconds":30-duration})
    (repository_root/"docs/PHASE5B_PROCESSOR_ILLUSTRATIONS.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    print("Created four anonymous training-only processor figures; no inference performed.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root",required=True,type=Path)
    parser.add_argument("--repository-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir",required=True,type=Path)
    args=parser.parse_args()
    render(args.smoke_root,args.repository_root,args.cache_dir)
