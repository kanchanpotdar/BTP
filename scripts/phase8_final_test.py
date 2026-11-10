"""Final sealed Nyishi test: declared B0 and selected F1 only.

This script is embedded in the companion Kaggle notebook. Inputs are private;
never publish the prediction JSONL files or checkpoint.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "neurlang/ipa-whisper-base"
REVISION = "949f7c4e4ee374b5202bc5886f6e4552e5ae2371"
EOS = 50257
CAP = 448
TEST_SHA = "e2f8edca8af674363b43cc79604b38c4408a7aa852484b3aec514112f576c80d"
N = 683
REFERENCE_CHARS = 65577
WORK = Path("/kaggle/working/phase8_final_test_private")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r" {2,}", " ", unicodedata.normalize("NFD", text).strip())


def edits(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Use the Phase 7 corpus-CER edit tie order: match, S, D, then I."""
    ref, hyp = list(reference), list(hypothesis)
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1]+(ref[i-1] != hyp[j-1]))
    s = deletion = insertion = 0
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and ref[i-1] == hyp[j-1] and d[i][j] == d[i-1][j-1]: i -= 1; j -= 1
        elif i and j and d[i][j] == d[i-1][j-1]+1: s += 1; i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j]+1: deletion += 1; i -= 1
        else: insertion += 1; j -= 1
    assert s + deletion + insertion == d[-1][-1]
    return s, deletion, insertion


def find_input(mounted: Path = Path("/kaggle/input")) -> tuple[Path, Path, dict]:
    """Find the frozen bundle as a directory or ZIP, plus the selected F1 model."""
    bundles = []
    for marker in mounted.rglob("bundle_record.json"):
        try:
            record = json.loads(marker.read_text())
            if record.get("bundle_type") == "phase7_final_test_v1": bundles.append((marker.parent, record))
        except (OSError, json.JSONDecodeError): pass
    # Kaggle can mount an uploaded ZIP without expanding it. Accept that form
    # too, but never silently choose between duplicate or conflicting bundles.
    test_archives = list(mounted.rglob("phase7_final_test_bundle_private.zip"))
    assert len(bundles) <= 1 and len(test_archives) <= 1, (
        f"Attach one final-test bundle; found {len(bundles)} directories "
        f"and {len(test_archives)} ZIPs."
    )
    assert bundles or test_archives, "No final-test bundle found under /kaggle/input."
    if not bundles:
        destination = WORK / "test_bundle"
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / "bundle_record.json"
        if not marker.is_file():
            with zipfile.ZipFile(test_archives[0]) as archive:
                assert archive.testzip() is None, "The final-test ZIP is corrupt."
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    target.relative_to(destination.resolve())
                    assert ((member.external_attr >> 16) & 0o170000) != 0o120000, (
                        f"Symlink in test ZIP: {member.filename}"
                    )
                for member in tqdm(archive.infolist(), desc="Extracting test bundle", unit="file"):
                    archive.extract(member, destination)
        assert marker.is_file(), "The test ZIP has no bundle_record.json at its root."
        bundles = [(destination, json.loads(marker.read_text()))]
    assert len(bundles) == 1 and bundles[0][1].get("bundle_type") == "phase7_final_test_v1"
    archives = list(mounted.rglob("phase7_F1_LR_2em5_best_checkpoint_private.zip"))
    # Kaggle may unpack a checkpoint dataset before mounting it. In that case
    # the model/ directory is usable directly and needs no 831 MiB ZIP.
    extracted_models = [p.parent for p in mounted.rglob("model.safetensors")
                        if p.parent.name == "model"
                        and p.parent.parent.name == "phase7_F1_LR_2e-5_best_checkpoint"
                        and (p.parent / "config.json").is_file()
                        and (p.parent / "generation_config.json").is_file()
                        and (p.parent.parent / "training_state.pt").is_file()]
    model_sources = extracted_models + archives
    reports = list(mounted.rglob("F1_report_private.json"))
    reports = [p for p in reports if json.loads(p.read_text()).get("learning_rate") == 2e-5]
    assert len(model_sources) == len(reports) == 1, (
        "Attach one selected 2e-5 Epoch 5 checkpoint (extracted model folder "
        "or ZIP) and its F1 report. Found "
        f"{len(extracted_models)} extracted models, {len(archives)} ZIPs, "
        f"and {len(reports)} matching reports."
    )
    report = json.loads(reports[0].read_text())
    assert report["best_epoch"] == 5 and math.isclose(report["best_validation_cer"], .47239502939021155, abs_tol=1e-12)
    assert report["model_revision"] == REVISION and report["repetition_penalty"] == 1.08
    return bundles[0][0], model_sources[0], report


def verify_test(root: Path) -> tuple[list[dict], dict]:
    """Verify every member hash and the frozen 683-row test contract."""
    record = json.loads((root / "bundle_record.json").read_text())
    assert record["bundle_type"] == "phase7_final_test_v1" and record["evaluation_partition"] == "test"
    assert record["frozen_test_manifest_sha256"] == TEST_SHA
    assert record["clips"] == N and record["reference_codepoints"] == REFERENCE_CHARS
    assert record["contains_train_audio"] is False and record["contains_validation_audio"] is False
    for line in tqdm((root / "SHA256SUMS.txt").read_text().splitlines(), desc="Verifying test bundle", unit="file"):
        expected, relative = line.split(maxsplit=1)
        path = (root / relative).resolve(); path.relative_to(root.resolve())
        assert sha(path) == expected, f"Checksum failed: {relative}"
    with (root / "test_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == N and len({r["sample_id"] for r in rows}) == N
    assert all(r["split"] == "test" for r in rows)
    assert sum(len(r["reference"]) for r in rows) == REFERENCE_CHARS
    for row in rows:
        assert normalize(row["reference_raw"]) == row["reference"]
        audio = (root / row["audio_file"]).resolve(); audio.relative_to(root.resolve())
        assert row["audio_file"].startswith("audio/") and sha(audio) == row["audio_sha256"]
    return rows, record


def checkpoint_model(source: Path) -> Path:
    """Use the mounted model directory or extract just the model from its ZIP."""
    if source.is_dir():
        assert source.name == "model" and source.parent.name == "phase7_F1_LR_2e-5_best_checkpoint"
        assert (source / "model.safetensors").stat().st_size > 100_000_000
        return source
    destination = WORK / "selected_f1_checkpoint"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve(); target.relative_to(destination.resolve())
        for member in tqdm(z.infolist(), desc="Extracting selected checkpoint", unit="file"):
            if "/model/" in member.filename: z.extract(member, destination)
    dirs = [p.parent for p in destination.rglob("model.safetensors") if (p.parent / "config.json").is_file()]
    assert len(dirs) == 1, f"Expected exactly one checkpoint model, found {len(dirs)}"
    return dirs[0]


def make_features(root: Path, batch: list[dict], processor: WhisperProcessor, mask: bool):
    """Use the same fixed 30-second Whisper feature window as F1 validation."""
    audios = []
    for row in batch:
        path = (root / row["audio_file"]).resolve(); path.relative_to(root.resolve())
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 16000 and audio.shape[1] == 1 and 0.5 <= len(audio) / rate < 30
        audios.append(audio[:, 0])
    packed = processor.feature_extractor(audios, sampling_rate=16000,
        return_tensors="pt", return_attention_mask=mask, padding="max_length",
        truncation=True, max_length=480000)
    features = packed.input_features
    if features.ndim == 2: features = features.unsqueeze(0)
    assert features.ndim == 3 and tuple(features.shape[:2]) == (len(batch), 80)
    # Preserve the normalization used by the working F1 runner when a feature
    # extractor patch returns a frame count slightly different from 3,000.
    if features.shape[-1] > 3000: features = features[..., :3000]
    elif features.shape[-1] < 3000: features = F.pad(features, (0, 3000-features.shape[-1]))
    assert tuple(features.shape) == (len(batch), 80, 3000) and torch.isfinite(features).all().item()
    if not mask: return features.float(), None
    attention = packed.attention_mask
    if attention.ndim == 1: attention = attention.unsqueeze(0)
    if attention.shape[-1] > 3000: attention = attention[..., :3000]
    elif attention.shape[-1] < 3000: attention = F.pad(attention, (0, 3000-attention.shape[-1]))
    assert tuple(attention.shape) == (len(batch), 3000)
    return features.float(), attention.long()


def generation_policy(root: Path, experiment: str) -> GenerationConfig:
    """Keep the historical B0 policy and validated F1 policy separate."""
    config = GenerationConfig.from_dict(json.loads((root / "generation_config.json").read_text()))
    config.max_new_tokens = None; config.do_sample = False; config.num_beams = 1
    config.num_return_sequences = 1; config.max_length = CAP
    config.forced_decoder_ids = None; config.language = None
    config.suppress_tokens = []; config.begin_suppress_tokens = [220, EOS]
    config.return_timestamps = False; config.return_dict_in_generate = True
    config.output_scores = False; config.eos_token_id = EOS; config.pad_token_id = EOS
    if experiment == "B0":
        config.task = None; config.repetition_penalty = 1.0
    else:
        config.task = "transcribe"; config.repetition_penalty = 1.08
        config.no_repeat_ngram_size = 0
    assert config.decoder_start_token_id == 50258
    return config


def evaluate(name: str, model, processor, root: Path, rows: list[dict], batch_size: int):
    """Generate all test predictions once; retain private detail and public-safe totals."""
    config = generation_policy(root, name)
    model.cuda().eval(); model.config.use_cache = True
    records = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for begin in tqdm(range(0, len(rows), batch_size), desc=f"{name} sealed test", unit="batch"):
        batch = rows[begin:begin+batch_size]
        features, attention = make_features(root, batch, processor, mask=name != "B0")
        kwargs = {"input_features": features.cuda(), "generation_config": copy.deepcopy(config)}
        if attention is not None: kwargs["attention_mask"] = attention.cuda()
        with torch.inference_mode(): output = model.generate(**kwargs)
        assert len(output.sequences) == len(batch)
        for row, sequence in zip(batch, output.sequences.detach().cpu().tolist()):
            ids = sequence[:sequence.index(EOS)+1] if EOS in sequence else sequence[:CAP]
            # Match both historical evaluators' default tokenizer decode call.
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
            reference = normalize(row["reference"]); hypothesis = normalize(raw)
            S, D, I = edits(reference, hypothesis)
            records.append({"sample_id":row["sample_id"], "reference":reference,
                "raw_prediction":raw, "normalized_prediction":hypothesis,
                "token_ids":ids, "S":S, "D":D, "I":I,
                "duration_seconds":float(row["duration_seconds"]),
                "eos":bool(ids and ids[-1]==EOS),
                "cap_without_eos":bool(len(ids)>=CAP and (not ids or ids[-1]!=EOS))})
    assert len(records) == N and sum(len(x["reference"]) for x in records) == REFERENCE_CHARS
    S=sum(x["S"] for x in records);D=sum(x["D"] for x in records);I=sum(x["I"] for x in records)
    summary={"experiment":name,"clips":N,"reference_codepoints":REFERENCE_CHARS,
        "S":S,"D":D,"I":I,"edits":S+D+I,"cer_percent":100*(S+D+I)/REFERENCE_CHARS,
        "eos_count":sum(x["eos"] for x in records),
        "cap_without_eos":sum(x["cap_without_eos"] for x in records),
        "replacement_character_outputs":sum("�" in x["normalized_prediction"] for x in records),
        "runtime_seconds":time.perf_counter()-started,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "batch_size":batch_size,"generation_config":config.to_dict()}
    private_path=WORK/f"{name}_test_predictions_private.jsonl"
    with private_path.open("w",encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    (WORK/f"{name}_test_aggregate.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    print({k:summary[k] for k in ("experiment","cer_percent","S","D","I","eos_count","cap_without_eos","runtime_seconds")})
    return summary, records


def plot_and_zip(summaries: list[dict], records_by_model: dict[str,list[dict]]):
    """Draw aggregate diagnostics and produce separate safe/private ZIP files."""
    plots=WORK/"plots";plots.mkdir(exist_ok=True)
    names=[s["experiment"] for s in summaries]
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    axes[0,0].bar(names,[s["cer_percent"] for s in summaries],color=["#e76f51","#2a9d8f"])
    axes[0,0].set(ylabel="Corpus CER (%)",title="Frozen test transcription error")
    bottom=np.zeros(len(names))
    for key,color in [("S","#457b9d"),("D","#e9c46a"),("I","#e76f51")]:
        values=np.array([s[key] for s in summaries]);axes[0,1].bar(names,values,bottom=bottom,label=key,color=color);bottom+=values
    axes[0,1].set(ylabel="Character edits",title="Error composition");axes[0,1].legend()
    for name,color in zip(names,["#e76f51","#2a9d8f"]):
        frame=pd.DataFrame(records_by_model[name]);frame["bin"]=pd.cut(frame.duration_seconds,[.5,3,10,20,30],include_lowest=True)
        grouped=frame.groupby("bin",observed=True).agg(S=("S","sum"),D=("D","sum"),I=("I","sum"),chars=("reference",lambda x:sum(map(len,x))))
        cer=100*(grouped.S+grouped.D+grouped.I)/grouped.chars
        axes[1,0].plot([str(x) for x in cer.index],cer.values,marker="o",label=name,color=color)
    axes[1,0].set(xlabel="Clip duration (seconds)",ylabel="Corpus CER within bin (%)",title="Duration slices (diagnostic)");axes[1,0].legend()
    axes[1,1].bar(names,[s["cap_without_eos"] for s in summaries],color=["#e76f51","#2a9d8f"])
    axes[1,1].set(ylabel="Clips",title="Generation cap hits without EOS")
    fig.savefig(plots/"test_aggregate_dashboard.png",dpi=180,bbox_inches="tight");plt.show();plt.close(fig)
    # The aggregate ZIP has no references, IDs, audio or prediction-level output.
    aggregate=WORK/"phase8_final_test_aggregate_plots.zip"
    with zipfile.ZipFile(aggregate,"w",zipfile.ZIP_DEFLATED) as z:
        for p in [*(WORK.glob("*_test_aggregate.json")),WORK/"comparison_aggregate.json",*(plots.glob("*.png"))]:z.write(p,p.relative_to(WORK))
    private=WORK/"phase8_final_test_predictions_private.zip"
    with zipfile.ZipFile(private,"w",zipfile.ZIP_DEFLATED) as z:
        for p in WORK.glob("*_test_predictions_private.jsonl"):z.write(p,p.name)
    for p in (aggregate,private):
        with zipfile.ZipFile(p) as z: assert z.testzip() is None
        print({"zip":str(p),"size_mib":round(p.stat().st_size/2**20,1),"sha256":sha(p)})
    return aggregate,private


def main():
    assert torch.cuda.is_available(), "Enable a Kaggle GPU before running final test."
    WORK.mkdir(parents=True,exist_ok=True)
    assert not (WORK/"comparison_aggregate.json").exists(), "Final test already completed in this output folder. Inspect saved results instead of rerunning."
    root,checkpoint_source,report=find_input()
    rows,record=verify_test(root)
    processor=WhisperProcessor.from_pretrained(MODEL_ID,revision=REVISION)
    # Run the predeclared zero-shot B0 reference under its original single-clip policy.
    b0=WhisperForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,dtype=torch.float32,attn_implementation="sdpa")
    b0_summary,b0_rows=evaluate("B0",b0,processor,root,rows,1)
    del b0;torch.cuda.empty_cache()
    # Run only the validation-selected 2e-5 Epoch 5 model with penalty 1.08.
    model_dir=checkpoint_model(checkpoint_source)
    f1=WhisperForConditionalGeneration.from_pretrained(model_dir,dtype=torch.float32,attn_implementation="sdpa")
    f1_summary,f1_rows=evaluate("F1_selected",f1,processor,root,rows,8)
    del f1;torch.cuda.empty_cache()
    (WORK/"comparison_aggregate.json").write_text(json.dumps({"scope":"final sealed test",
        "selected_validation_cer_percent":100*report["best_validation_cer"],
        "selected_learning_rate":report["learning_rate"],"selected_epoch":report["best_epoch"],
        "selected_repetition_penalty":report["repetition_penalty"],
        "results":[b0_summary,f1_summary]},indent=2,default=str)+"\n")
    aggregate,private=plot_and_zip([b0_summary,f1_summary],{"B0":b0_rows,"F1_selected":f1_rows})
    return aggregate,private


if __name__ == "__main__": main()
