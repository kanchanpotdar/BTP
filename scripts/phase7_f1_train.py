#!/usr/bin/env python3
"""Phase F1 full fine-tuning on the frozen Nyishi Narration split.

This is a private Kaggle runner. It starts from the pinned pretrained model,
uses lazy CPU feature extraction (no multi-gigabyte feature cache), trains with
FP16 AMP and gradient accumulation, validates with the Phase 6C-equivalent
batch-8 greedy policy, and saves resumable epoch checkpoints. It never reads
the test split.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)


MODEL_ID = "neurlang/ipa-whisper-base"
MODEL_REVISION = "949f7c4e4ee374b5202bc5886f6e4552e5ae2371"
EOS_ID = 50257
MAX_LENGTH = 448
EXPECTED_TRAIN = 3629
EXPECTED_VALIDATION = 732
EXPECTED_REFERENCE_CODEPOINTS = 64137


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def extract_bundle(archive: Path, bundle_type: str, destination: Path) -> tuple[Path, dict]:
    if archive.is_dir():
        marker = archive / "bundle_record.json"
        return archive, json.loads(marker.read_text())
    with zipfile.ZipFile(archive) as zf:
        markers = [n for n in zf.namelist() if n.endswith("bundle_record.json")]
        matches = [(n, json.loads(zf.read(n))) for n in markers]
        matches = [(n, r) for n, r in matches if r.get("bundle_type") == bundle_type]
        assert len(matches) == 1, f"Expected one {bundle_type} marker in {archive}"
        marker_name, record = matches[0]
        root = destination / bundle_type
        root.mkdir(parents=True, exist_ok=True)
        for member in tqdm(zf.infolist(), desc=f"Extracting {bundle_type}", unit="file"):
            target = (root / member.filename).resolve()
            target.relative_to(root.resolve())
        for member in tqdm(zf.infolist(), desc=f"Writing {bundle_type}", unit="file"):
            zf.extract(member, root)
        return (root / Path(marker_name).parent).resolve(), record


def discover_bundle(root: Path, bundle_type: str) -> Path:
    candidates = []
    for marker in root.rglob("bundle_record.json"):
        try:
            if json.loads(marker.read_text()).get("bundle_type") == bundle_type:
                candidates.append(marker.parent)
        except (OSError, json.JSONDecodeError):
            pass
    for archive in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    if name.endswith("bundle_record.json") and json.loads(zf.read(name)).get("bundle_type") == bundle_type:
                        candidates.append(archive)
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
            pass
    candidates = sorted(set(candidates))
    assert len(candidates) == 1, f"Expected one {bundle_type} input, found {len(candidates)}: {candidates}"
    return candidates[0]


def verify_bundle(root: Path, record: dict, expected_partition: str) -> None:
    sums = root / "SHA256SUMS.txt"
    if sums.exists():
        entries = [line.split("  ", 1) for line in sums.read_text().splitlines() if line]
        for expected, relative in tqdm(entries, desc="Verifying bundle checksums", unit="file"):
            assert sha256(safe(root, relative)) == expected, f"Checksum failed: {relative}"
    # Validation bundles declare evaluation_partition/contains_test_audio;
    # the full F1 training bundle declares contains_validation_or_test_audio.
    # Accept either explicit schema without weakening the split checks.
    if "evaluation_partition" in record:
        assert record["evaluation_partition"] == expected_partition
    if "contains_test_audio" in record:
        assert record["contains_test_audio"] is False
    if "contains_validation_or_test_audio" in record:
        assert record["contains_validation_or_test_audio"] is False


def normalize(value: str) -> str:
    return re.sub(r" {2,}", " ", unicodedata.normalize("NFD", str(value)).strip())


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    ref, hyp = list(reference), list(hypothesis)
    distance = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        distance[i][0] = i
    for j in range(len(hyp) + 1):
        distance[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    substitutions = deletions = insertions = 0
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and distance[i][j] == distance[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1; i, j = i - 1, j - 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1; i -= 1
        else:
            insertions += 1; j -= 1
    return substitutions, deletions, insertions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/phase7_F1_private"))
    parser.add_argument("--resume", type=Path, default=None, help="epoch_N directory from this exact F1 run")
    parser.add_argument("--physical-batch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--validation-batch", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args, unknown = parser.parse_known_args(argv)
    if argv is not None and unknown:
        raise ValueError(f"Unknown arguments: {unknown}")
    assert torch.cuda.is_available(), "A CUDA GPU is required."
    assert args.physical_batch * args.accumulation == 8
    assert args.validation_batch == 8
    input_root = Path("/kaggle/input")
    if args.train_input is None:
        args.train_input = discover_bundle(input_root, "phase7_f1_train_v1")
    if args.validation_input is None:
        args.validation_input = discover_bundle(input_root, "phase5_validation_baseline")
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = args.output / "_extracted_inputs"
    train_root, train_record = extract_bundle(args.train_input, "phase7_f1_train_v1", extracted)
    val_root, val_record = extract_bundle(args.validation_input, "phase5_validation_baseline", extracted)
    verify_bundle(train_root, train_record, "train")
    verify_bundle(val_root, val_record, "validation")
    assert train_record["clips"] == EXPECTED_TRAIN and val_record["clips"] == EXPECTED_VALIDATION
    assert train_record.get("contains_validation_or_test_audio") is False

    with (train_root / "train_manifest.csv").open(newline="", encoding="utf-8") as stream:
        train_rows = list(csv.DictReader(stream))
    with (val_root / "validation_manifest.csv").open(newline="", encoding="utf-8") as stream:
        val_rows = list(csv.DictReader(stream))
    assert len(train_rows) == EXPECTED_TRAIN and len(val_rows) == EXPECTED_VALIDATION
    assert sum(len(normalize(row["reference"])) for row in val_rows) == EXPECTED_REFERENCE_CODEPOINTS
    assert {row["sample_id"] for row in train_rows}.isdisjoint({row["sample_id"] for row in val_rows})

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    generation_config = GenerationConfig.from_dict(json.loads((val_root / "generation_config.json").read_text()))
    # Phase 6C proved this effective runtime policy is equivalent to single-example decoding.
    generation_config.forced_decoder_ids = [[1, None], [2, 50359]]
    generation_config.suppress_tokens = []
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_length = MAX_LENGTH
    generation_config.begin_suppress_tokens = [220, EOS_ID]
    generation_config.return_dict_in_generate = True
    assert generation_config.eos_token_id == EOS_ID and generation_config.max_length == MAX_LENGTH

    def read_audio(root: Path, row: dict) -> np.ndarray:
        path = safe(root, row["audio_file"])
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        assert sample_rate == 16000 and audio.shape[1] == 1
        assert 0.5 <= len(audio) / sample_rate < 30
        return audio[:, 0]

    def make_features(root: Path, rows: list[dict]) -> torch.Tensor:
        audios = [read_audio(root, row) for row in rows]
        # Force Whisper's fixed 30-second window for batched inputs. Plain
        # padding=True pads only to the longest item in this batch and can
        # return a time axis different from the required 3,000 frames.
        features = processor.feature_extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=480000,
        ).input_features
        if features.ndim == 2:
            features = features.unsqueeze(0)
        assert features.ndim == 3 and features.shape[1] == 80, (
            f"Unexpected Whisper feature shape before window normalization: {tuple(features.shape)}"
        )
        # Defensive normalization for minor Transformers/feature-extractor
        # patch differences: the model contract is always 80 x 3000.
        if features.shape[-1] > 3000:
            features = features[..., :3000]
        elif features.shape[-1] < 3000:
            features = F.pad(features, (0, 3000 - features.shape[-1]))
        assert tuple(features.shape[1:]) == (80, 3000)
        assert torch.isfinite(features).all().item(), "Non-finite Whisper features"
        return features.to(dtype=torch.float32)

    def make_labels(rows: list[dict]) -> torch.Tensor:
        # Use WhisperTokenizerFast.__call__ for batched tokenization. This
        # avoids the fast-tokenizer warning caused by encode-then-pad.
        references = [normalize(row["reference"]) for row in rows]
        encoded = processor.tokenizer(
            references,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        token_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        # Add one column so every example retains an explicit EOS token.
        input_ids = F.pad(encoded["input_ids"], (0, 1), value=processor.tokenizer.pad_token_id)
        attention_mask = F.pad(encoded["attention_mask"], (0, 1), value=0)
        for row_index, (reference, token_length) in enumerate(zip(references, token_lengths)):
            ids = encoded["input_ids"][row_index, :token_length].tolist()
            assert processor.tokenizer.decode(ids, skip_special_tokens=False) == reference
            assert token_length + 1 < MAX_LENGTH
            input_ids[row_index, token_length] = EOS_ID
            attention_mask[row_index, token_length] = 1
        labels = input_ids.masked_fill(attention_mask.ne(1), -100)
        for row_index, token_length in enumerate(token_lengths):
            assert labels[row_index, token_length].item() == EOS_ID
        return labels

    device = torch.device("cuda:0")
    model_source = args.resume / "model" if args.resume else MODEL_ID
    model_kwargs = {"dtype": torch.float32, "attn_implementation": "sdpa"}
    if not args.resume:
        model_kwargs["revision"] = MODEL_REVISION
    model = WhisperForConditionalGeneration.from_pretrained(model_source, **model_kwargs).to(device).train()
    model.config.use_cache = False
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith("bias") or "layer_norm" in name.lower() else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate,
    )
    micro_batches = math.ceil(len(train_rows) / args.physical_batch)
    steps_per_epoch = math.ceil(micro_batches / args.accumulation)
    total_steps = steps_per_epoch * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, math.ceil(0.05 * total_steps)), total_steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    fingerprint = hashlib.sha256(json.dumps({
        "model": MODEL_ID, "revision": MODEL_REVISION, "train": train_record.get("manifest_sha256"),
        "validation": val_record.get("manifest_sha256"), "seed": args.seed,
        "physical_batch": args.physical_batch, "accumulation": args.accumulation,
        "validation_batch": args.validation_batch, "max_epochs": args.max_epochs,
        "patience": args.patience, "learning_rate": args.learning_rate,
        "generation": generation_config.to_dict(),
    }, sort_keys=True, default=str).encode()).hexdigest()
    start_epoch = 0; completed_steps = 0; history = []; best_cer = float("inf"); best_epoch = None; stale_epochs = 0
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["fingerprint"] == fingerprint and state["epoch"] < args.max_epochs
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor): item[key] = value.to(device)
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"]); torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch = state["epoch"]; completed_steps = state["completed_steps"]; history = state["history"]; best_cer = state["best_cer"]; best_epoch = state["best_epoch"]; stale_epochs = state.get("stale_epochs", 0)
    print({"train_clips": len(train_rows), "validation_clips": len(val_rows), "steps_per_epoch": steps_per_epoch, "max_epochs": args.max_epochs, "patience": args.patience, "fingerprint": fingerprint})

    def evaluate(epoch: int) -> tuple[dict, list[dict]]:
        model.eval(); model.config.use_cache = True; records = []; started = time.perf_counter()
        for begin in tqdm(range(0, len(val_rows), args.validation_batch), desc=f"Epoch {epoch} validation", unit="batch"):
            rows = val_rows[begin:begin + args.validation_batch]
            features = make_features(val_root, rows).to(device)
            with torch.inference_mode():
                generated = model.generate(input_features=features, generation_config=copy.deepcopy(generation_config))
            for row, sequence in zip(rows, generated.sequences.detach().cpu().tolist()):
                ids = sequence[:sequence.index(EOS_ID) + 1] if EOS_ID in sequence else sequence[:MAX_LENGTH]
                raw = processor.tokenizer.decode(ids, skip_special_tokens=True); hypothesis = normalize(raw); reference = normalize(row["reference"])
                substitutions, deletions, insertions = edit_counts(reference, hypothesis)
                records.append({"sample_id": row["sample_id"], "reference": reference, "raw_prediction": raw, "normalized_prediction": hypothesis, "token_ids": ids, "eos": bool(ids and ids[-1] == EOS_ID), "cap_without_eos": len(ids) >= MAX_LENGTH and (not ids or ids[-1] != EOS_ID), "S": substitutions, "D": deletions, "I": insertions, "duration_seconds": float(row["duration_seconds"])})
            del features, generated
        model.train(); model.config.use_cache = False; gc.collect(); torch.cuda.empty_cache()
        reference_codepoints = sum(len(r["reference"]) for r in records); S = sum(r["S"] for r in records); D = sum(r["D"] for r in records); I = sum(r["I"] for r in records)
        return {"clips": len(records), "reference_codepoints": reference_codepoints, "S": S, "D": D, "I": I, "edit_count": S + D + I, "corpus_cer": (S + D + I) / reference_codepoints, "eos_count": sum(r["eos"] for r in records), "cap_without_eos": sum(r["cap_without_eos"] for r in records), "replacement_character_outputs": sum("�" in r["normalized_prediction"] for r in records), "elapsed_seconds": time.perf_counter() - started}, records

    run_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        model.train(); model.config.use_cache = False; indices = list(range(len(train_rows))); random.shuffle(indices); groups = [indices[i:i + args.physical_batch] for i in range(0, len(indices), args.physical_batch)]; losses = []; epoch_updates = 0; amp_retries = 0; epoch_started = time.perf_counter()
        for begin in tqdm(range(0, len(groups), args.accumulation), desc=f"Epoch {epoch} training", unit="update"):
            accumulation_groups = groups[begin:begin + args.accumulation]
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True); finite = True; group_losses = []
                try:
                    for group in accumulation_groups:
                        rows = [train_rows[i] for i in group]; features = make_features(train_root, rows).to(device); labels = make_labels(rows).to(device)
                        with torch.autocast(device_type="cuda", dtype=torch.float16): loss = model(input_features=features, labels=labels).loss
                        if not torch.isfinite(loss).item(): raise FloatingPointError("nonfinite forward loss")
                        group_losses.append(float(loss.detach())); scaler.scale(loss / len(accumulation_groups)).backward(); del features, labels
                    scaler.unscale_(optimizer); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False); assert float(grad_norm) > 0
                    old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() < old_scale: amp_retries += 1; continue
                    scheduler.step(); completed_steps += 1; epoch_updates += 1; losses.extend(group_losses); break
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache(); raise RuntimeError("CUDA OOM. Restart with --physical-batch 1 --accumulation 8; do not change the effective batch size.") from exc
            else: raise RuntimeError("No finite AMP update after eight attempts.")
        train_memory = torch.cuda.max_memory_allocated(device) / 1024**2
        validation, predictions = evaluate(epoch)
        improved = validation["corpus_cer"] < best_cer
        if improved: best_cer = validation["corpus_cer"]; best_epoch = epoch; stale_epochs = 0
        else: stale_epochs += 1
        record = {"epoch": epoch, "updates": epoch_updates, "cumulative_updates": completed_steps, "training_loss": sum(losses) / len(losses), "amp_retries": amp_retries, "train_peak_allocated_mib": train_memory, "epoch_seconds": time.perf_counter() - epoch_started, "validation": validation}
        history.append(record); checkpoint = args.output / f"epoch_{epoch:02d}"; checkpoint.mkdir(exist_ok=True); model.save_pretrained(checkpoint / "model", safe_serialization=True); torch.save({"fingerprint": fingerprint, "epoch": epoch, "completed_steps": completed_steps, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "history": history, "best_cer": best_cer, "best_epoch": best_epoch, "stale_epochs": stale_epochs}, checkpoint / "training_state.pt")
        (args.output / f"validation_epoch_{epoch:02d}_private.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + "\n")
        (args.output / "aggregate_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print({"epoch": epoch, "validation_cer_percent": 100 * validation["corpus_cer"], "best_epoch": best_epoch, "stale_epochs": stale_epochs, "train_peak_allocated_mib": train_memory, "validation_seconds": validation["elapsed_seconds"]})
        if stale_epochs >= args.patience: print(f"Early stopping after {args.patience} non-improving validation epochs."); break
    report = {"experiment": "F1", "status": "complete", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "train_clips": len(train_rows), "train_hours": train_record["hours"], "validation_clips": len(val_rows), "validation_hours": val_record["hours"], "test_data_used": False, "seed": args.seed, "physical_batch": args.physical_batch, "accumulation": args.accumulation, "effective_batch": 8, "validation_batch": args.validation_batch, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "max_epochs": args.max_epochs, "patience": args.patience, "best_epoch": best_epoch, "best_validation_cer": best_cer, "completed_steps": completed_steps, "total_wall_seconds": time.perf_counter() - run_started, "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2, "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2, "history": history}
    (args.output / "F1_report_private.json").write_text(json.dumps(report, indent=2) + "\n"); print("F1 complete; best epoch", best_epoch, "CER", 100 * best_cer)


if __name__ == "__main__":
    main()
