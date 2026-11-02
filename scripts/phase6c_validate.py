#!/usr/bin/env python3
"""Phase 6C private validation audit.

Runs only on the frozen validation bundle and a private P1 checkpoint. It
does not access the test split or train a model. The script compares the
existing single-example generation path with batched generation on a
deterministic 12-clip subset, records speed/memory, and writes private
diagnostic outputs plus a reproducible F1 configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor


MODEL_ID = "neurlang/ipa-whisper-base"
MODEL_REVISION = "949f7c4e4ee374b5202bc5886f6e4552e5ae2371"
EXPECTED_VALIDATION_CLIPS = 732
EXPECTED_REFERENCE_CODEPOINTS = 64137
EOS_ID = 50257
DECODER_START_ID = 50258
MAX_LENGTH = 448


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


def extract_bundle(archive: Path, expected_type: str, destination: Path) -> tuple[Path, dict]:
    with zipfile.ZipFile(archive) as archive_file:
        marker_names = [name for name in archive_file.namelist() if name.endswith("bundle_record.json")]
        matches = []
        for name in marker_names:
            record = json.loads(archive_file.read(name))
            if record.get("bundle_type") == expected_type:
                matches.append((name, record))
        assert len(matches) == 1, f"Expected one {expected_type} marker in {archive}, found {len(matches)}"
        marker_name, record = matches[0]
        root = destination / expected_type
        root.mkdir(parents=True, exist_ok=True)
        for member in tqdm(archive_file.infolist(), desc=f"Checking {expected_type} archive", unit="file"):
            target = (root / member.filename).resolve()
            target.relative_to(root.resolve())
        archive_file.extractall(root)
        return (root / Path(marker_name).parent).resolve(), record


def verify_bundle(root: Path, record: dict) -> None:
    entries = [line.split("  ", 1) for line in (root / "SHA256SUMS.txt").read_text().splitlines() if line]
    for expected, relative in tqdm(entries, desc="Verifying bundle checksums", unit="file"):
        assert sha256(safe(root, relative)) == expected, f"Checksum failed: {relative}"
    assert record["evaluation_partition"] == "validation"
    assert record["contains_train_audio"] is False
    assert record["contains_test_audio"] is False
    assert record["clips"] == EXPECTED_VALIDATION_CLIPS


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
            substitutions += 1
            i, j = i - 1, j - 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions


def trim_sequence(sequence: list[int]) -> list[int]:
    """Remove batch padding while retaining a generated EOS when present."""
    try:
        return sequence[: sequence.index(EOS_ID) + 1]
    except ValueError:
        return sequence[:MAX_LENGTH]


def discover_zip(input_root: Path, expected_type: str) -> Path:
    """Find a private bundle ZIP by reading its bundle_record.json marker."""
    matches = []
    for archive in input_root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as archive_file:
                for name in archive_file.namelist():
                    if not name.endswith("bundle_record.json"):
                        continue
                    record = json.loads(archive_file.read(name))
                    if record.get("bundle_type") == expected_type:
                        matches.append(archive)
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
            continue
    assert len(matches) == 1, f"Expected one {expected_type} ZIP under {input_root}, found {len(matches)}"
    return matches[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-zip", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("phase6c_private"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    # Jupyter adds a -f kernel-connection argument. Ignore it when the
    # source is pasted into a notebook; the notebook wrapper passes paths
    # explicitly, and direct Kaggle execution can auto-discover the ZIPs.
    args, unknown = parser.parse_known_args(argv)
    if unknown and argv is not None:
        raise ValueError(f"Unknown arguments: {unknown}")
    assert torch.cuda.is_available(), "A CUDA GPU is required for the measured comparison."

    if args.validation_zip is None or args.checkpoint is None:
        input_root = Path("/kaggle/input")
        assert input_root.is_dir(), "Set --validation-zip and --checkpoint when running outside Kaggle."
        args.validation_zip = args.validation_zip or discover_zip(input_root, "phase5_validation_baseline")
        checkpoint_candidates = sorted(input_root.rglob("*best_checkpoint*.zip"))
        assert len(checkpoint_candidates) == 1, (
            "Set --checkpoint explicitly, or attach exactly one *best_checkpoint*.zip private input."
        )
        args.checkpoint = args.checkpoint or checkpoint_candidates[0]
    print(f"[Phase 6C] validation ZIP: {args.validation_zip}")
    print(f"[Phase 6C] checkpoint: {args.checkpoint}")
    print(f"[Phase 6C] output directory: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    extracted = args.output / "_extracted"
    if args.validation_zip.suffix == ".zip":
        validation_root, validation_record = extract_bundle(
            args.validation_zip, "phase5_validation_baseline", extracted
        )
    else:
        validation_root = args.validation_zip.resolve()
        validation_record = json.loads((validation_root / "bundle_record.json").read_text())
    verify_bundle(validation_root, validation_record)

    checkpoint_root = args.checkpoint.resolve()
    if checkpoint_root.suffix == ".zip":
        checkpoint_root = extracted / "checkpoint"
        with zipfile.ZipFile(args.checkpoint) as archive_file:
            members = archive_file.infolist()
            for member in tqdm(members, desc="Checking checkpoint archive", unit="file"):
                target = (checkpoint_root / member.filename).resolve()
                target.relative_to(checkpoint_root.resolve())
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            for member in tqdm(members, desc="Extracting checkpoint", unit="file"):
                archive_file.extract(member, checkpoint_root)
    model_dirs = [p for p in checkpoint_root.rglob("config.json") if (p.parent / "model.safetensors").is_file()]
    assert len(model_dirs) == 1, f"Expected one checkpoint model directory, found {len(model_dirs)}"
    model_dir = model_dirs[0].parent

    with (validation_root / "validation_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == EXPECTED_VALIDATION_CLIPS
    assert sum(len(row["reference"]) for row in rows) == EXPECTED_REFERENCE_CODEPOINTS

    print("[Phase 6C] Loading processor and checkpoint model...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).cuda().eval()
    # Match Phase 6B's evaluator: cache is disabled during training but
    # enabled for validation generation. This affects speed, not the policy.
    model.config.use_cache = True

    # This is the effective policy observed in Phase 6B: the checkpoint's
    # model config supplied [[1, None], [2, 50359]] when the GenerationConfig
    # field was None. Make it explicit so the audit does not silently change it.
    generation_config = GenerationConfig.from_dict(
        json.loads((validation_root / "generation_config.json").read_text())
    )
    generation_config.forced_decoder_ids = [[1, None], [2, 50359]]
    generation_config.language = None
    generation_config.task = None
    generation_config.suppress_tokens = []
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_length = MAX_LENGTH
    generation_config.begin_suppress_tokens = [220, 50257]
    generation_config.return_dict_in_generate = True
    assert generation_config.eos_token_id == EOS_ID
    assert generation_config.decoder_start_token_id == DECODER_START_ID

    def feature(row: dict) -> torch.Tensor:
        audio, sample_rate = sf.read(
            safe(validation_root, row["audio_file"]), dtype="float32", always_2d=True
        )
        assert sample_rate == 16000 and audio.shape[1] == 1
        return processor.feature_extractor(
            audio[:, 0], sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

    ordered = sorted(enumerate(rows), key=lambda item: (float(item[1]["duration_seconds"]), item[1]["sample_id"]))
    selected_indices = [ordered[round(i * (len(ordered) - 1) / 11)][0] for i in range(12)]
    selected = [rows[i] for i in selected_indices]
    features = [feature(row) for row in tqdm(selected, desc="Extracting audit features", unit="clip")]

    def generate_batch(batch_features: list[torch.Tensor], batch_rows: list[dict]) -> list[dict]:
        input_features = torch.stack(batch_features).cuda()
        with torch.inference_mode():
            output = model.generate(
                input_features=input_features,
                generation_config=copy.deepcopy(generation_config),
            )
        records = []
        for row, sequence in zip(batch_rows, output.sequences.detach().cpu().tolist()):
            ids = trim_sequence(sequence)
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
            normalized = normalize(raw)
            reference = normalize(row["reference"])
            substitutions, deletions, insertions = edit_counts(reference, normalized)
            records.append({
                "sample_id": row["sample_id"],
                "reference": reference,
                "raw_prediction": raw,
                "normalized_prediction": normalized,
                "token_ids": ids,
                "output_length": len(ids),
                "eos": bool(ids and ids[-1] == EOS_ID),
                "cap_without_eos": len(ids) >= MAX_LENGTH and (not ids or ids[-1] != EOS_ID),
                "S": substitutions,
                "D": deletions,
                "I": insertions,
                "duration_seconds": float(row["duration_seconds"]),
            })
        return records

    # Existing path: one clip per generate call.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    single = []
    for index in tqdm(range(12), desc="Single-clip generation", unit="clip"):
        single.extend(generate_batch([features[index]], [selected[index]]))
    torch.cuda.synchronize()
    single_seconds = time.perf_counter() - start
    single_memory = torch.cuda.max_memory_allocated() / 1024**2

    comparisons = []
    for batch_size in tqdm(args.batch_sizes, desc="Testing batch sizes", unit="batch size"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        batched = []
        failed = None
        try:
            for begin in tqdm(range(0, 12, batch_size), desc=f"Batch-{batch_size} generation", unit="batch", leave=False):
                batch_features = features[begin : begin + batch_size]
                batched.extend(generate_batch(batch_features, selected[begin : begin + batch_size]))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            failed = f"CUDA OOM: {exc}"
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated() / 1024**2
        exact = failed is None and all(
            left["token_ids"] == right["token_ids"]
            and left["raw_prediction"] == right["raw_prediction"]
            and left["normalized_prediction"] == right["normalized_prediction"]
            for left, right in zip(single, batched)
        )
        if exact:
            single_aggregate = {
                "S": sum(item["S"] for item in single),
                "D": sum(item["D"] for item in single),
                "I": sum(item["I"] for item in single),
            }
            batched_aggregate = {
                "S": sum(item["S"] for item in batched),
                "D": sum(item["D"] for item in batched),
                "I": sum(item["I"] for item in batched),
            }
            exact = batched_aggregate == single_aggregate
        comparisons.append({
            "batch_size": batch_size,
            "status": "pass" if exact else ("oom" if failed else "mismatch"),
            "runtime_seconds": seconds,
            "peak_allocated_mib": memory,
            "speedup_vs_single": single_seconds / seconds if seconds else None,
            "error": failed,
            "same_tokens_text_and_normalization": exact,
        })
        print(
            f"[Phase 6C] batch={batch_size} status={comparisons[-1]['status']} "
            f"seconds={seconds:.2f} peak_allocated_mib={memory:.1f}"
        )
        if failed is not None:
            break

    passed = [item for item in comparisons if item["status"] == "pass"]
    assert passed, f"No batched size passed exact equivalence: {comparisons}"
    chosen = max(passed, key=lambda item: item["batch_size"])
    assert all(item["same_tokens_text_and_normalization"] for item in passed)

    def aggregate(records: list[dict]) -> dict:
        return {
            "clips": len(records),
            "reference_codepoints": sum(len(item["reference"]) for item in records),
            "S": sum(item["S"] for item in records),
            "D": sum(item["D"] for item in records),
            "I": sum(item["I"] for item in records),
            "eos": sum(item["eos"] for item in records),
            "cap_without_eos": sum(item["cap_without_eos"] for item in records),
            "corpus_cer": sum(item["S"] + item["D"] + item["I"] for item in records)
            / sum(len(item["reference"]) for item in records),
        }

    diagnostics = []
    for item in single:
        cer = (item["S"] + item["D"] + item["I"]) / max(1, len(item["reference"]))
        if item["cap_without_eos"]:
            pattern = "cap_without_eos"
        elif "�" in item["normalized_prediction"]:
            pattern = "replacement_character"
        elif item["I"] >= 2 * max(1, len(item["reference"])):
            pattern = "high_insertion"
        elif item["duration_seconds"] < 5 and cer > 1:
            pattern = "short_high_cer"
        else:
            pattern = "normal_eos_or_other"
        item = dict(item)
        item["cer"] = cer
        item["observed_pattern"] = pattern
        diagnostics.append(item)

    f1_config = {
        "experiment": "F1",
        "status": "prepared_not_run",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "train_clips": 3629,
        "train_hours": 10.851637,
        "validation_clips": 732,
        "test_clips": 683,
        "test_data_used": False,
        "micro_batch": None,
        "gradient_accumulation": None,
        "effective_batch": 8,
        "max_epochs": 3,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "precision": "FP16 AMP training / FP32 validation",
        "validation_batch_size": chosen["batch_size"],
        "validation_policy": generation_config.to_dict(),
        "select_by": "minimum validation corpus NFD code-point CER",
        "start_from": "original pinned pretrained checkpoint; do not warm-start P1",
    }
    output = {
        "status": "PASS",
        "single": {"runtime_seconds": single_seconds, "peak_allocated_mib": single_memory, "aggregate": aggregate(single)},
        "batched": comparisons,
        "chosen_batch_size": chosen["batch_size"],
        "equivalence": {"exact_tokens_text_normalization": True, "aggregate_single": aggregate(single)},
        "generation_policy": generation_config.to_dict(),
        "f1_config": f1_config,
        "test_data_used": False,
    }
    print("[Phase 6C] Writing private audit outputs...")
    (args.output / "phase6c_equivalence.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.output / "phase6c_diagnostics_private.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in diagnostics) + "\n"
    )
    (args.output / "f1_config_prepared.json").write_text(json.dumps(f1_config, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"[Phase 6C] COMPLETE. Chosen batch size: {chosen['batch_size']}")


if __name__ == "__main__":
    main()
