"""Create the private Phase 7 final-test bundle from the frozen split.

Run from any directory with the BTP Python 3.11 environment. This only packages
the already-frozen test partition; it does not train, tune, or run inference.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import zipfile

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "manifests/frozen_A_narration_phase3_5b_v1"
RAW_ROOT = (ROOT / "raw/A_narration").resolve()
VALIDATION_BUNDLE = ROOT / "outputs/phase5a_validation_bundle_20261101_155452_045404_9fb2cd/phase5_validation_bundle"
OUT = ROOT / "outputs/phase7_final_test_bundle_private_v1"
BUNDLE = OUT / "phase7_final_test_bundle"
AUDIO = BUNDLE / "audio"
EXPECTED_TEST_SHA = "e2f8edca8af674363b43cc79604b38c4408a7aa852484b3aec514112f576c80d"
EXPECTED_ALL_SHA = "77673030d6d7a4a864ef98c869eb9acf2295e80029862ca8a68e0489527903ac"
EXPECTED_TRAIN_SHA = "61e0e614a0622667eac14eda40249420ab163e3dc9b71c11f3bfc0d16d4234ef"
EXPECTED_VALIDATION_SHA = "9ff2622420f7e662da82371dbfae9e941c2ead9bcb7e0a47f4e6fd5639e1e3cd"


def sha256_file(path: Path) -> str:
    """Hash files in chunks so large source shards stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_group_offset(parquet: pq.ParquetFile, row_number: int) -> tuple[int, int]:
    """Map a source-shard row number to the Parquet row group and local offset."""
    start = 0
    for group_index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        if start <= row_number < start + count:
            return group_index, row_number - start
        start += count
    raise IndexError(f"Source row {row_number} is outside its Parquet shard")


def normalize_reference(text: str) -> str:
    """Apply the already-approved NFD and whitespace policy for scoring."""
    import re
    import unicodedata
    return re.sub(r" +", " ", unicodedata.normalize("NFD", text).strip())


if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing private output: {OUT}")
if not RAW_ROOT.is_dir() or not VALIDATION_BUNDLE.is_dir():
    raise FileNotFoundError("Need raw/A_narration and the verified Phase 5A validation bundle.")

# Verify the frozen records and ensure the three partitions still match the accepted hashes.
freeze_record = json.loads((FROZEN_DIR / "freeze_record.json").read_text())
assert freeze_record["status"] == "frozen" and freeze_record["split_version"] == "phase3_5b_v1"
recorded = freeze_record["manifest_sha256"]
assert recorded["test.parquet"] == EXPECTED_TEST_SHA
assert recorded["frozen_all.parquet"] == EXPECTED_ALL_SHA
assert recorded["train.parquet"] == EXPECTED_TRAIN_SHA
assert recorded["validation.parquet"] == EXPECTED_VALIDATION_SHA
for filename, digest in recorded.items():
    assert sha256_file(FROZEN_DIR / filename) == digest, f"Frozen hash mismatch: {filename}"

test = pd.read_parquet(FROZEN_DIR / "test.parquet").sort_values("sample_id", kind="stable").reset_index(drop=True)
frozen_all = pd.read_parquet(FROZEN_DIR / "frozen_all.parquet")
train_ids = set(pd.read_parquet(FROZEN_DIR / "train.parquet", columns=["sample_id"])["sample_id"])
validation_ids = set(pd.read_parquet(FROZEN_DIR / "validation.parquet", columns=["sample_id"])["sample_id"])
test_ids = set(test["sample_id"])
assert len(test) == 683 and len(test_ids) == 683
assert test["frozen_split"].eq("test").all()
assert not (test_ids & train_ids or test_ids & validation_ids or train_ids & validation_ids)
assert train_ids | validation_ids | test_ids == set(frozen_all["sample_id"])
assert test["transcript_phase2"].notna().all() and test["transcript_raw"].notna().all()
assert test["sample_id"].str.fullmatch(r"A_[0-9]{6}").all()
assert math.isclose(test["duration_exact"].sum() / 3600, 1.8910574826388888, abs_tol=1e-9)

# Carry the validated generation configuration and scoring policy forward unchanged.
generation_config = json.loads((VALIDATION_BUNDLE / "generation_config.json").read_text())
protocol = json.loads((VALIDATION_BUNDLE / "baseline_protocol.json").read_text())
protocol["evaluation_partition"] = "test"
protocol["experiment_id"] = "FINAL_TEST"
protocol["test_reserved_for_final_evaluation"] = True
protocol["no_training_or_tuning"] = True
for setting in ("do_sample", "num_beams", "max_length", "max_new_tokens", "forced_decoder_ids", "suppress_tokens", "begin_suppress_tokens", "language", "task"):
    assert setting in generation_config, f"Missing frozen generation setting: {setting}"

# Resolve only source shards referenced by frozen test rows; never include train/validation audio.
source_plan: dict[Path, list[tuple[object, int]]] = defaultdict(list)
for row in test.itertuples(index=False):
    relative_text, row_text = row.audio_path.rsplit("#row=", 1)
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts or not row_text.isdigit():
        raise ValueError("Unsafe or malformed frozen source locator")
    source = (ROOT / relative).resolve()
    if not source.is_relative_to(RAW_ROOT) or not source.is_file():
        raise FileNotFoundError(source)
    source_plan[source].append((row, int(row_text)))
assert len({row.sample_id for items in source_plan.values() for row, _ in items}) == 683

OUT.mkdir(parents=True)
AUDIO.mkdir(parents=True)
alias = {component: f"test_group_{i:02d}" for i, component in enumerate(sorted(test.component_id.unique()), 1)}
manifest_rows = []
source_hashes = {}

# Extract original WAV bytes and verify format, duration, frame count and frozen waveform hash.
for shard_number, source in enumerate(sorted(source_plan), 1):
    source_hashes[str(source.relative_to(ROOT))] = sha256_file(source)
    parquet = pq.ParquetFile(source)
    grouped: dict[int, list[tuple[object, int]]] = defaultdict(list)
    for row, row_number in source_plan[source]:
        group, offset = row_group_offset(parquet, row_number)
        grouped[group].append((row, offset))
    for group, entries in sorted(grouped.items()):
        audio_column = parquet.read_row_group(group, columns=["audio"]).column("audio")
        for row, offset in entries:
            item = audio_column[offset].as_py()
            audio_bytes = item.get("bytes") if isinstance(item, dict) else None
            if not audio_bytes:
                raise ValueError(f"Audio missing for frozen test row {row.sample_id}")
            info = sf.info(BytesIO(audio_bytes))
            if (info.format, info.subtype, info.samplerate, info.channels) != ("WAV", "PCM_16", 16000, 1):
                raise ValueError(f"Unexpected audio format for {row.sample_id}")
            samples, rate = sf.read(BytesIO(audio_bytes), dtype="int16", always_2d=True)
            if rate != 16000 or samples.shape != (int(row.decoded_frames), 1):
                raise ValueError(f"Decoded frame/rate mismatch for {row.sample_id}")
            duration = len(samples) / rate
            if not math.isclose(duration, float(row.duration_exact), abs_tol=1e-9):
                raise ValueError(f"Duration mismatch for {row.sample_id}")
            waveform_hash = hashlib.sha256(rate.to_bytes(4, "little") + samples.tobytes(order="C")).hexdigest()
            if waveform_hash != row.exact_waveform_sha256:
                raise ValueError(f"Frozen waveform hash mismatch for {row.sample_id}")
            audio_rel = Path("audio") / f"{row.sample_id}.wav"
            destination = BUNDLE / audio_rel
            destination.write_bytes(audio_bytes)
            raw_reference = row.transcript_raw
            reference = row.transcript_phase2
            if normalize_reference(raw_reference) != reference:
                raise ValueError(f"Approved normalization mismatch for {row.sample_id}")
            manifest_rows.append({
                "sample_id": row.sample_id, "split": "test",
                "component_id": alias[row.component_id], "audio_file": audio_rel.as_posix(),
                "reference_raw": raw_reference, "reference": reference,
                "reference_codepoints_nfd": len(reference), "duration_seconds": duration,
                "sample_rate": rate, "channels": 1, "frames": len(samples),
                "audio_sha256": sha256_bytes(audio_bytes), "waveform_sha256": waveform_hash,
            })
    print(f"Verified source shards {shard_number}/{len(source_plan)}; test clips {len(manifest_rows)}/683")

assert len(manifest_rows) == 683 and {r["sample_id"] for r in manifest_rows} == test_ids
manifest = pd.DataFrame(manifest_rows).sort_values("sample_id", kind="stable").reset_index(drop=True)
manifest_path = BUNDLE / "test_manifest.csv"
manifest.to_csv(manifest_path, index=False, encoding="utf-8")
(BUNDLE / "generation_config.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2) + "\n")
(BUNDLE / "evaluation_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
record = {
    "bundle_type": "phase7_final_test_v1", "split_version": "phase3_5b_v1",
    "evaluation_partition": "test", "clips": 683,
    "hours": float(manifest.duration_seconds.sum() / 3600),
    "reference_codepoints": int(manifest.reference_codepoints_nfd.sum()),
    "components": int(manifest.component_id.nunique()),
    "frozen_test_manifest_sha256": EXPECTED_TEST_SHA,
    "frozen_all_manifest_sha256": EXPECTED_ALL_SHA,
    "manifest_sha256": sha256_file(manifest_path),
    "generation_config_sha256": sha256_file(BUNDLE / "generation_config.json"),
    "evaluation_protocol_sha256": sha256_file(BUNDLE / "evaluation_protocol.json"),
    "source_audio_shards": len(source_hashes), "source_audio_shards_sha256": source_hashes,
    "contains_train_audio": False, "contains_validation_audio": False,
    "contains_references": True, "private": True,
    "no_training_or_tuning": True, "audio_transforms": "none; original verified PCM16 WAV bytes copied",
    "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
}
(BUNDLE / "bundle_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
(BUNDLE / "README_PRIVATE.md").write_text(
    "# Private final test bundle\n\n"
    "Contains only the 683 frozen Phase 3.5B test clips and references. Keep private.\n"
    "Use only after model and decoding selection is frozen. Never use for training or tuning.\n"
    "Audio is original WAV bytes; feature extraction and scoring follow the attached frozen protocol.\n",
    encoding="utf-8",
)

# Hash every bundle member and ZIP the private evaluation inputs.
payload = sorted(p for p in BUNDLE.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
checksums = {p.relative_to(BUNDLE).as_posix(): sha256_file(p) for p in payload}
(BUNDLE / "SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
zip_path = OUT / "phase7_final_test_bundle_private.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(BUNDLE).as_posix())
with zipfile.ZipFile(zip_path) as archive:
    assert archive.testzip() is None
    assert len([name for name in archive.namelist() if name.startswith("audio/") and name.endswith(".wav")]) == 683

(OUT / "source_shard_hashes_private.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
(OUT / "test_bundle_summary_private.json").write_text(json.dumps({
    "status": "passed", "clips": 683, "hours": record["hours"],
    "reference_codepoints": record["reference_codepoints"], "components": record["components"],
    "zip_path": str(zip_path), "zip_size_bytes": zip_path.stat().st_size,
    "zip_sha256": sha256_file(zip_path), "verified_wavs": 683,
    "frozen_test_manifest_sha256": EXPECTED_TEST_SHA,
    "train_or_validation_rows_included": False,
    "model_inference_or_training_performed": False,
}, indent=2) + "\n")
print({"status": "PASSED", "clips": 683, "hours": round(record["hours"], 6),
       "zip": str(zip_path), "size_mib": round(zip_path.stat().st_size / 1024**2, 1),
       "sha256": sha256_file(zip_path)})
