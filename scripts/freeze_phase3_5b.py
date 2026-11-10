"""Freeze and hash the accepted Phase 3.5B Narration split proposal.

This creates private derived manifests under manifests/ and a machine-readable
freeze record. It never edits source data or commits the resulting manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase3_5b_20261031/split_proposal.parquet"
DEST = ROOT / "manifests/frozen_A_narration_phase3_5b_v1"
SPLITS = ("train", "validation", "test")
KEYS = (
    "component_id",
    "speaker_group_id",
    "speaker_id",
    "speaker_id_raw",
    "recording_session_id",
    "exact_waveform_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Phase 3.5B proposal is missing: {SOURCE}")

    rows = pd.read_parquet(SOURCE).copy()
    required = {
        "sample_id", "component_id", "phase35b_split", "train_only_effective",
        "reference_present", "decode_succeeded", "has_audio_frames", "duration_exact",
        "transcript_raw", "transcript_phase2", *KEYS,
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing required freeze fields: {missing}")
    if len(rows) != 5044 or rows.sample_id.isna().any() or rows.sample_id.duplicated().any():
        raise ValueError("Expected 5,044 unique, non-null candidate sample IDs.")
    if rows.phase35b_split.isna().any() or not set(rows.phase35b_split).issubset(SPLITS):
        raise ValueError("Every row must have exactly one recognized split assignment.")
    if not rows.reference_present.fillna(False).all():
        raise ValueError("A frozen supervised row has no reference transcript.")
    if not rows.decode_succeeded.fillna(False).all() or not rows.has_audio_frames.fillna(False).all():
        raise ValueError("A frozen row has failed/empty audio.")
    durations = rows.duration_exact.astype(float)
    if not durations.between(0.5, 30.0, inclusive="both").all():
        raise ValueError("A frozen row is outside the selected 0.5–30 second range.")
    if rows.transcript_phase2.isna().any() or rows.transcript_phase2.astype(str).str.strip().eq("").any():
        raise ValueError("A frozen row has an empty normalized reference.")

    overlaps = {}
    for key in KEYS:
        valid = rows.loc[rows[key].notna() & rows[key].astype(str).ne("")]
        counts = valid.groupby(key, dropna=False).phase35b_split.nunique()
        overlaps[key] = int(counts.gt(1).sum())
    if any(overlaps.values()):
        raise ValueError(f"Cross-split leakage detected: {overlaps}")

    heldout_train_only = int(
        rows.loc[rows.phase35b_split.ne("train"), "train_only_effective"].fillna(False).sum()
    )
    if heldout_train_only:
        raise ValueError(f"Found {heldout_train_only} train-only rows in held-out splits.")

    rows = rows.sort_values("sample_id", kind="stable").reset_index(drop=True)
    rows["frozen_split"] = rows["phase35b_split"].astype("string")
    DEST.mkdir(parents=True, exist_ok=False)

    paths = {}
    combined = DEST / "frozen_all.parquet"
    rows.to_parquet(combined, index=False, engine="pyarrow", compression="zstd")
    paths[combined.name] = combined

    membership = rows[["sample_id", "component_id", "frozen_split"]].copy()
    member_csv = DEST / "split_membership.csv"
    membership.to_csv(member_csv, index=False, encoding="utf-8")
    paths[member_csv.name] = member_csv

    split_summaries = {}
    for split in SPLITS:
        part = rows.loc[rows.frozen_split.eq(split)].copy()
        path = DEST / f"{split}.parquet"
        part.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        paths[path.name] = path
        seconds = part.duration_exact.astype(float)
        split_summaries[split] = {
            "clips": int(len(part)),
            "hours": float(seconds.sum() / 3600),
            "components": int(part.component_id.nunique()),
        }

    hash_map = {name: sha256(path) for name, path in sorted(paths.items())}
    record = {
        "status": "frozen",
        "split_version": "phase3_5b_v1",
        "frozen_on": "2026-11-01",
        "source_proposal": str(SOURCE.relative_to(ROOT)),
        "source_proposal_sha256": sha256(SOURCE),
        "rows": int(len(rows)),
        "splits": split_summaries,
        "cross_split_overlap_counts": overlaps,
        "train_only_rows_in_validation_or_test": heldout_train_only,
        "prompt_overlap_distinct_transcript_strings": int(
            rows.groupby("transcript_phase2", dropna=False).frozen_split.nunique().gt(1).sum()
        ),
        "manifest_sha256": hash_map,
        "data_files_are_private_and_gitignored": True,
    }
    record_path = DEST / "freeze_record.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hash_map.items()), encoding="utf-8"
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
