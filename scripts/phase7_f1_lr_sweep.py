#!/usr/bin/env python3
"""Run the two independent Phase 7 F1 learning-rate checks on Kaggle.

This orchestration script deliberately reuses the validated F1 trainer. Each
rate starts from the pinned pretrained checkpoint, uses the same frozen train
and validation bundles, and writes to a separate private output directory.
The sealed test set is never discovered or read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # Keep discovery/help usable in a minimal local shell.
    def tqdm(items, **_kwargs):
        return items


RATES = (5e-6, 2e-5)
TRAIN_BUNDLE = "phase7_f1_train_v1"
VALIDATION_BUNDLE = "phase5_validation_baseline"


def bundle_type_from_zip(path: Path) -> str | None:
    """Read only bundle markers; never extract or inspect private rows here."""
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("bundle_record.json"):
                    record = json.loads(archive.read(member))
                    return record.get("bundle_type")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
        return None
    return None


def discover(root: Path, expected_type: str) -> Path:
    """Find exactly one attached directory or ZIP for the requested bundle."""
    matches: set[Path] = set()
    for marker in root.rglob("bundle_record.json"):
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("bundle_type") == expected_type:
                matches.add(marker.parent)
        except (OSError, json.JSONDecodeError):
            continue
    for archive in root.rglob("*.zip"):
        if bundle_type_from_zip(archive) == expected_type:
            matches.add(archive)
    if len(matches) != 1:
        found = "\n".join(f" - {item}" for item in sorted(matches)) or " (none)"
        raise FileNotFoundError(
            f"Expected exactly one {expected_type} bundle under {root}, found {len(matches)}:\n{found}"
        )
    return next(iter(matches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/phase7_F1_LR_private"))
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path(__file__).with_name("phase7_f1_train.py"),
        help="Validated F1 trainer copied/uploaded beside this script.",
    )
    args = parser.parse_args()

    if not args.trainer.is_file():
        raise FileNotFoundError(
            f"Missing validated trainer: {args.trainer}. Upload scripts/phase7_f1_train.py too."
        )
    train_bundle = discover(args.input_root, TRAIN_BUNDLE)
    validation_bundle = discover(args.input_root, VALIDATION_BUNDLE)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print({"train_bundle": str(train_bundle), "validation_bundle": str(validation_bundle), "rates": RATES})

    for rate in tqdm(RATES, desc="Learning-rate experiments", unit="run"):
        label = f"lr_{rate:.0e}".replace("-", "m")
        output = args.output_root / label
        command = [
            sys.executable,
            str(args.trainer),
            "--train-input", str(train_bundle),
            "--validation-input", str(validation_bundle),
            "--output", str(output),
            "--physical-batch", "2",
            "--accumulation", "4",
            "--validation-batch", "8",
            "--max-epochs", "10",
            "--patience", "3",
            "--learning-rate", str(rate),
            "--weight-decay", "0.01",
            "--seed", "0",
        ]
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing private output: {output}")
        print({"starting_learning_rate": rate, "output": str(output)})
        subprocess.run(command, check=True)
    print(f"Completed both LR runs. Private outputs are under {args.output_root}.")


if __name__ == "__main__":
    main()
