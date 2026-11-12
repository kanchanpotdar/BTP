"""Create privacy-safe descriptive diagnostics from saved Phase 8 results.

No training, inference, checkpoint selection, or test-based tuning occurs here.
The private input manifests and prediction ZIP remain outside public outputs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    # Keep an observable local progress tracker when tqdm is unavailable.
    def tqdm(items, *, desc: str, unit: str):
        for index, item in enumerate(items, 1):
            if index == 1 or index % 100 == 0:
                print(f"{desc}: {index} {unit}s processed", flush=True)
            yield item


EXPECTED = {
    "B0": {"S": 36940, "D": 4395, "I": 25050},
    "F1": {"S": 13525, "D": 6158, "I": 9547},
}
MEMBERS = {
    "B0": "B0_test_predictions_private.jsonl",
    "F1": "F1_selected_test_predictions_private.jsonl",
}
COLORS = {"B0": "#E87452", "F1": "#177F79"}


def sha256(path: Path) -> str:
    """Hash a manifest before using private source rows."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, record_path: Path, expected_clips: int) -> list[dict]:
    """Read frozen bundle CSV only after checking its recorded hash."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert sha256(path) == record["manifest_sha256"], f"Manifest hash mismatch: {path}"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected_clips == int(record["clips"])
    assert len({row["sample_id"] for row in rows}) == expected_clips
    return rows


def read_predictions(path: Path) -> dict[str, dict[str, dict]]:
    """Read saved predictions without writing any row-level output."""
    result = {}
    with ZipFile(path) as archive:
        for model, name in MEMBERS.items():
            assert name in archive.namelist(), f"Missing {name}"
            rows = [json.loads(line) for line in tqdm(
                archive.read(name).splitlines(), desc=f"Reading {model}", unit="clip")]
            assert len(rows) == 683
            result[model] = {row["sample_id"]: row for row in rows}
            assert len(result[model]) == 683
            assert {key: sum(int(row[key]) for row in rows) for key in ("S", "D", "I")} == EXPECTED[model]
            assert sum(len(row["reference"]) for row in rows) == 65577
    assert set(result["B0"]) == set(result["F1"])
    return result


def align(reference: str, prediction: str) -> list[tuple[str, str, str]]:
    """Match the Phase 8 evaluator's code-point DP and tie order M, S, D, I."""
    ref, hyp = list(reference), list(prediction)
    width = len(hyp) + 1
    d = [[0] * width for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(width):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        left = d[i]
        above = d[i - 1]
        for j in range(1, width):
            left[j] = min(above[j] + 1, left[j - 1] + 1,
                          above[j - 1] + (ref[i - 1] != hyp[j - 1]))
    operations = []
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            operations.append(("M", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and j and d[i][j] == d[i - 1][j - 1] + 1:
            operations.append(("S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            operations.append(("D", ref[i - 1], "")); i -= 1
        else:
            operations.append(("I", "", hyp[j - 1])); j -= 1
    operations.reverse()
    return operations


def symbol_label(char: str) -> str:
    """Use unambiguous public labels, including whitespace and combining marks."""
    if char == " ":
        return "SPACE U+0020"
    code = f"U+{ord(char):04X}"
    name = unicodedata.name(char, "UNNAMED")
    return f"{code} {name}"


def plot_label(char: str) -> str:
    """Render a combining mark with a dotted circle for chart readability."""
    glyph = "◌" + char if unicodedata.combining(char) else "␠" if char == " " else char
    return f"{glyph}  U+{ord(char):04X}"


def train_frequency_band(count: int) -> str:
    if count == 0:
        return "Absent"
    if count < 10:
        return "1–9"
    if count < 100:
        return "10–99"
    if count < 1000:
        return "100–999"
    return "≥1000"


def repetition_fraction(text: str) -> float:
    """Fraction of code-point trigrams occurring beyond their first occurrence."""
    trigrams = [text[i:i + 3] for i in range(max(0, len(text) - 2))]
    if not trigrams:
        return 0.0
    return (len(trigrams) - len(set(trigrams))) / len(trigrams)


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-record", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-record", type=Path, required=True)
    parser.add_argument("--predictions-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read and cross-check frozen sources before any descriptive calculation.
    train = read_manifest(args.train_manifest, args.train_record, 3629)
    test = read_manifest(args.test_manifest, args.test_record, 683)
    predictions = read_predictions(args.predictions_zip)
    assert sum(map(lambda row: len(row["reference"]), train)) == 383474
    assert sum(map(lambda row: len(row["reference"]), test)) == 65577
    test_by_id = {row["sample_id"]: row for row in test}
    assert set(test_by_id) == set(predictions["B0"])
    for sample_id, row in test_by_id.items():
        for model in MEMBERS:
            assert predictions[model][sample_id]["reference"] == row["reference"]

    # Reconstruct the evaluator's one-code-point edit operations. Every per-clip
    # S/D/I count must exactly match the saved scorer output before aggregation.
    train_symbols = collections.Counter("".join(row["reference"] for row in train))
    test_symbols = collections.Counter("".join(row["reference"] for row in test))
    symbol_ops = {model: collections.defaultdict(collections.Counter) for model in MEMBERS}
    pairs = {model: collections.Counter() for model in MEMBERS}
    clip_rows = []
    for row in tqdm(test, desc="Aligning saved predictions", unit="clip"):
        sample_id = row["sample_id"]
        base = {"component": row["component_id"],
                "reference_length": len(row["reference"]),
                "duration_seconds": float(row["duration_seconds"])}
        for model in MEMBERS:
            pred = predictions[model][sample_id]
            ops = align(row["reference"], pred["normalized_prediction"])
            counts = collections.Counter(op for op, _, _ in ops)
            assert all(counts[key] == int(pred[key]) for key in ("S", "D", "I"))
            for operation, source, target in ops:
                if operation in ("M", "S", "D"):
                    symbol_ops[model][source][operation] += 1
                if operation == "I":
                    symbol_ops[model][target]["I"] += 1
                if operation == "S":
                    pairs[model][(source, target)] += 1
            base[f"{model}_S"] = int(pred["S"])
            base[f"{model}_D"] = int(pred["D"])
            base[f"{model}_I"] = int(pred["I"])
            base[f"{model}_rep_pred"] = repetition_fraction(pred["normalized_prediction"])
            base[f"{model}_rep_ref"] = repetition_fraction(pred["reference"])
            base[f"{model}_eos"] = bool(pred["eos"])
            base[f"{model}_cap"] = bool(pred["cap_without_eos"])
        clip_rows.append(base)

    # Publish only corpus-level symbol totals and clearly labelled alignment
    # diagnostics. Insertions have no reference-symbol denominator.
    symbols = []
    symbol_chars = {}
    all_relevant_symbols = set(test_symbols)
    for model in MEMBERS:
        all_relevant_symbols.update(char for char, ops in symbol_ops[model].items() if ops["I"])
    for char in sorted(all_relevant_symbols):
        test_count = test_symbols[char]
        item = {"symbol": symbol_label(char), "unicode": f"U+{ord(char):04X}",
                "train_reference_count": train_symbols[char],
                "test_reference_count": test_count,
                "class": "space" if char == " " else
                "combining mark" if unicodedata.combining(char) else "other"}
        for model in MEMBERS:
            ops = symbol_ops[model][char]
            assert ops["M"] + ops["S"] + ops["D"] == test_count
            item[f"{model}_substituted_from"] = ops["S"]
            item[f"{model}_deleted"] = ops["D"]
            item[f"{model}_reference_error_rate_percent"] = (
                100 * (ops["S"] + ops["D"]) / test_count if test_count else None)
            item[f"{model}_inserted_as"] = ops["I"]
        symbols.append(item)
        symbol_chars[item["unicode"]] = char
    symbols.sort(key=lambda item: (-item["F1_substituted_from"] - item["F1_deleted"], item["unicode"]))
    for model in MEMBERS:
        assert sum(item[f"{model}_inserted_as"] for item in symbols) == EXPECTED[model]["I"]

    # Group test reference positions by symbol frequency in the training corpus.
    bands = []
    for label in ("Absent", "1–9", "10–99", "100–999", "≥1000"):
        group = [item for item in symbols if item["test_reference_count"] > 0
                 and train_frequency_band(item["train_reference_count"]) == label]
        n = sum(item["test_reference_count"] for item in group)
        err = sum(item["F1_substituted_from"] + item["F1_deleted"] for item in group)
        bands.append({"train_frequency_band": label, "distinct_test_symbols": len(group),
                      "test_reference_positions": n, "F1_substitutions_plus_deletions": err,
                      "F1_reference_position_error_percent": 100 * err / n if n else None})
    assert sum(row["test_reference_positions"] for row in bands) == 65577
    assert sum(row["F1_substitutions_plus_deletions"] for row in bands) == 13525 + 6158

    # Compare train/test reference length coverage using fixed descriptive bins.
    length_bands = []
    for label, lo, hi in (("1–40", 1, 40), ("41–80", 41, 80),
                          ("81–120", 81, 120), (">120", 121, math.inf)):
        train_n = sum(lo <= len(row["reference"]) <= hi for row in train)
        test_n = sum(lo <= len(row["reference"]) <= hi for row in test)
        length_bands.append({"reference_code_points": label, "train_clips": train_n,
                             "train_share_percent": 100 * train_n / 3629,
                             "test_clips": test_n, "test_share_percent": 100 * test_n / 683})
    assert sum(x["train_clips"] for x in length_bands) == 3629
    assert sum(x["test_clips"] for x in length_bands) == 683

    # High-error clips are summarized as a group; no IDs or strings are exported.
    difficult = [row for row in clip_rows if 100 * (row["F1_S"] + row["F1_D"] + row["F1_I"]) / row["reference_length"] >= 100]
    ordinary = [row for row in clip_rows if row not in difficult]
    assert len(difficult) == 14 and len(ordinary) == 669
    def summarize_group(group: list[dict]) -> dict:
        totals = {key: sum(row[f"F1_{key}"] for row in group) for key in ("S", "D", "I")}
        lengths = [row["reference_length"] for row in group]
        durations = [row["duration_seconds"] for row in group]
        return {"clips": len(group), "reference_code_points": sum(lengths),
                "median_reference_code_points": median(lengths),
                "reference_length_range": [min(lengths), max(lengths)],
                "median_duration_seconds": median(durations),
                "duration_range_seconds": [min(durations), max(durations)],
                "S": totals["S"], "D": totals["D"], "I": totals["I"],
                "insertion_share_of_edits_percent": 100 * totals["I"] / sum(totals.values()),
                "median_prediction_repeat_trigram_fraction": median([row["F1_rep_pred"] for row in group]),
                "median_reference_repeat_trigram_fraction": median([row["F1_rep_ref"] for row in group]),
                "eos_count": sum(row["F1_eos"] for row in group),
                "cap_hits": sum(row["F1_cap"] for row in group)}
    difficult_summary = {"F1_clip_cer_at_least_100_percent": summarize_group(difficult),
                         "F1_clip_cer_below_100_percent": summarize_group(ordinary)}

    # Anonymous component labels are assigned by decreasing test clip count.
    # The private component-to-label mapping is deliberately not exported.
    component_groups = collections.defaultdict(list)
    for row in clip_rows:
        component_groups[row["component"]].append(row)
    assert len(component_groups) == 13
    components = []
    for index, (_, group) in enumerate(sorted(component_groups.items(), key=lambda pair: (-len(pair[1]), pair[0])), 1):
        denominator = sum(row["reference_length"] for row in group)
        item = {"anonymous_component": f"C{index:02d}", "clips": len(group),
                "reference_code_points": denominator}
        for model in MEMBERS:
            for key in ("S", "D", "I"):
                item[f"{model}_{key}"] = sum(row[f"{model}_{key}"] for row in group)
            item[f"{model}_corpus_cer_percent"] = 100 * sum(item[f"{model}_{key}"] for key in ("S", "D", "I")) / denominator
        components.append(item)
    assert sum(row["clips"] for row in components) == 683
    assert sum(row["reference_code_points"] for row in components) == 65577
    for model in MEMBERS:
        assert {key: sum(row[f"{model}_{key}"] for row in components) for key in ("S", "D", "I")} == EXPECTED[model]

    # Four standalone publication figures provide readable thesis visuals.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    top = [x for x in symbols if x["class"] != "space" and x["test_reference_count"]][:18]
    fig, ax = plt.subplots(figsize=(9.6, 7.4), constrained_layout=True)
    y = np.arange(len(top))
    ax.barh(y, [x["F1_substituted_from"] for x in top], color="#4D7EA8", label="Substitutions")
    ax.barh(y, [x["F1_deleted"] for x in top], left=[x["F1_substituted_from"] for x in top],
            color="#EF9B4B", label="Deletions")
    ax.set(yticks=y, yticklabels=[plot_label(symbol_chars[x["unicode"]]) for x in top], xlabel="Test reference code-point errors",
           title="Most affected non-space reference code points")
    ax.invert_yaxis(); ax.legend(frameon=False); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.output_dir / "01_symbol_errors.png"); plt.close(fig)

    inserted = sorted([x for x in symbols if x["class"] != "space"],
                      key=lambda item: -item["F1_inserted_as"])[:18]
    fig, ax = plt.subplots(figsize=(9.6, 7.4), constrained_layout=True)
    y = np.arange(len(inserted))
    ax.barh(y, [x["F1_inserted_as"] for x in inserted], color="#B17D5B")
    ax.set(yticks=y, yticklabels=[plot_label(symbol_chars[x["unicode"]]) for x in inserted],
           xlabel="Inserted output code points", title="Most frequent non-space F1 insertions")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.output_dir / "01b_inserted_symbols.png"); plt.close(fig)

    common_pairs = pairs["F1"].most_common(12)
    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    y = np.arange(len(common_pairs))
    ax.barh(y, [count for _, count in common_pairs], color="#836AA6")
    ax.set(yticks=y, yticklabels=[f"{plot_label(source)} → {plot_label(target)}"
                                   for (source, target), _ in common_pairs],
           xlabel="Alignment count", title="Common F1 reference-to-output substitutions")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.output_dir / "01c_substitution_pairs.png"); plt.close(fig)

    nonempty = [row for row in bands if row["test_reference_positions"]]
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    bars = ax.bar([row["train_frequency_band"] for row in nonempty],
                  [row["F1_reference_position_error_percent"] for row in nonempty], color="#6A77A8")
    ax.bar_label(bars, labels=[f"n={row['test_reference_positions']:,}" for row in nonempty], padding=3, fontsize=9)
    ax.set(xlabel="Frequency of a symbol in train references", ylabel="F1 substituted or deleted positions (%)",
           title="Test reference-symbol errors by training frequency")
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.output_dir / "02_train_symbol_coverage.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    x = np.arange(len(length_bands)); width = .36
    for shift, split, color in ((-width/2, "train", "#7898BD"), (width/2, "test", COLORS["F1"])):
        bars = ax.bar(x + shift, [row[f"{split}_share_percent"] for row in length_bands],
                      width, label=split.title(), color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=[row["reference_code_points"] for row in length_bands],
           xlabel="Reference length (NFD code points)", ylabel="Share of clips (%)",
           title="Reference-length coverage in frozen train and test")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.output_dir / "03_reference_length_coverage.png"); plt.close(fig)

    labels = ["CER ≥100%\n(14 clips)", "CER <100%\n(669 clips)"]
    groups = [difficult_summary["F1_clip_cer_at_least_100_percent"],
              difficult_summary["F1_clip_cer_below_100_percent"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
    for ax, key, title, unit in [
        (axes[0], "median_reference_code_points", "Median reference length", "code points"),
        (axes[1], "insertion_share_of_edits_percent", "Insertions among edits", "%"),
        (axes[2], "median_prediction_repeat_trigram_fraction", "Median repeated trigrams", "fraction")]:
        values = [group[key] for group in groups]
        bars = ax.bar(labels, values, color=["#B85B58", COLORS["F1"]])
        ax.bar_label(bars, fmt="%.2f" if key.endswith("fraction") else "%.1f", padding=3)
        ax.set(title=title, ylabel=unit)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("The 14 highest-CER clips are shorter and insertion-heavy")
    fig.savefig(args.output_dir / "03b_difficult_clips.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    x = np.arange(len(components)); width = .38
    for shift, model in ((-width/2, "B0"), (width/2, "F1")):
        ax.bar(x + shift, [row[f"{model}_corpus_cer_percent"] for row in components],
               width, color=COLORS[model], label=model)
    ax.set(xticks=x, xticklabels=[row["anonymous_component"] for row in components],
           xlabel="Anonymous held-out component, ordered by clip count", ylabel="Component corpus CER (%)",
           title="Held-out error varies across 13 observed components")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.output_dir / "04_component_cer.png"); plt.close(fig)

    # Write aggregate-only JSON, including a compact confusion-pair summary.
    output = {
        "scope": "saved Phase 8 test predictions and frozen train/test manifests; descriptive only",
        "validation_checks": {"train_clips": 3629, "test_clips": 683, "test_components": 13,
                              "train_reference_code_points": 383474,
                              "test_reference_code_points": 65577, "B0_edits": EXPECTED["B0"],
                              "F1_edits": EXPECTED["F1"], "clip_alignment_counts_exact": True},
        "method_notes": {
            "symbol_analysis": "NFD Unicode code points, evaluator tie order M/S/D/I; substitutions and deletions assigned to reference symbols; insertions counted as output symbols without a reference denominator. Alignment paths can be ambiguous.",
            "repetition_fraction": "Repeated code-point trigrams after their first occurrence divided by all trigrams. Descriptive; natural repetition may also raise this measure.",
            "component_labels": "Anonymous labels sorted by decreasing held-out clip count; private IDs and mapping excluded.",
            "causality": "Training frequency, recording group, length and errors are associated; no causal attribution or test-based selection."},
        "distinct_symbols": {"train": len(train_symbols), "test": len(test_symbols),
                             "test_symbols_absent_from_train": sum(train_symbols[s] == 0 for s in test_symbols),
                             "test_positions_on_absent_symbols": sum(n for s, n in test_symbols.items() if train_symbols[s] == 0)},
        "symbol_rows": symbols,
        "top_F1_substitutions": [
            {"reference_symbol": symbol_label(source), "predicted_symbol": symbol_label(target), "count": count}
            for (source, target), count in pairs["F1"].most_common(20)],
        "train_frequency_bands": bands, "reference_length_bands": length_bands,
        "difficult_clip_groups": difficult_summary, "anonymous_components": components,
    }
    destination = args.output_dir / "phase8_thesis_diagnostics_aggregate.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"distinct_symbols": output["distinct_symbols"],
                      "difficult": difficult_summary, "components": len(components)}, indent=2))


if __name__ == "__main__":
    main()
