"""Summarize saved Phase 8 test predictions without exposing clip records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    # Local review environments may omit tqdm; Kaggle normally provides it.
    def tqdm(items, *, desc: str, unit: str):
        for index, item in enumerate(items, 1):
            if index == 1 or index % 100 == 0:
                print(f"{desc}: {index} {unit}s processed", flush=True)
            yield item


EXPECTED = {
    "B0": {"S": 36940, "D": 4395, "I": 25050},
    "Selected F1": {"S": 13525, "D": 6158, "I": 9547},
}
FILES = {
    "B0": "B0_test_predictions_private.jsonl",
    "Selected F1": "F1_selected_test_predictions_private.jsonl",
}
COLORS = {"B0": "#EF8354", "Selected F1": "#287A78"}
THRESHOLDS = [10, 25, 50, 75, 100]
QUANTILES = [0, 10, 25, 50, 75, 90, 100]


def read_saved_rows(archive: ZipFile, name: str) -> list[dict]:
    """Read one saved prediction file and verify its declared coverage."""
    assert name in archive.namelist(), f"Missing ZIP member: {name}"
    rows = []
    for line in tqdm(archive.read(name).splitlines(), desc=f"Reading {name}", unit="clip"):
        rows.append(json.loads(line))
    assert len(rows) == 683, f"Expected 683 clips in {name}; found {len(rows)}"
    ids = [row["sample_id"] for row in rows]
    assert len(ids) == len(set(ids)), f"Duplicate clip IDs in {name}"
    return rows


def summarize(rows: list[dict], model: str) -> tuple[dict, dict[str, float]]:
    """Compute private row values in memory and return aggregates only."""
    by_id = {}
    totals = {key: 0 for key in ("S", "D", "I")}
    denominator = 0
    for row in tqdm(rows, desc=f"Scoring {model}", unit="clip"):
        reference_length = len(row["reference"])
        assert reference_length > 0, "An empty reference has no defined clip CER"
        edits = 0
        for key in totals:
            value = int(row[key])
            assert value >= 0
            totals[key] += value
            edits += value
        denominator += reference_length
        by_id[row["sample_id"]] = 100.0 * edits / reference_length
    assert denominator == 65577, f"Unexpected reference denominator: {denominator}"
    assert totals == EXPECTED[model], f"Edit totals disagree with published Phase 8 results: {totals}"
    values = np.asarray(list(by_id.values()), dtype=float)
    aggregate = {
        "clips": len(values),
        "reference_code_points": denominator,
        "corpus_cer_percent": 100.0 * sum(totals.values()) / denominator,
        "substitutions": totals["S"],
        "deletions": totals["D"],
        "insertions": totals["I"],
        "exact_zero_cer_clips": int(np.count_nonzero(values == 0)),
        "thresholds_strictly_below_percent": {
            str(t): {"clips": int(np.count_nonzero(values < t)),
                     "percent_of_clips": float(100.0 * np.mean(values < t))}
            for t in THRESHOLDS
        },
        "cer_at_least_100_percent": int(np.count_nonzero(values >= 100)),
        "clip_cer_quantiles_percent": {
            str(q): float(np.percentile(values, q)) for q in QUANTILES
        },
    }
    return aggregate, by_id


def save_plots(scores: dict[str, dict[str, float]], summary: dict, out: Path) -> None:
    """Draw aggregate plots; no sample identifiers or predictions are plotted."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})

    # Empirical cumulative curves answer how many clips fall below a CER threshold.
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    for model, by_id in scores.items():
        values = np.sort(np.asarray(list(by_id.values())))
        ax.step(values, 100 * np.arange(1, len(values) + 1) / len(values),
                where="post", lw=2.6, color=COLORS[model], label=model)
    for threshold in (25, 50, 100):
        ax.axvline(threshold, color="#AAB1B7", lw=1, ls="--")
    ax.set(xlim=(0, 200), ylim=(0, 100), xlabel="Clip CER (%)",
           ylabel="Test clips at or below CER (%)",
           title="How transcription errors are distributed across 683 test clips")
    ax.text(0.99, 0.03, "Curves shown to 200%; higher-CER clips remain in all counts",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "01_clip_cer_cumulative.png")
    plt.close(fig)

    # Strict threshold bars give the requested counts and shares directly.
    fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
    x = np.arange(len(THRESHOLDS))
    width = 0.36
    for offset, model in [(-width / 2, "B0"), (width / 2, "Selected F1")]:
        counts = [summary[model]["thresholds_strictly_below_percent"][str(t)]["clips"]
                  for t in THRESHOLDS]
        bars = ax.bar(x + offset, counts, width, label=model, color=COLORS[model])
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=[f"<{t}%" for t in THRESHOLDS], ylim=(0, 735),
           xlabel="Clip CER threshold (strictly below)", ylabel="Number of test clips",
           title="Clips meeting each error threshold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(out / "02_clip_cer_thresholds.png")
    plt.close(fig)

    # Quantiles show typical clips and the high-error tail without averaging per-clip CER.
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    visible_quantiles = QUANTILES[:-1]
    for model in FILES:
        y = [summary[model]["clip_cer_quantiles_percent"][str(q)]
             for q in visible_quantiles]
        ax.plot(visible_quantiles, y, marker="o", lw=2.4,
                color=COLORS[model], label=model)
    ax.set(xlabel="Percentile of test clips", ylabel="Clip CER at percentile (%)",
           title="Typical clips and the 90th percentile", xlim=(0, 92), ylim=(0, 180))
    ax.text(0.02, 0.96, "Maximum shown in aggregate JSON; extreme clips are off this scale",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(out / "03_clip_cer_quantiles.png")
    plt.close(fig)

    # Exclusive bands sum to all 683 clips and make CER over 100% visible.
    fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
    bands = ["0–<25", "25–<50", "50–<75", "75–<100", "≥100"]
    edges = [0, 25, 50, 75, 100, np.inf]
    x = np.arange(len(bands))
    for offset, model in [(-width / 2, "B0"), (width / 2, "Selected F1")]:
        values = np.asarray(list(scores[model].values()))
        counts = [int(np.count_nonzero((values >= lo) & (values < hi)))
                  for lo, hi in zip(edges[:-1], edges[1:])]
        assert sum(counts) == 683
        summary[model]["exclusive_cer_bands"] = dict(zip(bands, counts))
        bars = ax.bar(x + offset, counts, width, label=model, color=COLORS[model])
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=bands, xlabel="Clip CER band (%)",
           ylabel="Number of test clips", title="Where the 683 clips fall")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(out / "04_clip_cer_bands.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The ZIP stays private. Only checked aggregate values and figures are written.
    with ZipFile(args.input_zip) as archive:
        data = {model: read_saved_rows(archive, member) for model, member in FILES.items()}
    assert {r["sample_id"] for r in data["B0"]} == {
        r["sample_id"] for r in data["Selected F1"]}, "Model coverage differs"
    assert {r["sample_id"]: r["reference"] for r in data["B0"]} == {
        r["sample_id"]: r["reference"] for r in data["Selected F1"]}, "References differ"
    results = {model: summarize(rows, model) for model, rows in data.items()}
    summary = {model: pair[0] for model, pair in results.items()}
    scores = {model: pair[1] for model, pair in results.items()}
    save_plots(scores, summary, args.output_dir)
    output = {
        "source": "saved Phase 8 B0 and selected F1 test predictions",
        "scope": "same 683 frozen held-out clips; no inference or tuning",
        "clip_cer_definition": "100 * (S + D + I) / len(NFD-normalized reference)",
        "threshold_rule": "strictly below; equality excluded",
        "quantile_method": "NumPy percentile, linear interpolation",
        "models": summary,
    }
    (args.output_dir / "clip_cer_distribution_aggregate.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
