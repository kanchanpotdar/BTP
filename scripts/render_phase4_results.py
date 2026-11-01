"""Rebuild public Phase 4 figures from anonymous diagnostics, without audio or transcripts.

Usage: python scripts/render_phase4_results.py [--root /path/to/BTP]
Requires NumPy and Matplotlib. Does not load a model or run inference.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(root):
    metrics = json.loads((root / "docs/PHASE4_RESULTS_AGGREGATE.json").read_text())
    diagnostics = json.loads((root / "docs/PHASE4_SMOKE_DIAGNOSTICS.json").read_text())
    rows = diagnostics["rows"]
    output = root / "assets/phase4"
    output.mkdir(parents=True, exist_ok=True)
    x = np.array([r["example"] for r in rows])
    values = lambda key: np.array([r[key] for r in rows])

    def save(fig, name):
        fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        axes[0].bar(x, values("duration_seconds"), color="#377eb8")
        axes[0].set(title="Training-only audio examples", xlabel="Example position", ylabel="Audio seconds", xticks=x)
        axes[1].scatter(values("duration_seconds"), values("inference_seconds"), color="#377eb8")
        axes[1].set(title="Features + generation latency", xlabel="Audio seconds", ylabel="Inference seconds")
        fig.suptitle("Phase 4: audio duration and inference timing")
        save(fig, "01_inputs_and_timing.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        histogram = diagnostics["reference_histogram"]
        edges = np.array(histogram["bin_edges"])
        for scope, counts in histogram["scope_counts"].items():
            axes[0].bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.6,
                        label=f"{scope} (n={sum(counts):,})")
        axes[0].axvline(448, color="#e41a1c", linestyle="--", label="Decoder capacity 448")
        axes[0].set(title="Reference content tokens (special tokens excluded)",
                    xlabel="Content tokens", ylabel="Reference count (log scale)", yscale="log")
        axes[0].legend(fontsize=7)
        axes[1].bar(x, values("returned_token_count_including_prefix"), color="#984ea3")
        axes[1].axhline(448, color="#e41a1c", linestyle="--", label="Full-sequence limit 448")
        axes[1].set(title="Full decoder sequences, including prefix/EOS", xlabel="Example position",
                    ylabel="Returned tokens", xticks=x)
        axes[1].legend(fontsize=8)
        fig.suptitle("Phase 4: references fit; six generated sequences reach the limit")
        save(fig, "02_token_lengths.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        axes[0].bar(x - 0.2, values("reference_nfd_characters"), width=0.4, label="Reference", color="#377eb8")
        axes[0].bar(x + 0.2, values("prediction_nfd_characters"), width=0.4, label="Prediction", color="#ff7f00")
        axes[0].set(title="Normalized text lengths (not CER)", xlabel="Example position", ylabel="NFD code points, including spaces", xticks=x)
        axes[0].legend()
        axes[1].bar(x, values("longest_identical_token_run"), color="#984ea3")
        axes[1].set(title="Identical-token runs; phrase repetition may be missed", xlabel="Example position", ylabel="Longest consecutive run", xticks=x)
        fig.suptitle("Phase 4: output growth and repetition")
        save(fig, "03_output_growth_and_repetition.png")

        flag_keys = ["ended_with_eos", "reached_length_limit", "possible_truncation", "replacement_character_count"]
        flags = np.array([[bool(r[key]) for key in flag_keys] for r in rows], dtype=int)
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.imshow(flags, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(4), ["Ended with EOS", "Reached limit", "Limit without EOS", "Contains U+FFFD"], rotation=12, ha="right")
        ax.set_yticks(np.arange(len(rows)), [f"Example {i}" for i in x])
        for i in range(len(rows)):
            for j in range(4):
                ax.text(j, i, "Yes" if flags[i, j] else "No", ha="center", va="center", color="white" if flags[i, j] else "black")
        ax.set_title("Phase 4: stopping flags verified against full token sequences")
        save(fig, "04_stopping_and_unicode_flags.png")

        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        memory = [metrics["peak_allocated_gpu_mib"], metrics["peak_reserved_gpu_mib"]]
        bars = ax.bar(["Peak allocated", "Peak reserved"], memory, color=["#377eb8", "#984ea3"])
        ax.bar_label(bars, fmt="%.1f MiB", padding=4)
        ax.set(ylim=(0, max(memory) * 1.2), ylabel="PyTorch CUDA memory (MiB)", title="Phase 4: T4 batch-1 FP32 inference memory")
        save(fig, "05_inference_memory.png")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(f"Saved five figures to {render(args.root)}")
