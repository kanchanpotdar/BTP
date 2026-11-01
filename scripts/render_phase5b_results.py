"""Render public Phase 5B plots from aggregate JSON only; no inference or private inputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(root):
    root = Path(root)
    data = json.loads((root / "docs/PHASE5B_RESULTS_AGGREGATE.json").read_text())
    assert data["predictions_available"] == data["requested_validation_clips"] == 732
    assert data["reference_codepoints_scored"] == 64137
    title = "B0 full validation · 732 clips · 64,137 reference code points"
    target = root / "assets/phase5b"
    target.mkdir(parents=True, exist_ok=True)
    for key, filename, color, horizontal in [
        ("duration_breakdown", "01_cer_by_duration.png", "#4c78a8", False),
        ("component_breakdown", "02_cer_by_component.png", "#59a14f", True),
    ]:
        rows = data[key]
        assert sum(r["clips"] for r in rows) == 732
        assert sum(r["reference_codepoints"] for r in rows) == 64137
        assert sum(r["edits"] for r in rows) == data["edit_count"]
        labels = [f"{r['group']} (n={r['clips']})" for r in rows]
        values = [100 * r["cer"] for r in rows]
        fig, ax = plt.subplots(figsize=(11, 7 if horizontal else 5), constrained_layout=True)
        if horizontal:
            ax.barh(labels, values, color=color)
            ax.invert_yaxis()
            ax.set_xlabel("Corpus code-point CER (%) within anonymous component")
            ax.set_xlim(0, max(values) * 1.2)
            for i, value in enumerate(values):
                ax.text(value + 1, i, f"{value:.1f}%", va="center")
            ax.axvline(100, color="#888", linestyle="--", linewidth=1)
        else:
            ax.bar(labels, values, color=color)
            ax.set_ylabel("Corpus code-point CER (%)")
            ax.tick_params(axis="x", rotation=20)
            ax.set_ylim(0, max(values) * 1.22)
            for i, value in enumerate(values):
                ax.text(i, value + 1, f"{value:.1f}%", ha="center")
            ax.axhline(100, color="#888", linestyle="--", linewidth=1)
        ax.set_title(title)
        fig.savefig(target / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    counts = [data[k] for k in ("substitutions", "deletions", "insertions")]
    ax.bar(["Substitutions", "Deletions", "Insertions"], counts, color=["#e15759", "#f28e2b", "#b07aa1"])
    for i, count in enumerate(counts):
        ax.text(i, count + 500, f"{count:,}", ha="center")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_ylabel("Character edit operations")
    ax.set_title(title)
    fig.savefig(target / "03_edit_counts.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    labels = ["EOS terminated", "448-token cap without EOS", "Contains U+FFFD", "Decode errors", "Empty outputs"]
    counts = [data[k] for k in ("eos_terminated", "hit_448_token_cap_without_eos", "replacement_character_outputs", "decode_errors", "empty_predictions")]
    ax.barh(labels, counts, color="#4c78a8")
    ax.invert_yaxis()
    for i, count in enumerate(counts):
        ax.text(count + 8, i, f"{count} ({count / 732:.1%})", va="center")
    ax.set_xlim(0, 800)
    ax.set_xlabel("Clips; U+FFFD may overlap stopping flags")
    ax.set_title(title)
    fig.savefig(target / "04_generation_flags.png", dpi=160)
    plt.close(fig)


    rows = data["stopping_breakdown"]
    assert sum(r["clips"] for r in rows) == 732
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    values = [100*r["cer"] for r in rows]
    ax.bar([f"{r['group']}\n(n={r['clips']}, ref={r['reference_codepoints']:,})" for r in rows], values, color=["#59a14f", "#e15759"])
    for i,value in enumerate(values): ax.text(i,value+2,f"{value:.1f}%",ha="center")
    ax.set_ylim(0,max(values)*1.18)
    ax.set_ylabel("Corpus code-point CER (%)")
    ax.axhline(100,color="#888",linestyle="--",linewidth=1)
    ax.set_title(title + "\nStopping outcome is observational; this is not a cap intervention")
    fig.savefig(target / "05_cer_by_stopping.png", dpi=160)
    plt.close(fig)
    rows = data["generated_token_length_histogram"]
    assert sum(r["clips"] for r in rows) == 732
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    cap = [r["cap_without_eos_clips"] for r in rows]
    other = [r["clips"]-r["cap_without_eos_clips"] for r in rows]
    ax.bar([r["group"] for r in rows],other,color="#4c78a8",label="EOS terminated")
    ax.bar([r["group"] for r in rows],cap,bottom=other,color="#e15759",label="Cap without EOS")
    for i,r in enumerate(rows): ax.text(i,r["clips"]+2,str(r["clips"]),ha="center")
    ax.set_ylim(0,max(r["clips"] for r in rows)*1.2)
    ax.set_ylabel("Clips")
    ax.set_xlabel("Full generated token count, including prefix/EOS (unequal-width bins)")
    ax.tick_params(axis="x",rotation=20)
    ax.set_title(title)
    ax.legend()
    fig.savefig(target / "06_generation_length_distribution.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1])
