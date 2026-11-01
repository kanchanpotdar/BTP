"""Audit saved PRIVATE Phase 5B predictions; write only allowlisted aggregate JSON.

No model loading or inference. Requires pandas/pyarrow for the private frozen manifest.
Run with --help; never commit the private ZIP or either input manifest.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path
import pandas as pd

def normalize_reference_or_hypothesis(text):
    text = unicodedata.normalize("NFD", str(text))
    text = text.strip()
    return re.sub(r" {2,}", " ", text)


def edit_counts(reference, hypothesis):
    """Return one optimal code-point edit decomposition (S, D, I)."""
    ref = list(reference)
    hyp = list(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + substitution_cost,
            )
    substitutions = deletions = insertions = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and distance[i][j] == distance[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and distance[i][j] == distance[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            raise RuntimeError("Could not backtrack the character edit-distance matrix.")
    return substitutions, deletions, insertions


def score_text_pair(reference, hypothesis):
    ref = normalize_reference_or_hypothesis(reference)
    hyp = normalize_reference_or_hypothesis(hypothesis)
    s, d, i = edit_counts(ref, hyp)
    return {"reference": ref, "hypothesis": hyp, "S": s, "D": d, "I": i, "reference_codepoints": len(ref)}


# Small deterministic checks for exact match, substitution, deletion, insertion,
# combining marks, ordinary-space policy, and the empty-output rule.
assert score_text_pair("abc", "abc")["S"] == 0
assert score_text_pair("abc", "axc")["S"] == 1
assert score_text_pair("abc", "ac")["D"] == 1
assert score_text_pair("ac", "abc")["I"] == 1
assert score_text_pair("ã", "a\u0303")["S"] == 0
assert score_text_pair("a  b", "a b")["S"] == score_text_pair("a  b", "a b")["D"] == score_text_pair("a  b", "a b")["I"] == 0
assert score_text_pair("a b", "ab")["D"] == 1  # Spaces are part of CER.
assert score_text_pair("abc", "")["D"] == 3  # Empty predictions remain deletions.
assert normalize_reference_or_hypothesis("  a   b  ") == "a b"
print("CER self-checks passed. IPA marks are not stripped; only NFD and repeated ordinary spaces are normalized.")

def audit(private_zip, bundle_manifest, frozen_manifest, executed_notebook=None):
    with zipfile.ZipFile(private_zip) as z:
        def read(tail):
            matches = [n for n in z.namelist() if n.endswith("/" + tail)]
            if len(matches) != 1:
                raise ValueError("Expected one private result file per type.")
            return z.read(matches[0]).decode("utf-8")
        config = json.loads(read("run_config.json"))
        old = json.loads(read("metrics.json"))
        records = [json.loads(line) for line in read("predictions.jsonl").splitlines() if line.strip()]
        passes = [json.loads(line) for line in read("inference_passes.jsonl").splitlines() if line.strip()]
        csv_records = list(csv.DictReader(io.StringIO(read("predictions_private.csv"))))
    expected_hash = config["validation_manifest_sha256"]
    assert hashlib.sha256(Path(bundle_manifest).read_bytes()).hexdigest() == expected_hash
    with Path(bundle_manifest).open(encoding="utf-8") as f:
        manifest = {r["sample_id"]: r for r in csv.DictReader(f)}
    frozen = pd.read_parquet(frozen_manifest).set_index("sample_id")
    assert len(frozen) == len(manifest) == 732 and frozen.index.is_unique
    assert set(frozen.index) == set(manifest)
    assert set(frozen["frozen_split"]) == {"validation"}
    latest = {}
    for row in records:
        assert row["run_fingerprint"] == config["run_fingerprint"]
        latest[row["sample_id"]] = row
    assert len(latest) == 732 and set(latest) == set(manifest)
    assert len(csv_records) == 732 and {r["sample_id"] for r in csv_records} == set(latest)
    for row in csv_records:
        original = latest[row["sample_id"]]
        for key in ("reference", "reference_raw", "prediction", "raw_prediction", "component_id", "status"):
            assert row[key] == original[key]
        for key in ("S", "D", "I", "reference_codepoints"):
            assert int(row[key]) == original[key]
    rows = list(latest.values())
    for row in rows:
        expected = manifest[row["sample_id"]]
        source = frozen.loc[row["sample_id"]]
        assert row["split"] == expected["split"] == "validation"
        assert row["reference"] == expected["reference"] == source.transcript_phase2
        assert row["reference_raw"] == expected["reference_raw"] == source.transcript_raw
        assert row["component_id"] == expected["component_id"]
        assert expected["waveform_sha256"] == source.exact_waveform_sha256
        assert math.isclose(row["duration_seconds"], float(expected["duration_seconds"]), abs_tol=1e-9)
        assert math.isclose(row["duration_seconds"], float(source.duration_exact), abs_tol=1e-9)
        scored = score_text_pair(row["reference"], row["raw_prediction"])
        assert scored["hypothesis"] == row["prediction"]
        assert all(scored[k] == row[k] for k in ("S", "D", "I", "reference_codepoints"))
        assert row["contains_replacement_character"] == ("\ufffd" in row["raw_prediction"])
        assert not (row["eos_terminated"] and row["hit_448_token_cap_without_eos"])
        if row["hit_448_token_cap_without_eos"]:
            assert row["generated_token_count_including_prefix"] >= 448
    assert all(p["run_fingerprint"] == config["run_fingerprint"] for p in passes)
    full_passes = [p for p in passes if p["scope"] == "full_validation"]
    assert full_passes
    last = full_passes[-1]
    duration = sum(r["duration_seconds"] for r in rows)
    elapsed = sum(r["elapsed_seconds"] for r in rows)
    s, d, i = (sum(r[k] for r in rows) for k in ("S", "D", "I"))
    denominator = sum(r["reference_codepoints"] for r in rows)
    assert denominator == 64137
    assert (s, d, i) == (old["substitutions"], old["deletions"], old["insertions"])
    assert math.isclose((s+d+i)/denominator, old["corpus_cer"])
    gen = config["generation_config"]
    data = {
        "experiment_id": "B0", "status": "complete", "scope": "full_validation",
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "seed": config["seed"], "transformers_version": config["transformers"],
        "torch_version": config["torch"], "gpu": old["gpu"], "precision": "FP32",
        "batch_size": 1, "do_sample": gen["do_sample"], "num_beams": gen["num_beams"],
        "max_length": gen["max_length"], "language": gen["language"], "task": gen["task"],
        "begin_suppress_tokens": gen["begin_suppress_tokens"], "suppress_tokens": gen["suppress_tokens"],
        "forced_decoder_ids": gen["forced_decoder_ids"],
        "normalization_policy": config["normalization_policy"],
        "requested_validation_clips": 732, "predictions_available": len(rows),
        "components": len({r["component_id"] for r in rows}),
        "reference_codepoints_scored": denominator,
        "substitutions": s, "deletions": d, "insertions": i, "edit_count": s+d+i,
        "corpus_cer": (s+d+i)/denominator, "corpus_cer_percent": 100*(s+d+i)/denominator,
        "decode_errors": sum(r["status"] == "decode_error" for r in rows),
        "empty_predictions": sum(r["prediction"] == "" for r in rows),
        "eos_terminated": sum(r["eos_terminated"] for r in rows),
        "hit_448_token_cap_without_eos": sum(r["hit_448_token_cap_without_eos"] for r in rows),
        "replacement_character_outputs": sum(r["contains_replacement_character"] for r in rows),
        "automatic_english_prefix_clips": sum(r["decoder_prefix_ids"] == [50258,50259,50359] for r in rows),
        "evaluated_audio_hours": duration/3600, "summed_sample_runtime_seconds": elapsed,
        "mean_sample_runtime_seconds": elapsed/len(rows), "real_time_factor": elapsed/duration,
        "inference_passes": len(passes), "summed_pass_wall_seconds": sum(p["elapsed_seconds"] for p in passes),
        "full_validation_pass_elapsed_seconds": last["elapsed_seconds"],
        "full_validation_pass_newly_processed_clips": last["newly_processed_clips"],
        "full_validation_pass_peak_allocated_gpu_mib": last["peak_allocated_gpu_bytes"]/1024**2,
        "full_validation_pass_peak_reserved_gpu_mib": last["peak_reserved_gpu_bytes"]/1024**2,
        "test_data_used": False, "inference_rerun_for_reporting": False,
        "checks": {"membership_and_references_match_frozen_validation": True,
                   "all_per_clip_SDI_recomputed": True, "csv_matches_jsonl": True,
                   "all_plot_breakdowns_cover_full_validation": True},
    }
    def aggregate(group, label):
        ref = sum(r["reference_codepoints"] for r in group)
        edits = sum(r["S"]+r["D"]+r["I"] for r in group)
        return {"group": label, "clips": len(group), "reference_codepoints": ref,
                "edits": edits, "cer": edits/ref if ref else None}
    bands = [(0,3,"(0, 3] s"),(3,5,"(3, 5] s"),(5,10,"(5, 10] s"),
             (10,15,"(10, 15] s"),(15,20,"(15, 20] s"),(20,30,"(20, 30] s")]
    data["duration_breakdown"] = [aggregate([r for r in rows if low < r["duration_seconds"] <= high], label) for low,high,label in bands]
    # Reassign neutral plot positions by CER; do not export frozen component aliases or a mapping.
    components = [aggregate([r for r in rows if r["component_id"] == key], "") for key in {r["component_id"] for r in rows}]
    components.sort(key=lambda r:(r["cer"],r["clips"]))
    for index,row in enumerate(components,1): row["group"] = f"Component {index:02d}"
    data["component_breakdown"] = components
    for key in ("duration_breakdown", "component_breakdown"):
        assert sum(r["clips"] for r in data[key]) == 732
        assert sum(r["reference_codepoints"] for r in data[key]) == denominator
        assert sum(r["edits"] for r in data[key]) == s+d+i
    data["stopping_breakdown"] = [
        aggregate([r for r in rows if r["eos_terminated"]], "EOS terminated"),
        aggregate([r for r in rows if r["hit_448_token_cap_without_eos"]], "448-token cap without EOS"),
    ]
    assert sum(r["clips"] for r in data["stopping_breakdown"]) == 732
    assert sum(r["reference_codepoints"] for r in data["stopping_breakdown"]) == denominator
    assert sum(r["edits"] for r in data["stopping_breakdown"]) == s+d+i
    token_bands = [(0,64),(64,128),(128,192),(192,256),(256,320),(320,384),(384,447),(447,448)]
    data["generated_token_length_histogram"] = [
        {"group": f"({low}, {high}]", "clips": sum(low < r["generated_token_count_including_prefix"] <= high for r in rows),
         "cap_without_eos_clips": sum(low < r["generated_token_count_including_prefix"] <= high and r["hit_448_token_cap_without_eos"] for r in rows)}
        for low,high in token_bands
    ]
    assert sum(r["clips"] for r in data["generated_token_length_histogram"]) == 732
    assert sum(r["cap_without_eos_clips"] for r in data["generated_token_length_histogram"]) == data["hit_448_token_cap_without_eos"]
    if executed_notebook is not None:
        notebook = json.loads(Path(executed_notebook).read_text())
        output_text = "\n".join("".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", []) if o["output_type"] == "stream")
        assert config["run_fingerprint"] in output_text
        checks = {
            "737_payload_checks_passed": "737 hashes; 732 validation rows" in output_text,
            "scorer_self_checks_passed": "CER self-checks passed" in output_text,
            "model_parameters_fp32": "Model parameters are FP32: True" in output_text,
            "full_continuation_completed": "Processed 720/720" in output_text,
            "saved_full_coverage": "Full B0 validation baseline complete: True" in output_text,
            "two_finite_processor_examples": output_text.count("feature shape=(1, 80, 3000), dtype=torch.float32, finite=True") == 2,
        }
        assert all(checks.values())
        data["executed_notebook_checks"] = checks
        data["python_version"] = "3.13.15"
        data["tokenizers_version"] = "0.22.2"
        data["huggingface_hub_version"] = "0.36.2"
        data["numpy_version"] = "2.1.3"
        data["soundfile_version"] = "0.14.0"
        for name, key in [("Python", "python_version"), ("Tokenizers", "tokenizers_version"), ("Hugging Face Hub", "huggingface_hub_version"), ("NumPy", "numpy_version"), ("SoundFile", "soundfile_version")]:
            assert f"{name}: {data[key]}" in output_text
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-zip", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Aggregate JSON destination")
    parser.add_argument("--executed-notebook", type=Path, help="Optional private executed notebook to cross-check runtime and gate outputs")
    args = parser.parse_args()
    data = audit(args.private_zip, args.bundle_manifest, args.frozen_manifest, args.executed_notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Verified 732/732 frozen validation clips; wrote privacy-safe aggregate JSON.")
