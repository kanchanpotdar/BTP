"""Propose and audit a third, group-disjoint Narration split.

This is a diagnostic proposal. It does not freeze manifests or run a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase3_5_20261031_224609/revised_split_proposal.parquet"
OUTPUT = ROOT / "outputs/phase3_5b_20261031"
SPLITS = ("train", "validation", "test")


def group_data(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.assign(
        hours=rows.duration_exact.astype(float) / 3600,
        short=rows.duration_exact.astype(float).lt(3).astype(int),
        female_hours=np.where(rows.gender.eq("Female"), rows.duration_exact / 3600, 0),
        male_hours=np.where(rows.gender.eq("Male"), rows.duration_exact / 3600, 0),
        age_other_hours=np.where(rows.age_group.ne("18-30"), rows.duration_exact / 3600, 0),
        month_later_hours=np.where(rows.recording_month.ne("2024-11"), rows.duration_exact / 3600, 0),
    )
    return work.groupby("component_id", sort=True).agg(
        hours=("hours", "sum"), clips=("sample_id", "size"), short=("short", "sum"),
        female_hours=("female_hours", "sum"), male_hours=("male_hours", "sum"),
        age_other_hours=("age_other_hours", "sum"), month_later_hours=("month_later_hours", "sum"),
        train_only=("train_only_effective", "max"),
    )


def split_metrics(groups: pd.DataFrame, assignment: np.ndarray) -> dict:
    result = {}
    for split_id, name in enumerate(SPLITS):
        g = groups.iloc[np.flatnonzero(assignment == split_id)]
        hours = float(g.hours.sum())
        clips = int(g.clips.sum())
        female = float(g.female_hours.sum())
        male = float(g.male_hours.sum())
        result[name] = dict(
            hours=hours, clips=clips, components=len(g),
            female_hours_share=female / hours if hours else 0,
            male_hours_share=male / hours if hours else 0,
            short_clip_share=int(g.short.sum()) / clips if clips else 0,
            largest_component_hours_share=float(g.hours.max()) / hours if hours else 0,
            other_age_hours_share=float(g.age_other_hours.sum()) / hours if hours else 0,
            later_month_hours_share=float(g.month_later_hours.sum()) / hours if hours else 0,
        )
    return result


def score(groups: pd.DataFrame, assignment: np.ndarray) -> float:
    metrics = split_metrics(groups, assignment)
    total = float(groups.hours.sum())
    cost = 0.0
    for name in ("validation", "test"):
        m = metrics[name]
        h = m["hours"]
        # Feasible target is about 1.9 h in each held-out split, since three
        # large clean groups and the short-clip-heavy group stay in training.
        cost += 4.0 * ((h - 0.13 * total) / (0.13 * total)) ** 2
        cost += 25.0 * max(0, 0.25 - m["female_hours_share"]) ** 2
        cost += 25.0 * max(0, m["female_hours_share"] - 0.75) ** 2
        cost += 6.0 * (m["female_hours_share"] - 0.50) ** 2
        cost += 35.0 * max(0, m["largest_component_hours_share"] - 0.45) ** 2
        cost += 20.0 * max(0, m["short_clip_share"] - 0.35) ** 2
        cost += 0.4 * max(0, 6 - m["components"]) ** 2
        cost += 1.0 * max(0, 0.15 - m["other_age_hours_share"]) ** 2
        cost += 0.2 * max(0, 0.10 - m["later_month_hours_share"]) ** 2
    cost += 0.25 * ((metrics["validation"]["hours"] - metrics["test"]["hours"]) / total) ** 2
    return cost


def propose(groups: pd.DataFrame) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(20261031)
    values = {name: groups[name].to_numpy(dtype=float) for name in (
        "hours", "clips", "short", "female_hours", "age_other_hours", "month_later_hours"
    )}
    total = float(values["hours"].sum())

    def fast_score(assignment: np.ndarray) -> float:
        cost = 0.0
        heldout_hours = []
        for split_id in (1, 2):
            mask = assignment == split_id
            h = float(values["hours"][mask].sum())
            c = float(values["clips"][mask].sum())
            if h == 0 or c == 0:
                return float("inf")
            female = float(values["female_hours"][mask].sum()) / h
            largest = float(values["hours"][mask].max()) / h
            short = float(values["short"][mask].sum()) / c
            other_age = float(values["age_other_hours"][mask].sum()) / h
            later_month = float(values["month_later_hours"][mask].sum()) / h
            cost += 4.0 * ((h - 0.13 * total) / (0.13 * total)) ** 2
            cost += 25.0 * max(0, 0.25 - female) ** 2
            cost += 25.0 * max(0, female - 0.75) ** 2
            cost += 6.0 * (female - 0.50) ** 2
            cost += 35.0 * max(0, largest - 0.45) ** 2
            cost += 20.0 * max(0, short - 0.35) ** 2
            cost += 0.4 * max(0, 6 - int(mask.sum())) ** 2
            cost += 1.0 * max(0, 0.15 - other_age) ** 2
            cost += 0.2 * max(0, 0.10 - later_month) ** 2
            heldout_hours.append(h)
        cost += 0.25 * ((heldout_hours[0] - heldout_hours[1]) / total) ** 2
        return cost

    ids = groups.index.to_numpy()
    # These are structural constraints, chosen before looking at model scores.
    fixed = groups.train_only.to_numpy() | np.isin(ids, ["A_000008", "A_000036", "A_000022", "A_000166"])
    movable = np.flatnonzero(~fixed)
    best_assignment = None
    best_score = float("inf")
    for restart in range(10):
        assignment = np.zeros(len(groups), dtype=np.int8)
        assignment[movable] = rng.choice([0, 1, 2], size=len(movable), p=[0.08, 0.46, 0.46])
        current = fast_score(assignment)
        for step in range(18000):
            i = int(rng.choice(movable))
            previous = int(assignment[i])
            candidate = int(rng.choice([x for x in (0, 1, 2) if x != previous]))
            assignment[i] = candidate
            proposed = fast_score(assignment)
            temperature = 0.10 * (1 - step / 18000) + 0.0001
            if proposed < current or rng.random() < np.exp((current - proposed) / temperature):
                current = proposed
            else:
                assignment[i] = previous
            if current < best_score:
                best_score = current
                best_assignment = assignment.copy()
    assert best_assignment is not None
    return best_assignment, best_score


def audit(rows: pd.DataFrame, groups: pd.DataFrame, assignment: np.ndarray) -> dict:
    metrics = split_metrics(groups, assignment)
    overlap = {}
    for key in ("component_id", "speaker_group_id", "speaker_id", "speaker_id_raw", "recording_session_id", "exact_waveform_sha256"):
        n = rows.loc[rows[key].notna()].groupby(key).phase35b_split.nunique()
        overlap[key] = int(n.gt(1).sum())
    heldout_train_only = int(rows.loc[rows.phase35b_split.ne("train"), "train_only_effective"].sum())
    gates = {
        "no_known_cross_split_leakage": all(x == 0 for x in overlap.values()),
        "no_train_only_in_heldout": heldout_train_only == 0,
        "validation_and_test_hours_1_5_to_2_3": all(1.5 <= metrics[x]["hours"] <= 2.3 for x in ("validation", "test")),
        "heldout_female_hours_25_to_75_percent": all(0.25 <= metrics[x]["female_hours_share"] <= 0.75 for x in ("validation", "test")),
        "heldout_largest_component_at_most_45_percent": all(metrics[x]["largest_component_hours_share"] <= 0.45 for x in ("validation", "test")),
        "heldout_under_3s_clips_at_most_35_percent": all(metrics[x]["short_clip_share"] <= 0.35 for x in ("validation", "test")),
        "heldout_at_least_6_components": all(metrics[x]["components"] >= 6 for x in ("validation", "test")),
    }
    return {"metrics": metrics, "cross_split_overlaps": overlap, "heldout_train_only_rows": heldout_train_only, "gates": gates, "all_gates_pass": all(gates.values()), "status": "proposed_not_frozen"}


def main() -> None:
    rows = pd.read_parquet(SOURCE)
    groups = group_data(rows)
    assignment, objective = propose(groups)
    mapping = dict(zip(groups.index, [SPLITS[i] for i in assignment]))
    rows["phase35b_split"] = rows.component_id.map(mapping)
    result = audit(rows, groups, assignment)
    result["objective"] = objective
    result["source"] = str(SOURCE.relative_to(ROOT))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "split_proposal.parquet", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(result["metrics"]).T.to_csv(OUTPUT / "split_summary.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
