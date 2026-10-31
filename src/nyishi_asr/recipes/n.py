"""Deterministic recovered n workflow. Execute via nyishi_asr.restore."""


# Original zero-based notebook cell 2
print("=" * 100)
print("CELL N0 - SETUP: imports, folders and a SAFE decision-log helper")
print("=" * 100)
from pathlib import Path
import io, csv, json, re, datetime, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from nyishi_asr.runtime import display

pd.set_option("display.max_columns", 60); pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)

import os
ROOT = Path(os.environ.get("BTP_ROOT", Path.cwd())).resolve()
RAW_A = ROOT / "raw" / "A_narration" / "data"                     # the untouched raw Narration data
MAN = ROOT / "manifests"; CLEAN = ROOT / "cleaning"; META = ROOT / "metadata"
for d in (MAN, CLEAN, META):
    d.mkdir(exist_ok=True)                                         # create the folder if it is missing

TODAY = datetime.date.today().isoformat()
LOG = CLEAN / "decision_log_A.csv"                                 # the plan keeps the decision log in cleaning/
LOG_COLS = ["date", "step", "decision", "reason", "rows_affected", "hours_affected"]

from nyishi_asr.runtime import log_decision

# Recovery starts a current log; no missing historical rows are fabricated.
print("\nSetup done. Folders:", MAN.name, CLEAN.name, META.name)


# Original zero-based notebook cell 4
print("=" * 100)
print("CELL N1 - LOAD master_A (version 1) AND THE RAW COLUMNS THAT WERE LEFT OUT")
print("=" * 100)
M1 = pd.read_parquet(MAN / "master_A.parquet")                     # version 1: we never overwrite it, we build version 1.1 next to it
assert len(M1) == 6718, "unexpected number of rows"
print(f"master_A (v1): {M1.shape[0]} rows x {M1.shape[1]} columns")
print("speaker_status counts (expected ok 5712, linked_to_pair 622, multi_person 208, uncertain_identity 176):")
print(M1["speaker_status"].value_counts().to_string())

# Every row says where its raw data is: "raw/A_narration/data/<split>/<file>#row=<n>". Split that into three keys.
keys = M1["audio_path"].str.extract(r"data/(?P<split>\w+)/(?P<file>[^#]+)#row=(?P<row_in_file>\d+)")
keys["row_in_file"] = keys["row_in_file"].astype(int)

# Re-read the raw table (everything except the audio bytes) so we can attach the columns that master v1 dropped.
parts = []
for f in sorted(RAW_A.glob("*/*.parquet")):
    t = pq.read_table(f, columns=[c for c in pq.ParquetFile(f).schema_arrow.names if c != "audio"]).to_pandas()
    t["split"] = f.parent.name; t["file"] = f.name; t["row_in_file"] = range(len(t))
    parts.append(t)
RAW = pd.concat(parts, ignore_index=True)
R = keys.merge(RAW, on=["split", "file", "row_in_file"], how="left", validate="one_to_one", sort=False)      # one raw row for every master row, same order as M1
assert len(R) == len(M1) and R["audio_id"].notna().all(), "some master rows could not be found in the raw data"
assert (R["audio_id"].values == M1["original_audio_id"].values).all(), "raw rows are not aligned with the master rows"
print(f"\nAll {len(R)} master rows were matched to their raw row and the audio IDs agree.")


# Original zero-based notebook cell 5
print("=" * 100)
print("CELL N2 - SPEAKER GROUPS: find one person who appears under several speaker IDs")
print("=" * 100)
print("""WHY: the same person can be recorded in two campaigns and receive two different public IDs. If we later split by speaker ID,
that person could end up in train AND test. A new column speaker_group_id merges the IDs that very probably belong to one person.
The column speaker_id itself is NOT changed.
RULE A (automatic): public IDs whose raw speakerId has the same NAME STEM (name without the age code and timestamp) are one person.
RULE B (manual)   : name stems that you judge to be spelling variants of one name. Edit the list below if you disagree.
""")
MANUAL_STEM_MERGES = [("likhaaku", "likkaaku")]     # both Female 18-30; names differ by one letter. Delete this entry if you think they are different people.

solo = M1[~M1["is_pair_recording"]].copy()                                       # pair recordings have two names in the ID, so they have no single stem
solo["stem"] = solo["speaker_id_raw"].str.replace(r"\d+(_\d+)?$", "", regex=True)  # "tajjarbo3045_22561" -> "tajjarbo"
ids_by_stem = solo.groupby("stem")["speaker_id"].unique()

parent = {p: p for p in M1["speaker_id"].unique()}                                 # union-find: every public ID starts as its own group
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)

reason = {}
for stem, ids in ids_by_stem.items():                                              # RULE A
    for other in ids[1:]:
        union(ids[0], other)
        reason[stem] = "same name stem under more than one public ID"
for a, b in MANUAL_STEM_MERGES:                                                    # RULE B
    ids = list(ids_by_stem.get(a, [])) + list(ids_by_stem.get(b, []))
    for other in ids[1:]:
        union(ids[0], other)
    reason[a + "~" + b] = "manual: look-alike names (possible spelling variants)"

members = collections.defaultdict(list)
for p in parent: members[find(p)].append(p)
stems_of = solo.groupby("speaker_id")["stem"].apply(lambda s: sorted(set(s))).to_dict()
group_name = {}
for root, ids in members.items():
    if len(ids) == 1:
        group_name[ids[0]] = ids[0]                                                # not merged: the group is just the public ID
    else:
        name = "SG_" + "_".join(sorted({s for i in ids for s in stems_of.get(i, [])}))
        for i in ids: group_name[i] = name

M1["speaker_group_id"] = M1["speaker_id"].map(group_name)
M1["group_train_only"] = M1.groupby("speaker_group_id")["train_only"].transform("max")   # if ANY member is train-only, the whole group is

def label_of(s):
    u = sorted(s.dropna().unique())
    return "unknown" if len(u) == 0 else "/".join(u)
M1["recording_month"] = pd.to_datetime(M1["recording_session_id"].str[1:9], format="%Y%m%d", errors="coerce").dt.to_period("M").astype(str)

merged = M1[M1["speaker_group_id"] != M1["speaker_id"]]
G = merged.groupby("speaker_group_id").agg(
    public_ids=("speaker_id", "nunique"), clips=("sample_id", "size"), hours=("duration_seconds", lambda x: round(x.sum() / 3600, 3)),
    gender=("gender", label_of), age=("age_group", label_of), months=("recording_month", lambda x: ",".join(sorted(set(x)))),
    status=("speaker_status", lambda x: "/".join(sorted(set(x)))))
print(f"Public speaker IDs: {M1['speaker_id'].nunique()}  ->  speaker groups: {M1['speaker_group_id'].nunique()}  (expected 60 -> 55)")
print("\nGroups that were merged (check gender/age: a group with two different genders is only expected for the unresolved pair recordings):")
display(G)
ok_groups = M1[~M1["group_train_only"]]["speaker_group_id"].nunique()
print(f"\nGroups that may go to validation/test (no train-only member): {ok_groups}   (before grouping: {M1[~M1['train_only']]['speaker_id'].nunique()} speaker IDs)")


# Original zero-based notebook cell 6
print("=" * 100)
print("CELL N3 - DECODE EVERY CLIP AND STORE THE FACTS (format, real length) - takes about 30 seconds the first time")
print("=" * 100)
FACTS = MAN / "audio_facts_A.parquet"
if FACTS.exists():
    AF = pd.read_parquet(FACTS); print("Found a saved result, loaded it:", FACTS.name)
else:
    lookup = dict(zip(zip(keys["split"], keys["file"], keys["row_in_file"]), M1["sample_id"]))
    rec = []
    for f in sorted(RAW_A.glob("*/*.parquet")):
        pf = pq.ParquetFile(f); base = 0
        for g in range(pf.num_row_groups):
            for i, a in enumerate(pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()):
                try:
                    info = sf.info(io.BytesIO(a["bytes"]))                       # reads only the WAV header: format, rate, channels, length
                    rec.append(dict(sample_id=lookup[(f.parent.name, f.name, base + i)], decodes=True, audio_subtype=info.subtype,
                                    audio_sr=info.samplerate, audio_channels=info.channels, n_frames=info.frames, duration_decoded=round(info.duration, 3)))
                except Exception as e:
                    rec.append(dict(sample_id=lookup[(f.parent.name, f.name, base + i)], decodes=False, audio_subtype=None,
                                    audio_sr=None, audio_channels=None, n_frames=0, duration_decoded=0.0))
            base += pf.metadata.row_group(g).num_rows
    AF = pd.DataFrame(rec); AF.to_parquet(FACTS, index=False); print("Saved:", FACTS)

AF = M1[["sample_id", "duration_seconds"]].merge(AF, on="sample_id", how="left", validate="one_to_one")
AF["format_ok"] = AF["decodes"] & (AF["audio_sr"] == 16000) & (AF["audio_channels"] == 1) & (AF["audio_subtype"] == "PCM_16")
AF["length_matches_stated"] = (AF["duration_decoded"] - AF["duration_seconds"]).abs() <= 0.05
AF["audio_issue"] = np.where(~AF["decodes"], "does_not_decode", np.where(AF["n_frames"] == 0, "empty_audio",
                    np.where(~AF["format_ok"], "unexpected_format", np.where(~AF["length_matches_stated"], "length_differs_from_stated", ""))))
AF["audio_ok"] = AF["audio_issue"] == ""
print("\nAudio format of all clips (expected: one row, 6718 clips, 16000 Hz, 1 channel, PCM_16):")
display(AF.groupby(["audio_subtype", "audio_sr", "audio_channels"]).size().rename("clips").reset_index())
print("Clips with an audio problem (expected 4: A_004972 truncated, A_004973 / A_006614 / A_006615 empty):")
display(AF[~AF["audio_ok"]][["sample_id", "duration_seconds", "duration_decoded", "audio_issue"]])


# Original zero-based notebook cell 7
print("=" * 100)
print("CELL N4 - BUILD master_A version 1.1 (version 1 stays untouched)")
print("=" * 100)
M2 = M1.copy()

# (1) the five original columns that version 1 dropped; the literal text 'N/A' becomes a real empty value
def clean(s): return s.where(~s.astype(str).str.strip().isin(["N/A", "", "nan", "None"]), None)
M2["education_medium_after_12"] = clean(R["speaker1-educationMediumAfter12-list"]).values
M2["education_medium_upto_12"] = clean(R["speaker1-educationMediumUpto12-list"]).values
M2["life_speaker_id"] = clean(R["speaker1-lifespeakerid"]).values
M2["speaker2_place_of_recording"] = clean(R["speaker2-placeOfRecording"]).values
M2["speaker2_type_of_place"] = clean(R["speaker2-typeOfPlace"]).values

# (2) the decoded audio facts from Cell N3
add = AF.set_index("sample_id")[["audio_subtype", "audio_sr", "audio_channels", "duration_decoded", "length_matches_stated", "audio_ok", "audio_issue"]]
M2 = M2.merge(add, left_on="sample_id", right_index=True, how="left", validate="one_to_one")
M2["duration_basis"] = "stated (end_time - start_time); the decoded length is in duration_decoded and is the one to use"

assert len(M2) == 6718 and M2["sample_id"].is_unique
unchanged = [c for c in pd.read_parquet(MAN / "master_A.parquet").columns if c != "duration_basis"]
old = pd.read_parquet(MAN / "master_A.parquet")
for c in unchanged:
    same = (old[c].fillna("<NA>").astype(str).values == M2[c].fillna("<NA>").astype(str).values).all()
    assert same, f"column {c} changed unexpectedly"
print(f"Version 1.1: {M2.shape[0]} rows x {M2.shape[1]} columns (version 1 had {old.shape[1]}).")
print("All original version-1 columns are identical (except duration_basis, which now points to duration_decoded).")
new_cols = [c for c in M2.columns if c not in old.columns]
print("\nNew columns:", new_cols)
M2.to_parquet(MAN / "master_A_v1_1.parquet", index=False)
M2.to_csv(MAN / "master_A_v1_1.csv", index=False, encoding="utf-8-sig")
back = pd.read_parquet(MAN / "master_A_v1_1.parquet")
print(f"\nSaved master_A_v1_1.parquet and .csv. Reloaded shape: {back.shape} | transcripts identical to v1: {back['transcript_raw'].equals(old['transcript_raw'])}")
print("\nFilled-in counts of the five recovered columns (rows that actually have a value):")
print(M2[["education_medium_after_12", "education_medium_upto_12", "life_speaker_id", "speaker2_place_of_recording", "speaker2_type_of_place"]].notna().sum().to_string())


# Original zero-based notebook cell 8
print("=" * 100)
print("CELL N5 - SAVE THE SPEAKER TABLES (the plan asks for the conflicting mappings to be listed and marked resolved/uncertain)")
print("=" * 100)
spk = M2.groupby("speaker_id").agg(
    speaker_group_id=("speaker_group_id", "first"), speaker_status=("speaker_status", "first"), clips=("sample_id", "size"),
    hours=("duration_decoded", lambda x: round(x.sum() / 3600, 3)), gender=("gender", label_of), age=("age_group", label_of),
    places=("place_of_recording", label_of), months=("recording_month", lambda x: ",".join(sorted(set(x)))),
    original_splits=("original_split", lambda x: "/".join(sorted(set(x)))), raw_speaker_ids=("speaker_id_raw", lambda x: " | ".join(sorted(set(x)))),
    in_pair_recording=("is_pair_recording", "max")).reset_index().sort_values("hours", ascending=False)
spk.to_csv(CLEAN / "speaker_table_A.csv", index=False, encoding="utf-8-sig")

grp = M2.groupby("speaker_group_id").agg(
    public_ids=("speaker_id", lambda x: " | ".join(sorted(set(x)))), n_public_ids=("speaker_id", "nunique"), clips=("sample_id", "size"),
    hours=("duration_decoded", lambda x: round(x.sum() / 3600, 3)), gender=("gender", label_of), age=("age_group", label_of),
    statuses=("speaker_status", lambda x: "/".join(sorted(set(x)))), train_only=("group_train_only", "max")).reset_index()
grp = grp[grp["n_public_ids"] > 1]
grp["why_merged"] = grp["speaker_group_id"].map(lambda g: "manual: look-alike names" if g == "SG_likhaaku_likkaaku" else "same name stem under >1 public ID")
grp["confidence"] = grp["why_merged"].map(lambda w: "possible (not verifiable)" if w.startswith("manual") else "high")
grp.to_csv(CLEAN / "speaker_group_table_A.csv", index=False, encoding="utf-8-sig")

# conflicts: (a) a public ID with several raw IDs / genders / ages, (b) a raw ID that maps to several public IDs
a = M2.groupby("speaker_id").agg(raw_ids=("speaker_id_raw", "nunique"), genders=("gender", lambda x: x.dropna().nunique()), ages=("age_group", lambda x: x.dropna().nunique()),
                                 clips=("sample_id", "size"), hours=("duration_decoded", lambda x: round(x.sum() / 3600, 3)), status=("speaker_status", "first"))
a = a[(a["raw_ids"] > 1) | (a["genders"] > 1) | (a["ages"] > 1)].reset_index()
a.insert(0, "conflict_type", "public ID has several raw IDs, genders or age groups")
b = M2.groupby("speaker_id_raw").agg(public_ids=("speaker_id", "nunique"), clips=("sample_id", "size"), hours=("duration_decoded", lambda x: round(x.sum() / 3600, 3)), status=("speaker_status", "first"))
b = b[b["public_ids"] > 1].reset_index()
b.insert(0, "conflict_type", "raw ID maps to several public IDs")
conf = pd.concat([a, b], ignore_index=True)
conf["resolution"] = "uncertain"
conf["handling"] = "kept in master; train-only (speaker_status)"
conf.to_csv(CLEAN / "speaker_id_conflicts_A.csv", index=False, encoding="utf-8-sig")
print(f"speaker_table_A.csv        : {len(spk)} public speaker IDs")
print(f"speaker_group_table_A.csv  : {len(grp)} merged groups")
print(f"speaker_id_conflicts_A.csv : {len(conf)} conflicts (all marked 'uncertain' and train-only)")
print("\nThe merged groups:"); display(grp[["speaker_group_id", "n_public_ids", "clips", "hours", "gender", "age", "why_merged", "confidence"]])
print("\nThe conflicts:"); display(conf[["conflict_type", "speaker_id", "speaker_id_raw", "status", "clips", "resolution"]].fillna(""))


# Original zero-based notebook cell 9
print("=" * 100)
print("CELL N6 - DATASET RECORD for Narration (the plan's Step 2 asks for this: rows, hours, columns, license, limitations)")
print("=" * 100)
prov = json.loads((META / "provenance.json").read_text(encoding="utf-8"))["A_narration"]
readme = (ROOT / "raw" / "A_narration" / "README.md").read_text(encoding="utf-8")
license_ = re.search(r"^license:\s*(.+)$", readme, re.M).group(1).strip()
language = re.search(r"^language:\s*\n-\s*(\w+)", readme, re.M).group(1)
schema = pq.ParquetFile(next(RAW_A.glob("*/*.parquet"))).schema_arrow
cols = {f.name: str(f.type) for f in schema}
splits = M2.groupby("original_split").agg(clips=("sample_id", "size"), hours=("duration_decoded", lambda x: round(x.sum() / 3600, 3))).round(3)
top5 = M2.groupby("speaker_id")["duration_decoded"].sum().sort_values(ascending=False)
spk_split = M2.groupby("speaker_id")["original_split"].nunique()
limits = [
  "Transcripts are IPA only (column sentence-IPA-transcription); the card promises Roman script too, but no such column exists.",
  f"The card claims over 200 hours; the files contain {M2['duration_decoded'].sum()/3600:.2f} hours.",
  f"{int(M2['transcript_missing'].sum())} clips ({M2.loc[M2['transcript_missing'], 'duration_decoded'].sum()/3600:.2f} h) have the transcript N/A, including clips in the authors' test and validation splits.",
  f"The authors' splits are not speaker-disjoint: {int((spk_split > 1).sum())} of {len(spk_split)} speaker IDs occur in more than one split.",
  f"The five biggest speakers hold {100*top5.head(5).sum()/top5.sum():.0f}% of all audio; {100*M2.loc[M2['age_group']=='18-30','duration_decoded'].sum()/M2['duration_decoded'].sum():.0f}% of the audio is from speakers aged 18-30.",
  f"{int((M2['duration_decoded'] > 30).sum())} clips ({M2.loc[M2['duration_decoded'] > 30, 'duration_decoded'].sum()/3600:.2f} h) are longer than 30 s.",
  "Speaker IDs are inconsistent for some speakers (see cleaning/speaker_id_conflicts_A.csv) and the same person can have several public IDs (see speaker_group_table_A.csv).",
  "The card contains no citation, no stated limitations and no annotation guideline.",
]
record = dict(dataset="Narration (Dataset A)", repo=prov["repo"], revision=prov["revision"], accessed_utc=prov["accessed_utc"], license=license_, language=language,
              rows=len(M2), hours_decoded=round(M2["duration_decoded"].sum() / 3600, 3), hours_stated=round(M2["duration_seconds"].sum() / 3600, 3),
              original_splits=splits.to_dict("index"), columns=cols, audio_format="WAV, PCM 16-bit, 16 kHz, mono (all clips)",
              source_of_audio_and_transcripts="Karya / Atekho recordings, transcribed and exported with MATra Lab (LiFE Suite), per the dataset card",
              known_limitations=limits)
(META / "dataset_record_A.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
md = [f"# Dataset record - Narration (Dataset A)", "", f"- Repository: `{prov['repo']}`", f"- Revision (commit): `{prov['revision']}`", f"- Accessed (UTC): {prov['accessed_utc']}",
      f"- License (from the card): {license_}", f"- Language tag: {language}", f"- Rows: {len(M2)} | hours (decoded): {record['hours_decoded']} | hours (stated): {record['hours_stated']}",
      f"- Audio: {record['audio_format']}", f"- Source: {record['source_of_audio_and_transcripts']}", "", "## Original splits", "", "| split | clips | hours |", "|---|---|---|"]
md += [f"| {k} | {v['clips']} | {v['hours']} |" for k, v in record["original_splits"].items()]
md += ["", "## Columns in the raw files", ""] + [f"- `{k}`: {v}" for k, v in cols.items()] + ["", "## Known limitations (from our own audit)", ""] + [f"- {x}" for x in limits]
(META / "dataset_record_A.md").write_text("\n".join(md), encoding="utf-8")
print(f"License: {license_} | language: {language} | revision: {prov['revision'][:10]}...")
print(f"Columns recorded: {len(cols)} | original splits: {splits['clips'].to_dict()}")
print("\nKnown limitations that were written down:")
for x in limits: print("  -", x)
print("\nSaved metadata/dataset_record_A.json and metadata/dataset_record_A.md")


# Original zero-based notebook cell 10
print("=" * 100)
print("CELL N7 - RECORD THE DECISIONS IN THE LOG (safe to run several times)")
print("=" * 100)
n_merged_rows = int((M2["speaker_group_id"] != M2["speaker_id"]).sum()); h_merged = round(M2.loc[M2["speaker_group_id"] != M2["speaker_id"], "duration_decoded"].sum() / 3600, 3)
bad = M2[~M2["audio_ok"]]
long_ = M2[M2["duration_decoded"] > 30]
miss = M2[M2["transcript_missing"]]
log_decision("3", "speaker_group_id merges public IDs that share a name stem (same person in several campaigns/sessions) plus the manual look-alike merge likhaaku~likkaaku; train/validation/test must be split by speaker_group_id, not speaker_id",
             "avoid one person appearing in train and test under two IDs; 60 public IDs -> 55 groups", n_merged_rows, h_merged)
log_decision("3", "master_A_v1_1 keeps all version-1 columns and adds the 5 dropped original columns, recording_month and the decoded-audio columns",
             "the plan says to retain all original fields; decoded facts belong in the master table", len(M2), round(M2["duration_decoded"].sum() / 3600, 3))
log_decision("3", "duration_decoded (from the WAV header of every clip) is the duration to use from now on; all 6718 clips are 16 kHz mono PCM_16",
             "closes the open 'validate durations' item", len(M2), round(M2["duration_decoded"].sum() / 3600, 3))
log_decision("3", "4 clips flagged audio_ok=False (A_004972 truncated; A_004973, A_006614, A_006615 empty); to be excluded from experiments",
             "audio shorter than stated or empty although a transcript exists", len(bad), round(bad["duration_seconds"].sum() / 3600, 3))
log_decision("3", "the SNR-based audio-quality proxy is NOT used to filter or label data",
             "it did not detect the car noise heard by the user and its thresholds were never validated; noise is handled by listening (audit) instead", None, None)
log_decision("3", "clips longer than 30 s are only flagged, not deleted; the cutoff is decided per model later",
             "OmniASR CTC allows up to 60 s, Whisper needs 30 s; re-segmentation may recover them", len(long_), round(long_["duration_decoded"].sum() / 3600, 3))
log_decision("3", "clips with transcript N/A keep transcript_missing=True; they are excluded from supervised training and evaluation but their audio is kept",
             "no reference text; the audio may be used later for pseudo-labelling", len(miss), round(miss["duration_decoded"].sum() / 3600, 3))
print("\nThe decision log now contains:")
display(pd.read_csv(LOG, encoding="utf-8-sig")[["step", "decision", "rows_affected", "hours_affected"]])


# Original zero-based notebook cell 11
print("=" * 100)
print("CELL N8 - FINAL CHECKS ON WHAT IS SAVED ON DISK")
print("=" * 100)
chk = pd.read_parquet(MAN / "master_A_v1_1.parquet")
log = pd.read_csv(LOG, encoding="utf-8-sig")
tests = {
  "6718 rows, unique sample_id": len(chk) == 6718 and chk["sample_id"].is_unique,
  "every row has a speaker_group_id": chk["speaker_group_id"].notna().all(),
  "no speaker_id is split across two groups": chk.groupby("speaker_id")["speaker_group_id"].nunique().max() == 1,
  "group_train_only is True whenever train_only is True": bool((chk.loc[chk["train_only"], "group_train_only"]).all()),
  "decoded duration present for every row": chk["duration_decoded"].notna().all(),
  "exactly 4 clips with audio_ok=False": int((~chk["audio_ok"]).sum()) == 4,
  "transcript_raw unchanged": chk["transcript_raw"].equals(pd.read_parquet(MAN / "master_A.parquet")["transcript_raw"]),
  "decision log has no duplicate decisions": not log["decision"].duplicated().any(),
  "speaker tables and dataset record exist": all(p.exists() for p in [CLEAN / "speaker_table_A.csv", CLEAN / "speaker_group_table_A.csv", CLEAN / "speaker_id_conflicts_A.csv", META / "dataset_record_A.json", META / "dataset_record_A.md"]),
}
for k, v in tests.items(): print(("PASS  " if v else "FAIL  ") + k)
print("\nOverview (decoded hours):")
display(chk.groupby("speaker_status").agg(clips=("sample_id", "size"), speaker_ids=("speaker_id", "nunique"), speaker_groups=("speaker_group_id", "nunique"),
                                          hours=("duration_decoded", lambda x: round(x.sum() / 3600, 3))))
assert all(tests.values()), "at least one check failed - read the FAIL lines above"
print("\nAll checks passed. master_A_v1_1 is the table to use from now on.")
