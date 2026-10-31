"""Deterministic recovered a_v2 workflow. Execute via nyishi_asr.restore."""


# Original zero-based notebook cell 23
print("=" * 100)
print("CELL V0 - SETUP FOR POLICY A_v2: what changes compared with A_v1, the evidence, and your confirmation")
print("=" * 100)
from pathlib import Path
import csv, datetime, re, unicodedata as ud, collections
import numpy as np
import pandas as pd
from nyishi_asr.runtime import display

pd.set_option("display.max_columns", 60); pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 110)
import os
ROOT = Path(os.environ.get("BTP_ROOT", Path.cwd())).resolve()
MAN = ROOT / "manifests"; CLEAN = ROOT / "cleaning"; META = ROOT / "metadata"
TODAY = datetime.date.today().isoformat()
LOG = CLEAN / "decision_log_A.csv"
LOG_COLS = ["date", "step", "decision", "reason", "rows_affected", "hours_affected"]

from nyishi_asr.runtime import log_decision

# --- what A_v2 starts from: the table with the A_v1 result (frozen, never modified) ---
for p in [MAN / "master_A_v1_1.parquet", MAN / "master_A_v1_2.parquet", CLEAN / "normalization_policy_A_v1.csv", LOG]:
    assert p.exists(), f"missing file: {p}"
MV = pd.read_parquet(MAN / "master_A_v1_2.parquet")
assert len(MV) == 6718 and MV["sample_id"].is_unique
HAS = ~MV["transcript_missing"]
RAW = MV.loc[HAS, "transcript_raw"]; N1 = MV.loc[HAS, "transcript_normalized"]       # raw text, and the A_v1 result
HRS = MV["duration_decoded"] / 3600
print(f"Loaded master_A_v1_2 (the A_v1 result): {len(MV)} clips, {int(HAS.sum())} with a transcript ({HRS[HAS].sum():.2f} h)")

# --- the evidence for the three punctuation marks (looked at in the A_v1 result) ---
def where(sym):
    """For one symbol: occurrences, transcripts, position inside the word, how often a combining mark follows it, and in how many of its
    transcripts the REAL glottal stop 'ʔ' also occurs."""
    n = tr = fin = ini = mid = mark = with_glottal = 0
    for t in N1:
        if sym not in t: continue
        tr += 1; with_glottal += ("\u0294" in t)
        for i, c in enumerate(t):
            if c != sym: continue
            n += 1
            nxt = t[i + 1] if i + 1 < len(t) else " "; prv = t[i - 1] if i > 0 else " "
            if nxt == " ": fin += 1
            elif prv == " ": ini += 1
            else: mid += 1
            mark += ud.category(nxt).startswith("M")
    return dict(symbol=sym, occurrences=n, transcripts=tr, word_final=fin, word_initial=ini, inside_word=mid, followed_by_combining_mark=mark, transcripts_with_real_glottal_stop=with_glottal)
EV = pd.DataFrame([where(s) for s in (".", ",", "?")]).set_index("symbol")
print("\nWhere do the three marks stand? (counted in the A_v1 result)")
display(EV)
print("""How to read it:
  '.'  every full stop is INSIDE a word, most often between two identical letters (kon.na, pjop.pa): that is the IPA SYLLABLE BREAK. It is used
       inconsistently (some of the same words are written without it), and the audio cannot tell a reader where the syllable break is.
  ','  a comma inside a word (mostly after ɡ, right before a suffix: hoɡ,e) or at the end of a word (ordinary clause punctuation). No IPA symbol looks like a comma.
  '?'  several question marks are followed by a combining mark (h-ə-b-?-mark) and some stand in transcripts that also contain the real glottal stop 'ʔ'.
       Punctuation never carries a combining mark, so many of them are probably a TYPED 'ʔ'. Deleting them would erase sounds.""")

CHANGES = pd.DataFrame([
 ("NORM_010", "full stop between two symbols of a word", "unchanged (pending)", "DELETED (provisional)", "syllable-break marker, inconsistent, no sound"),
 ("NORM_019", "comma (NEW rule)", "unchanged (pending, inside NORM_012)", "DELETED (provisional)", "no IPA reading; clause punctuation or stray marker"),
 ("NORM_011", "question mark", "unchanged (pending)", "STAYS UNCHANGED and flagged", "probably a typed glottal stop: only the maintainers can say"),
 ("NORM_012", "other punctuation  \"  =  '  \\", "unchanged (pending)", "STAYS UNCHANGED and flagged", "one occurrence each: too few for a rule"),
 ("NORM_005/006", "vowel set for the length rules", "no ʉ", "ʉ added (one shared set)", "ʉ is a vowel; no ':' or ';' follows it, so no output changes"),
 ("NORM_007", "'[' -> dental mark", "checks only the letter right before '['", "skips accents in between, like the length rules", "no output changes (all 28 '[' follow t/d directly)"),
 ("NORM_008", "g -> ɡ exception words", "exact token match", "match survives attached punctuation", "no output changes (no exception word has punctuation glued to it)"),
], columns=["rule", "what", "A_v1", "A_v2", "why"])
print("\nWHAT CHANGES IN POLICY A_v2 (everything else is identical to A_v1; rules NORM_013-NORM_018 are not touched):")
display(CHANGES)
print("Note: compared with the table I proposed in chat, the question mark is NOT deleted any more: the evidence above shows that it is probably a glottal stop.")

CONFIRM_A_V2 = True      # <- you confirmed this list in chat ("yes"). Set it to False if you want to stop here.
assert CONFIRM_A_V2, "A_v2 was not confirmed: set CONFIRM_A_V2 = True after reading the list above"
print("\nConfirmed: policy A_v2 will be created. A_v1 and master_A_v1_2 stay exactly as they are (frozen).")


# Original zero-based notebook cell 24
from nyishi_asr.norm_a_v2 import *
print("=" * 100)
print("CELL V1 - THE A_v2 RULES AS FUNCTIONS, AND A DRY RUN (nothing is saved, no transcript is changed)")
print("=" * 100)
print("""All names here end in '2' on purpose: the A_v1 functions (PIPELINE, VOWELS, ...) stay untouched, so the A_v1 cells can still be re-run.
The rules run in this order:  NFD -> delete full stops inside words -> delete commas -> trim -> collapse spaces -> ':' -> ';' -> '[' -> g -> lowercase -> NFD.
""")




# --- dry run on a COPY of the raw transcripts ---
stats2, examples2, exc_detail2 = {}, collections.defaultdict(list), collections.defaultdict(list)
ids = MV.loc[HAS, "sample_id"].tolist(); hrs_t = HRS[HAS].to_numpy()
cur = RAW.tolist()
for rid, fn in PIPELINE2:
    res = [fn(t) for t in cur]
    ch = np.array([r[1] for r in res]); ex = np.array([r[2] for r in res])
    stats2[rid] = dict(rows=int((ch > 0).sum()), occurrences=int(ch.sum()), exceptions=int(ex.sum()), exception_rows=int((ex > 0).sum()), hours=round(float(hrs_t[ch > 0].sum()), 3))
    for k in np.where(ch > 0)[0][:3]:
        b_, a_ = cur[k], res[k][0]; bt, at = b_.split(" "), a_.split(" ")
        d = [(x, y) for x, y in zip(bt, at) if x != y][:1] if len(bt) == len(at) else []
        examples2[rid].append(f"{ids[k]}: {d[0][0]} -> {d[0][1]}" if d else f"{ids[k]}: {b_[:30]} -> {a_[:30]}")
    for k in np.where(ex > 0)[0]: exc_detail2[rid] += [(ids[k], dd) for dd in res[k][3]]
    cur = [r[0] for r in res]
DRY2 = pd.Series(cur, index=ids)
S2 = pd.DataFrame(stats2).T.astype({"rows": int, "occurrences": int, "exceptions": int, "exception_rows": int}); S2.index.name = "rule"
print("Dry-run result per rule (rows = transcripts changed; occurrences = symbols changed or deleted; hours = audio of those transcripts, counted once):")
display(S2)

# --- the A_v2 text must differ from the A_v1 text ONLY by the deleted full stops / commas ---
same_as_v1 = [v1_to_v2(a) == b for a, b in zip(N1, DRY2)]
print(f"\nA_v2 text == A_v1 text with only the full stops inside words and the commas removed: {sum(same_as_v1)} of {len(same_as_v1)} transcripts")
assert all(same_as_v1), "A_v2 changes something else than the planned deletions: stop and look at the differing transcripts"
rep1 = pd.read_csv(CLEAN / "normalization_change_report_A_v1.csv", encoding="utf-8-sig", keep_default_na=False).set_index("rule_id")
same_nums = {r: (int(rep1.loc[r, "rows_affected"]), int(rep1.loc[r, "occurrences_changed"])) == (stats2[r]["rows"], stats2[r]["occurrences"]) for r in ["NORM_003", "NORM_005", "NORM_006", "NORM_007", "NORM_008", "NORM_009"]}
print("Rules that existed in A_v1 give the same numbers as before (rows and symbols):", same_nums)
assert all(same_nums.values())
vow_after = sum(1 for t in RAW for i, c in enumerate(t) if c in ":;" and prev_base2(t, i) == "\u0289")
print(f"ʉ added to the vowel set: {vow_after} ':' or ';' follow a ʉ, so this does not change any output (it only makes the set consistent).")

print("\nBefore -> after examples:")
for rid in ("NORM_010", "NORM_019"): print(" ", rid, "|", " ;  ".join(examples2[rid][:4]))
print("Full stops left unchanged because they are NOT inside a word:", len(exc_detail2["NORM_010"]))
print("Unchanged exceptions of the other rules:", {r: len(exc_detail2[r]) for r in ("NORM_006", "NORM_007", "NORM_008")})


# Original zero-based notebook cell 25
print("=" * 100)
print("CELL V2 - THE A_v2 POLICY TABLE  ->  cleaning/normalization_policy_A_v2.csv")
print("=" * 100)
print("""The A_v2 table is the A_v1 table with the changes listed in V0. Nothing in the transcripts is changed here.
Statuses: Approved = safe technical clean-up | Provisional = supported by evidence, NOT confirmed by the maintainers | Pending review | Rejected.
""")
REVIEWER = ""                      # <- type your name here if you want it recorded as the person who approves A_v2
POLICY_VERSION2 = "A_v2"
POLICY_FILE2 = CLEAN / "normalization_policy_A_v2.csv"
pol1 = pd.read_csv(CLEAN / "normalization_policy_A_v1.csv", encoding="utf-8-sig", keep_default_na=False)
POL2 = pol1.copy().set_index("rule_id")

def sym_info2(chars):
    """Transcripts containing any of the symbols, total occurrences, and the audio hours of those transcripts (in the A_v2 dry-run text)."""
    mask = DRY2.map(lambda s: any(c in s for c in chars)); joined = "".join(DRY2)
    return int(mask.sum()), sum(joined.count(c) for c in chars), round(float(HRS[HAS][mask.to_numpy()].sum()), 3)

# --- evidence for the changed rules, computed from the data ---
dots = [w for t in N1 for w in t.split(" ") if "." in w]
vocab_all = set(w2 for t in N1 for w2 in t.split(" "))
dotless = sum(1 for w in dots if "".join(w.split(".")) in vocab_all)
geminate = sum(1 for w in dots if any(a[-1:] == b[:1] for a, b in zip(w.split("."), w.split(".")[1:])))
E_dot, E_com, E_q = EV.loc["."], EV.loc[","], EV.loc["?"]
VOWEL_TXT2 = "".join(sorted(VOWELS2))
OP2 = sym_info2(['"', "=", "'", "\\"])

def setrow(rid, **kw):
    for k, v in kw.items(): POL2.loc[rid, k] = v
setrow("NORM_005", condition=f"only right after a vowel (nearest preceding letter, accents skipped); vowels: {VOWEL_TXT2}")
setrow("NORM_006", condition=f"only right after a vowel; any other semicolon is left unchanged and flagged; vowels: {VOWEL_TXT2}")
setrow("NORM_007", condition="only right after t or d (accents between the letter and '[' are skipped; if the letter already has the dental mark the '[' is left unchanged and flagged)")
setrow("NORM_008", condition="every g, except in the words of the exception list (ordinary English spellings; matched even if punctuation is glued to the word): " + ", ".join(sorted(G_EXCEPTION_WORDS2)))
setrow("NORM_010", replacement="(deleted)", condition="only a full stop between two symbols of one word (the IPA syllable break); any other full stop is left unchanged and flagged",
       evidence=f"{int(E_dot.occurrences)} occurrences in {int(E_dot.transcripts)} transcripts, all inside words (0 word-final, 0 word-initial); {geminate} of {len(dots)} dotted words have the dot between identical letters (kon.na); for {dotless} of {len(dots)} the same word also occurs without the dot: an inconsistently used syllable break that the audio cannot show",
       linguistic_risk="medium (information about syllabification is dropped)", status="Provisional", action="Apply and log", examples=" ; ".join(examples2["NORM_010"][:2]))
setrow("NORM_011", evidence=f"{int(E_q.occurrences)} occurrences: {int(E_q.inside_word)} inside a word, {int(E_q.followed_by_combining_mark)} directly followed by a combining mark, {int(E_q.transcripts_with_real_glottal_stop)} in transcripts that also contain the real glottal stop ʔ: in many cases probably a typed ʔ. NOT deleted and NOT converted: only the maintainers can say",
       action="Preserve and flag")
setrow("NORM_012", issue="Other ASCII punctuation (comma moved to NORM_019)", source_pattern="\" = ' \\", evidence="one occurrence each (A_001474, A_001275, A_005052, A_006383); meaning not established; the apostrophe in mɪlɪ'ŋŋa may be a glottal stop",
       action="Preserve and flag")
POL2["policy_version"] = POLICY_VERSION2; POL2["review_date"] = TODAY; POL2["reviewer"] = REVIEWER
POL2.loc["NORM_019"] = pd.Series(dict(issue="Comma", source_pattern=",", replacement="(deleted)", condition="every comma",
       examples=" ; ".join(examples2["NORM_019"][:2]), evidence=f"{int(E_com.occurrences)} occurrences in {int(E_com.transcripts)} transcripts: {int(E_com.inside_word)} inside a word (mostly after ɡ, before a suffix: hoɡ,e) and {int(E_com.word_final)} at the end of a word (clause punctuation); no IPA symbol looks like a comma",
       linguistic_risk="medium (a stray marker might carry a meaning nobody has documented)", status="Provisional", action="Apply and log", reviewer=REVIEWER, review_date=TODAY, policy_version=POLICY_VERSION2,
       dry_run_rows=0, dry_run_occurrences=0, dry_run_exceptions=0, dry_run_hours=0.0))
# dry-run columns: the numbers of THIS dry run for every rule that is applied; for the preserved symbols the counts of the A_v2 text
for rid in ["NORM_001", "NORM_002", "NORM_003", "NORM_005", "NORM_006", "NORM_007", "NORM_008", "NORM_009", "NORM_010", "NORM_019"]:
    s = stats2[rid]; POL2.loc[rid, ["dry_run_rows", "dry_run_occurrences", "dry_run_exceptions", "dry_run_hours"]] = [s["rows"], s["occurrences"], s["exceptions"], s["hours"]]
POL2.loc["NORM_011", ["dry_run_rows", "dry_run_occurrences", "dry_run_exceptions", "dry_run_hours"]] = [*sym_info2(["?"])[:2], 0, sym_info2(["?"])[2]]
POL2.loc["NORM_012", ["dry_run_rows", "dry_run_occurrences", "dry_run_exceptions", "dry_run_hours"]] = [*OP2[:2], 0, OP2[2]]
POL2 = POL2.reset_index()[list(pol1.columns)].sort_values("rule_id").reset_index(drop=True)
for c in ["dry_run_rows", "dry_run_occurrences", "dry_run_exceptions"]: POL2[c] = POL2[c].astype(int)
POL2["dry_run_hours"] = POL2["dry_run_hours"].astype(float)

KEY = ["rule_id", "source_pattern", "replacement", "condition", "status", "action"]
if POLICY_FILE2.exists():
    old2 = pd.read_csv(POLICY_FILE2, encoding="utf-8-sig", keep_default_na=False)
    if old2[KEY].astype(str).reset_index(drop=True).equals(POL2[KEY].astype(str).reset_index(drop=True)):
        print(f"{POLICY_FILE2.name} already exists and its rules are identical: NOT rewritten (a frozen version is never edited).")
    else:
        alt = CLEAN / "normalization_policy_A_v2_PROPOSED_CHANGES.csv"; POL2.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"{POLICY_FILE2.name} exists and the rules DIFFER. It was not overwritten; the new version was saved as {alt.name}. A change after A_v2 is applied means A_v3.")
else:
    POL2.to_csv(POLICY_FILE2, index=False, encoding="utf-8-sig"); print("Saved:", POLICY_FILE2)
print("\nThe A_v2 policy table (key columns):")
display(POL2[["rule_id", "issue", "source_pattern", "replacement", "status", "action", "dry_run_rows", "dry_run_occurrences", "dry_run_exceptions", "dry_run_hours"]])
print("\nRules that differ from A_v1 (what the evidence says):")
for rid in ["NORM_010", "NORM_011", "NORM_012", "NORM_019"]: print(f"  {rid} [{POL2.loc[POL2.rule_id == rid, 'status'].iloc[0]}] {POL2.loc[POL2.rule_id == rid, 'evidence'].iloc[0]}")
print("\nStatus counts:", POL2["status"].value_counts().to_dict())


# Original zero-based notebook cell 26
print("=" * 100)
print("CELL V3 - APPLY A_v2: build transcript_normalized (transcript_raw is NOT touched)  ->  master_A_v1_3")
print("=" * 100)
print("""Every transcript goes through the A_v2 pipeline on a working copy; we remember which rule changed what. The result goes into a NEW file
(master_A_v1_3). master_A_v1_2 (the A_v1 result) and master_A_v1_1 are not modified. The three normalization columns are rebuilt under A_v2.
In the change list, 'n' = number of SYMBOLS changed or deleted (NORM_001/002: 1 if the transcript changed; NORM_003: number of spaces removed).
""")
assert CONFIRM_A_V2, "run cell V0 first"
assert MV["transcript_raw"].map(lambda x: isinstance(x, str)).all()
def example_pair2(rid, before, after):
    if rid == "NORM_003":
        i = before.find("  "); b = before[max(0, i - 12): i + 14]; return b.replace("  ", "\u2423\u2423"), re.sub(" {2,}", " ", b)
    bt, at = before.split(" "), after.split(" ")
    if len(bt) == len(at):
        d = [(x, y) for x, y in zip(bt, at) if x != y]
        if d: return d[0]
    return before[:30], after[:30]

norm2, applied2, excs2, changes2 = [], [], [], []
for sid, raw, miss in zip(MV["sample_id"], MV["transcript_raw"], MV["transcript_missing"]):
    if miss: norm2.append(None); applied2.append(None); excs2.append(None); continue
    cur, a, e = raw, [], []
    for rid, fn in PIPELINE2:
        new, ch, ex, det = fn(cur)
        if ch > 0:
            a.append(rid); b_, a_ = example_pair2(rid, cur, new)
            changes2.append(dict(sample_id=sid, rule_id=rid, kind="changed", n=ch, before=b_, after=a_))
        if ex > 0:
            e.append(f"{rid}:{ex}"); changes2.append(dict(sample_id=sid, rule_id=rid, kind="exception", n=ex, before=" | ".join(det), after=""))
        cur = new
    norm2.append(cur); applied2.append(";".join(a) or None); excs2.append(";".join(e) or None)
CH2 = pd.DataFrame(changes2)
M3 = MV.copy()
M3["transcript_normalized"] = norm2
M3["normalization_policy_version"] = np.where(M3["transcript_missing"], None, POLICY_VERSION2)
M3["normalization_rules_applied"] = applied2
M3["normalization_exceptions"] = excs2

OUT13 = MAN / "master_A_v1_3.parquet"; NEWCOLS = ["sample_id", "transcript_normalized", "normalization_policy_version", "normalization_rules_applied", "normalization_exceptions"]
if OUT13.exists():
    prev = pd.read_parquet(OUT13)
    assert prev[NEWCOLS].fillna("<NA>").astype(str).equals(M3[NEWCOLS].fillna("<NA>").astype(str)), "master_A_v1_3 already exists with DIFFERENT content. A_v2 is frozen: a change means A_v3."
    print("master_A_v1_3 already exists and is identical to the result of this run: nothing was rewritten.")
else:
    M3.to_parquet(OUT13, index=False); M3.to_csv(MAN / "master_A_v1_3.csv", index=False, encoding="utf-8-sig")
    CH2.to_csv(CLEAN / "normalization_row_changes_A_v2.csv", index=False, encoding="utf-8-sig")
    print("Saved: master_A_v1_3.parquet / .csv and cleaning/normalization_row_changes_A_v2.csv")

has2 = ~M3["transcript_missing"]
changed2 = has2 & (M3["transcript_normalized"] != M3["transcript_raw"])
changed_vs_v1 = has2 & (M3["transcript_normalized"] != MV["transcript_normalized"])
print(f"\nTranscripts normalized                         : {int(has2.sum())}")
print(f"  changed by >= 1 rule compared with the RAW text : {int(changed2.sum())}  ({HRS[changed2].sum():.3f} h)    (A_v1 had 2,487)")
print(f"  differing from the A_v1 text (full stops/commas): {int(changed_vs_v1.sum())}  ({HRS[changed_vs_v1].sum():.3f} h)")
print(f"Clips without transcript (empty)                : {int((~has2).sum())}")
print("\nFive random transcripts that A_v2 changed compared with A_v1:")
for _, r in M3[changed_vs_v1].sample(5, random_state=2).iterrows():
    a1 = MV.loc[MV.sample_id == r["sample_id"], "transcript_normalized"].iloc[0]
    diff = [(x, y) for x, y in zip(a1.split(" "), r["transcript_normalized"].split(" ")) if x != y][:3]       # the words that differ (the deletions never change the number of words)
    print(f"  {r['sample_id']} | A_v2 rules: {r['normalization_rules_applied']} | changed words (A_v1 -> A_v2): " + " ; ".join(f"{x} -> {y}" for x, y in diff))


# Original zero-based notebook cell 27
print("=" * 100)
print("CELL V4 - VALIDATE THE A_v2 RESULT (every check must pass before we trust the new column)")
print("=" * 100)
print("""(1) Consistency checks on the whole table. (2) A CHARACTER-LEVEL AUDIT that is written INDEPENDENTLY of the rule functions (regular expressions instead of
loops): every changed character must be one that a rule may change, where the rule applies, and every place where a rule applies must really have been changed.
Passing shows that the rules were applied correctly. It does NOT show that they are linguistically right: that needs the maintainers or a Nyishi speaker.
""")
chk2 = pd.read_parquet(MAN / "master_A_v1_3.parquet")
t_raw2, t_new2 = chk2.loc[has2, "transcript_raw"], chk2.loc[has2, "transcript_normalized"]
allnew2, allraw2 = "".join(t_new2), "".join(t_raw2)
comb = lambda s: sum(ud.category(c).startswith("M") for c in s)
exc_counts2 = {r: int(CH2[(CH2.rule_id == r) & (CH2.kind == "exception")]["n"].sum()) for r in ("NORM_005", "NORM_006", "NORM_007", "NORM_008", "NORM_010")}
ALLOWED2 = {(":", "\u02d0"), (";", "\u02d0"), ("[", "\u032a"), ("g", "\u0261")}

def char_audit2(raw, new):
    """Audit ONE transcript. Returns a list of problems (empty = every change is allowed and nothing that should change was missed)."""
    base = ud.normalize("NFD", raw)
    base = re.sub(r"(?<=\S)\.(?=\S)", "", base)                 # the planned deletions, written with regular expressions
    base = base.replace(",", "")
    base = re.sub(" {2,}", " ", base.strip())                   # the only other step that changes the length
    if len(base) != len(new): return ["length differs after the planned deletions and the removal of repeated spaces"]
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^ ]+", base)]
    def in_exception(i): return any(s <= i < e and w.strip(PUNCT_EDGE).lower() in G_EXCEPTION_WORDS2 for s, e, w in spans)
    def bracket_applies(i):
        j = i - 1
        while j >= 0 and ud.category(base[j]).startswith("M"): j -= 1
        return j >= 0 and base[j] in ("t", "d") and "\u032a" not in base[j + 1: i]
    problems = []
    for i, (a, b) in enumerate(zip(base, new)):
        if a == b:
            if a in ":;" and prev_base2(base, i) in VOWELS2: problems.append(f"{a!r} after a vowel was not converted (position {i})")
            elif a == "[" and bracket_applies(i): problems.append(f"'[' after t/d was not converted (position {i})")
            elif a == "g" and not in_exception(i): problems.append(f"g outside the exception list was not converted (position {i})")
            elif "A" <= a <= "Z": problems.append(f"a capital letter was not lowercased (position {i})")
        elif (a, b) in ALLOWED2:
            if a in ":;" and prev_base2(base, i) not in VOWELS2: problems.append(f"{a!r} converted although no vowel precedes it (position {i})")
            elif a == "[" and not bracket_applies(i): problems.append(f"'[' converted although the condition does not hold (position {i})")
            elif a == "g" and in_exception(i): problems.append(f"g converted inside an exception word (position {i})")
        elif "A" <= a <= "Z" and b == a.lower(): pass
        else: problems.append(f"UNEXPECTED change {a!r} -> {b!r} (position {i})")
    return problems
audit2 = {sid: char_audit2(r, n) for sid, r, n in zip(chk2.loc[has2, "sample_id"], t_raw2, t_new2)}
bad_audit2 = {k: v for k, v in audit2.items() if v}
glued2 = [(sid, w) for sid, t in zip(chk2.loc[has2, "sample_id"], t_raw2) for w in t.split(" ") if "g" in w and w.strip(PUNCT_EDGE).lower() in G_EXCEPTION_WORDS2 and w.lower() not in G_EXCEPTION_WORDS2]

old11 = pd.read_parquet(MAN / "master_A_v1_1.parquet")
rule_ok2 = {
  "NORM_001": bool((t_new2 == t_new2.map(lambda s: ud.normalize("NFD", s))).all()),
  "NORM_002": bool(not t_new2.str.contains(r"^ | $", regex=True).any()),
  "NORM_003": bool(not t_new2.str.contains("  ", regex=False).any()),
  "NORM_005": allnew2.count(":") == exc_counts2["NORM_005"],
  "NORM_006": allnew2.count(";") == exc_counts2["NORM_006"],
  "NORM_007": allnew2.count("[") == exc_counts2["NORM_007"] and comb(allnew2) == comb(allraw2) + int(CH2[(CH2.rule_id == "NORM_007") & (CH2.kind == "changed")]["n"].sum()),
  "NORM_008": all(w.strip(PUNCT_EDGE).lower() in G_EXCEPTION_WORDS2 for t in t_new2 for w in t.split(" ") if "g" in w),
  "NORM_009": bool(not t_new2.str.contains(r"[A-Z]", regex=True).any()),
  "NORM_010": bool(not t_new2.str.contains(r"(?<=\S)\.(?=\S)", regex=True).any()) and allnew2.count(".") == exc_counts2["NORM_010"],
  "NORM_019": "," not in allnew2,
}
in_pipeline2 = {rid for rid, _ in PIPELINE2 if rid != "NORM_001b"}
agree2 = {}
for r in sorted(in_pipeline2):
    c = CH2[(CH2.rule_id == r) & (CH2.kind == "changed")]; p = POL2[POL2.rule_id == r].iloc[0]
    hrs = float(HRS[MV["sample_id"].isin(c["sample_id"]).to_numpy()].sum())
    agree2[r] = bool(c["sample_id"].nunique() == p["dry_run_rows"] and int(c["n"].sum()) == p["dry_run_occurrences"]
                     and int(CH2[(CH2.rule_id == r) & (CH2.kind == "exception")]["n"].sum()) == p["dry_run_exceptions"] and abs(hrs - float(p["dry_run_hours"])) < 0.0015)
checks2 = {
  "transcript_raw is identical to version 1.2 (and 1.1)": chk2["transcript_raw"].equals(MV["transcript_raw"]) and chk2["transcript_raw"].equals(old11["transcript_raw"]),
  "every transcript has a normalized version": bool(t_new2.notna().all() and (t_new2.str.len() > 0).all()),
  "clips without a transcript stay empty": bool(chk2.loc[~has2, "transcript_normalized"].isna().all()),
  "all other columns of version 1.2 are unchanged": all(chk2[c].fillna("<NA>").astype(str).equals(MV[c].fillna("<NA>").astype(str)) for c in MV.columns if c not in NEWCOLS),
  "the text is valid Unicode (UTF-8 round trip)": all(s.encode("utf-8").decode("utf-8") == s for s in t_new2),
  "combining marks: none removed, exactly the ones from '[' added": comb(allnew2) == comb(allraw2) + int(CH2[(CH2.rule_id == "NORM_007") & (CH2.kind == "changed")]["n"].sum()),
  "word count unchanged: no word was merged, split or lost": int(t_raw2.str.split().str.len().sum()) == int(t_new2.str.split().str.len().sum()),
  "A_v2 text == A_v1 text minus the full stops inside words and the commas (all rows)": all(v1_to_v2(a) == b for a, b in zip(MV.loc[has2, "transcript_normalized"], t_new2)),
  "the question mark is untouched: same number as in the raw text": allnew2.count("?") == allraw2.count("?"),
  "CHARACTER-LEVEL AUDIT: every change is allowed, nothing that should change was missed": len(bad_audit2) == 0,
  "every changed transcript is traceable to a rule": bool(chk2.loc[changed2, "normalization_rules_applied"].notna().all()),
  "every transcript with a rule recorded really changed": bool(((chk2["normalization_rules_applied"].notna()) <= changed2).all()),
  "re-running the pipeline on the raw text gives identical text": all(normalize_text2(a) == b for a, b in zip(t_raw2, t_new2)),
  "running it again on the normalized text changes nothing": all(normalize_text2(b) == b for b in t_new2),
  "the policy version A_v2 is recorded on every normalized row": bool((chk2.loc[has2, "normalization_policy_version"] == POLICY_VERSION2).all()),
  "all 10 rule-level checks pass": len(rule_ok2) == 10 and all(rule_ok2.values()),
  "the numbers of all 10 rules agree with the dry run stored in the A_v2 policy file": len(agree2) == 10 and all(agree2.values()),
}
for k, v in checks2.items(): print(("PASS  " if v else "FAIL  ") + k)
print("\nRule-level checks:", rule_ok2); print("Agreement with dry run per rule:", agree2)
print("Exception words that have punctuation glued to them (matched by A_v2, would have been missed by A_v1):", len(glued2))
if bad_audit2:
    print(f"\nCharacter-level problems in {len(bad_audit2)} transcripts; the first 5:")
    for k in list(bad_audit2)[:5]: print("  ", k, bad_audit2[k][:3])
assert all(checks2.values()), "at least one check failed - read the FAIL lines above and do not use the new column yet"
b_sym, a_sym = collections.Counter(allraw2), collections.Counter(allnew2)
print(f"\nDistinct symbols: raw {len(b_sym)} -> A_v2 text {len(a_sym)}  (A_v1 text had 98)")
print("Symbols that disappeared:", {c: n for c, n in b_sym.items() if c not in a_sym})
print("Symbols that are NEW      :", {c: n for c, n in a_sym.items() if c not in b_sym} or "(none)")
print("Full stops / commas left in the text:", a_sym["."], "/", a_sym[","], "| question marks:", a_sym["?"], "(unchanged)")


# Original zero-based notebook cell 28
print("=" * 100)
print("CELL V5 - CHANGE REPORT FOR A_v2, AND THE NEW COUNT OF TRANSCRIPTS THAT STILL NEED A HUMAN LOOK")
print("=" * 100)
print("""Part 1 - the change report (one line per rule). rows_affected = DIFFERENT transcripts changed; occurrences_changed = symbols changed or deleted;
hours_affected = audio of those transcripts, counted once. Rules that existed in A_v1 must show the same numbers as in the A_v1 report.
Part 2 - the review list: transcripts that still contain punctuation, a digit or a rare symbol (a review signal, not proof of an error).
This cell only counts and saves lists. Nothing is flagged in the master table and nothing is excluded from validation/test yet.
""")
def validation_of2(r): return "Not checked" if (r not in rule_ok2 or r not in agree2) else ("Yes" if (rule_ok2[r] and agree2[r]) else "No")
rep2 = []
for _, p in POL2[POL2["rule_id"].isin(sorted(in_pipeline2))].iterrows():
    r = p["rule_id"]; c = CH2[(CH2.rule_id == r) & (CH2.kind == "changed")]; e = CH2[(CH2.rule_id == r) & (CH2.kind == "exception")]
    rep2.append(dict(rule_id=r, issue=p["issue"], policy_version=POLICY_VERSION2, status=p["status"], rows_affected=int(c["sample_id"].nunique()), occurrences_changed=int(c["n"].sum()),
                     hours_affected=round(float(HRS[MV["sample_id"].isin(c["sample_id"]).to_numpy()].sum()), 3), exceptions_flagged=int(e["n"].sum()), exception_rows=int(e["sample_id"].nunique()),
                     examples_before=" | ".join(f"{s}: {b}" for s, b in zip(c["sample_id"].head(3), c["before"].head(3))), examples_after=" | ".join(f"{s}: {a}" for s, a in zip(c["sample_id"].head(3), c["after"].head(3))),
                     exception_examples=" | ".join(f"{s}: {b}" for s, b in zip(e["sample_id"].head(3), e["before"].head(3))), validation_passed=validation_of2(r)))
REP2 = pd.DataFrame(rep2)
anyc2 = CH2[CH2.kind == "changed"]["sample_id"].unique()
REP2 = pd.concat([REP2, pd.DataFrame([dict(rule_id="ALL RULES (distinct)", issue="transcripts changed by at least one rule", policy_version=POLICY_VERSION2, status="", rows_affected=len(anyc2),
      occurrences_changed=int(CH2[CH2.kind == "changed"]["n"].sum()), hours_affected=round(float(HRS[MV["sample_id"].isin(anyc2).to_numpy()].sum()), 3), exceptions_flagged=int(CH2[CH2.kind == "exception"]["n"].sum()),
      exception_rows=int(CH2[CH2.kind == "exception"]["sample_id"].nunique()), examples_before="", examples_after="", exception_examples="", validation_passed="Yes" if all(checks2.values()) else "No")])], ignore_index=True)
REP2.to_csv(CLEAN / "normalization_change_report_A_v2.csv", index=False, encoding="utf-8-sig")
display(REP2[["rule_id", "issue", "status", "rows_affected", "occurrences_changed", "hours_affected", "exceptions_flagged", "validation_passed"]])
print("Note: the last line counts every transcript once, so it is NOT the sum of the lines above.")
print("\nBefore -> after examples of the two NEW rules (look at them: the checks cannot judge whether a change is linguistically right):")
for _, r in REP2[REP2.rule_id.isin(["NORM_010", "NORM_019"])].iterrows(): print(f"  {r['rule_id']}\n      before: {r['examples_before'][:150]}\n      after : {r['examples_after'][:150]}")
print("\nSaved: cleaning/normalization_change_report_A_v2.csv")

# ---- part 2: the review list ----
ACCEPTED_SYMBOLS2 = set()           # e.g. {"ɜ", "ɬ"} after a reviewer has looked at the examples and accepted them
cn2 = collections.Counter(allnew2)
rare2 = {c for c, n in cn2.items() if n < 5 and c != " " and (not c.isascii() or c.islower()) and c not in ACCEPTED_SYMBOLS2}
def reasons2(s):
    r = []
    for c in sorted(set(s)):
        if c in ACCEPTED_SYMBOLS2: continue
        if c.isascii() and c.isdigit(): r.append(("digit", c))
        elif c.isascii() and not c.isalnum() and c != " ": r.append(("punctuation", c))
        elif c in rare2: r.append(("rare symbol", c))
    return r
U2 = chk2.loc[has2, ["sample_id", "original_split", "speaker_status", "duration_decoded", "transcript_normalized"]].copy()
U2["why"] = U2["transcript_normalized"].map(reasons2); U2 = U2[U2["why"].map(len) > 0].copy()
U2["categories"] = U2["why"].map(lambda r: "; ".join(sorted({k for k, _ in r})))
U2["symbols"] = U2["why"].map(lambda r: "; ".join(f"{k} {c} (U+{ord(c):04X})" for k, c in r)); U2["hours"] = U2["duration_decoded"] / 3600
u1 = pd.read_csv(CLEAN / "unresolved_symbol_rows_A_v1.csv", encoding="utf-8-sig", keep_default_na=False)
print(f"\nTranscripts that still need a look under A_v2: {len(U2)}  ({U2['hours'].sum():.2f} h = {100 * len(U2) / int(has2.sum()):.1f}% of the transcribed clips)")
print(f"Under A_v1 it was {len(u1)} transcripts (0.74 h). Resolved by A_v2: {len(set(u1.sample_id) - set(U2.sample_id))}; new: {len(set(U2.sample_id) - set(u1.sample_id))}")
print("\nBy category (a transcript can be in several):"); display(pd.Series(collections.Counter(k for r in U2["why"] for k in {x for x, _ in r})).rename("transcripts").to_frame())
print("By original split (they must stay out of validation/test until reviewed):"); display(U2["original_split"].value_counts().rename("transcripts").to_frame())
sym_rows2 = collections.defaultdict(list)
for sid, why, txt in zip(U2["sample_id"], U2["why"], U2["transcript_normalized"]):
    for k, c in why: sym_rows2[(k, c)].append((sid, txt))
S4b = pd.DataFrame([dict(category=k, symbol=c, code=f"U+{ord(c):04X}", transcripts=len(v), example=f"{v[0][0]}: {v[0][1][:60]}") for (k, c), v in sym_rows2.items()]).sort_values(["category", "transcripts"], ascending=[True, False])
print("Every symbol that causes a flag:"); display(S4b)
U2[["sample_id", "original_split", "speaker_status", "hours", "categories", "symbols", "transcript_normalized"]].to_csv(CLEAN / "unresolved_symbol_rows_A_v2.csv", index=False, encoding="utf-8-sig")
S4b.to_csv(CLEAN / "unresolved_symbol_summary_A_v2.csv", index=False, encoding="utf-8-sig")
print("Saved: cleaning/unresolved_symbol_rows_A_v2.csv and cleaning/unresolved_symbol_summary_A_v2.csv")


# Original zero-based notebook cell 29
print("=" * 100)
print("CELL V6 - RECORD THE A_v2 DECISIONS IN THE DECISION LOG (safe to run several times)")
print("=" * 100)
print("The log keeps three things apart: (1) what was APPLIED, (2) what is NOT linguistically confirmed, (3) what review is still PENDING.\n")
log_decision("4", f"Policy A_v2 adopted (cleaning/normalization_policy_A_v2.csv, 19 rules) and applied: transcript_normalized rebuilt for {int(has2.sum())} transcripts in master_A_v1_3; transcript_raw untouched; A_v1 and master_A_v1_2 kept frozen",
             "A_v2 = the A_v1 rules plus the changes listed in this log; every consistency and character-level check passed", int(has2.sum()), round(float(HRS[has2].sum()), 3))
log_decision("4", "User confirmed policy A_v2. New PROVISIONAL rules: NORM_010 deletes a full stop between two symbols of a word (IPA syllable break, used inconsistently); NORM_019 deletes commas. Not confirmed by the maintainers",
             "neither has a sound value of its own; reversible because transcript_raw is unchanged", int(stats2["NORM_010"]["rows"]) + int(stats2["NORM_019"]["rows"]), None)
log_decision("4", "The question mark is NOT deleted: it stays unchanged and flagged (NORM_011)",
             "several question marks carry a combining mark or stand next to the real glottal stop in the same transcript, so they are probably a typed ʔ; deleting them would erase sounds; the maintainers must say", int(E_q.transcripts), None)
log_decision("4", "A_v2 consistency fixes without any change of output: one shared vowel set that includes ʉ; the '[' rule skips accents like the length rules (and never creates a second dental mark); the NORM_008 exception match survives punctuation glued to the word",
             "the code and the policy text agree; checked: no ':' or ';' follows ʉ, all 28 '[' follow t/d directly, no exception word has punctuation attached", None, None)
log_decision("4", "Clarification of the earlier NORM_008 wording 'an unmapped comparison version is kept for an ablation': no such column exists; the unmapped text can be recomputed from transcript_raw by running the pipeline without NORM_008",
             "the log must not claim an artefact that is not on disk; the ablation is still planned", None, None)
log_decision("4", "Change report and per-row change list for A_v2 saved (cleaning/normalization_change_report_A_v2.csv, cleaning/normalization_row_changes_A_v2.csv); review lists recounted (cleaning/unresolved_symbol_*_A_v2.csv)",
             "every changed transcript can be traced to one or more rule IDs", len(CH2), None)
log_decision("4", f"Recount after A_v2: {len(U2)} transcripts ({U2['hours'].sum():.2f} h) still contain punctuation, a digit or a rare symbol (a review signal, not proof of error); replaces the A_v1 count of 103",
             "they are NOT yet flagged in the master table and NOT yet excluded from validation/test; that happens in the next step", len(U2), round(float(U2["hours"].sum()), 3))
log_decision("4", "QUESTIONS FOR THE MAINTAINERS: is '?' a typed glottal stop; what do full stops and commas inside words mean; is the dental mark meant to be consistent; why does ː follow consonants; what are y (19x), ʉ (103x), ᵾ and the rare symbols; is tone intentionally unmarked (no tone letters occur); is d̪0 an o",
             "these are linguistic facts that the data cannot establish", None, None)
log_decision("4", "REVIEW STILL PENDING: maintainers' answers on all provisional rules; manual listening audit; review of the unresolved transcripts; row flags in the master table; Step 5 filters; final split",
             "automated checks cannot establish linguistic correctness", None, None)
print("\nThe decision log now has", len(pd.read_csv(LOG, encoding="utf-8-sig")), "entries.")


# Original zero-based notebook cell 30
print("=" * 100)
print("CELL V7 - FINAL CHECKS ON WHAT IS SAVED ON DISK")
print("=" * 100)
v13 = pd.read_parquet(MAN / "master_A_v1_3.parquet"); v12 = pd.read_parquet(MAN / "master_A_v1_2.parquet")
pol_disk = pd.read_csv(CLEAN / "normalization_policy_A_v2.csv", encoding="utf-8-sig", keep_default_na=False)
rep_disk2 = pd.read_csv(CLEAN / "normalization_change_report_A_v2.csv", encoding="utf-8-sig", keep_default_na=False)
rows_disk2 = pd.read_csv(CLEAN / "normalization_row_changes_A_v2.csv", encoding="utf-8-sig", keep_default_na=False)
unres_disk2 = pd.read_csv(CLEAN / "unresolved_symbol_rows_A_v2.csv", encoding="utf-8-sig", keep_default_na=False)
v1_rep_disk = pd.read_csv(CLEAN / "normalization_change_report_A_v1.csv", encoding="utf-8-sig", keep_default_na=False)
log = pd.read_csv(LOG, encoding="utf-8-sig")
J = v12.merge(v13, on="sample_id", how="outer", suffixes=("_12", "_13"), indicator=True, validate="one_to_one")      # compare by sample_id, never by row order
ok2 = {
  "every sample_id is in both versions (compared by sample_id)": bool((J["_merge"] == "both").all()),
  "transcript_raw identical to v1.2, row by row": bool((J["transcript_raw_12"] == J["transcript_raw_13"]).all()),
  "v1.3 has exactly the same 57 columns as v1.2": list(v13.columns) == list(v12.columns),
  "transcript_normalized is filled for exactly the clips that have a transcript": int(v13["transcript_normalized"].notna().sum()) == int((~v13["transcript_missing"]).sum()) and bool(v13.loc[v13["transcript_missing"], "transcript_normalized"].isna().all()),
  "master_A_v1_2 still holds the A_v1 result (policy version A_v1 on every normalized row)": bool((v12.loc[~v12["transcript_missing"], "normalization_policy_version"] == "A_v1").all()),
  "the A_v2 policy file has 19 rules with unique ids and A_v2 on every row": len(pol_disk) == 19 and pol_disk["rule_id"].is_unique and set(pol_disk["policy_version"]) == {"A_v2"},
  "the A_v1 change report on disk is unchanged (9 lines, all validated)": len(v1_rep_disk) == 9 and (v1_rep_disk["validation_passed"] == "Yes").all(),
  "the A_v2 change report is complete: 10 rule lines + 1 total line, all validated": len(rep_disk2) == 11 and (rep_disk2["validation_passed"] == "Yes").all(),
  "the change list on disk is consistent with the change report": int(rows_disk2[rows_disk2["kind"] == "changed"]["sample_id"].nunique()) == int(rep_disk2.iloc[-1]["rows_affected"]),
  "the review list on disk is readable and has the expected columns": {"sample_id", "categories", "symbols", "transcript_normalized"} <= set(unres_disk2.columns) and len(unres_disk2) == len(U2),
  "decision log has no duplicate decisions": not log["decision"].duplicated().any(),
}
for k, v in ok2.items(): print(("PASS  " if v else "FAIL  ") + k)
print("\nRegression checks for THIS data version:")
reg2 = {"6718 rows": len(v13) == 6718, "5651 transcripts normalized": int((~v13["transcript_missing"]).sum()) == 5651}
for k, v in reg2.items(): print(("PASS  " if v else "FAIL  ") + k)
assert all(ok2.values()) and all(reg2.values()), "at least one check failed - read the FAIL lines above"
print("\nAll checks passed. master_A_v1_3 is now the table with the A_v2 normalized transcripts; master_A_v1_2 keeps the A_v1 result.")
print("Reminder: 'passed' means the rules were applied correctly. The linguistic review of the provisional rules is still pending.")
