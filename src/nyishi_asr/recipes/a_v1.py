"""Deterministic recovered a_v1 workflow. Execute via nyishi_asr.restore."""


# Original zero-based notebook cell 0
print("=" * 100)
print("CELL T0 - SETUP: imports, folders, and the safe decision-log helper")
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

print("Folders OK:", MAN.exists(), CLEAN.exists(), META.exists(), "| decision log exists:", LOG.exists())


# Original zero-based notebook cell 1
print("=" * 100)
print("CELL T1 - LOAD THE MASTER TABLE (version 1.1) AND SELECT THE TRANSCRIPTS TO WORK ON")
print("=" * 100)
M = pd.read_parquet(MAN / "master_A_v1_1.parquet")           # always read the PARQUET file: it keeps the text 'N/A' exactly as it is
assert len(M) == 6718 and M["sample_id"].is_unique
M["hours"] = M["duration_decoded"] / 3600                    # decoded duration in hours (the one we decided to use)

tx = M[~M["transcript_missing"]].copy()                      # only clips that have a real transcript
tx["t"] = tx["transcript_raw"]                               # short name for the raw transcript text
print(f"Master table: {M.shape[0]} rows x {M.shape[1]} columns")
print(f"Clips with a transcript: {len(tx)} ({tx['hours'].sum():.2f} h)   |   clips with N/A (left out here): {len(M) - len(tx)}")
print("\nColumns that this work will use:", ["sample_id", "transcript_raw", "transcript_missing", "transcript_normalized", "transcript_review_status", "language_status", "duration_decoded"])
print("transcript_normalized is still empty (expected 0 filled):", int(M["transcript_normalized"].notna().sum()))
print("\nA first look at 5 random transcripts:")
display(tx.sample(5, random_state=3)[["sample_id", "speaker_id", "duration_decoded", "t"]])


# Original zero-based notebook cell 2
print("=" * 100)
print("CELL T2 - THE VOWEL SET (needed by the ':' and ';' rules) - please check it")
print("=" * 100)
print("""The rules that turn ':' or ';' into the length mark 'ː' only apply when the symbol comes right after a VOWEL.
'Right after' means: the nearest preceding LETTER, skipping any accent marks (combining marks) in between. We do NOT remove accents.
Below is the vowel set the rules will use, with how often each vowel occurs. The second line lists every other letter in the
transcripts, so you can see whether something that is really a vowel was left out.
""")
from nyishi_asr.norm_a_v1 import VOWELS
cnt = collections.Counter("".join(tx["t"]))
vow_found = sorted([(c, n) for c, n in cnt.items() if c in VOWELS], key=lambda x: -x[1])
others = sorted([(c, n) for c, n in cnt.items() if c not in VOWELS and (ud.category(c).startswith("L")) ], key=lambda x: -x[1])
print("VOWELS found in the data :", "  ".join(f"{c}:{n}" for c, n in vow_found))
print("OTHER letters in the data:", "  ".join(f"{c}:{n}" for c, n in others))
print("\nVowel letters in the set that never occur:", "".join(sorted(VOWELS - set(cnt))) or "(none)")


# Original zero-based notebook cell 3
from nyishi_asr.norm_a_v1 import normalize_text
from nyishi_asr.norm_a_v1 import *
print("=" * 100)
print("CELL T3 - THE NORMALIZATION RULES AS FUNCTIONS, AND A DRY RUN (nothing is saved, no transcript is changed)")
print("=" * 100)
print("""Each rule is a small function that takes ONE transcript and returns: the new text, how many symbols it changed, and how many
symbols it deliberately left alone (exceptions). We run them in the planned order on a COPY of the transcripts to count what each rule
would do. The numbers become the evidence in the policy table. The same functions will be used later to apply the rules for real.
""")

# ^ words typed in ordinary English spelling (found by manual review): their g belongs to an English spelling, not to an IPA symbol.
#   English words written phonetically in IPA letters (gorment̪, egzam, ...) are NOT in this list: for them g -> ɡ is correct.


stats, examples, exc_detail = {}, collections.defaultdict(list), collections.defaultdict(list)
cur = tx["t"].copy()
for rid, fn in PIPELINE:
    res = [fn(t) for t in cur]
    new = pd.Series([r[0] for r in res], index=cur.index)
    ch = np.array([r[1] for r in res]); ex = np.array([r[2] for r in res])
    stats[rid] = dict(rows=int((ch > 0).sum()), occurrences=int(ch.sum()), exceptions=int(ex.sum()), exception_rows=int((ex > 0).sum()),
                      hours=round(float(tx.loc[ch > 0, "hours"].sum()), 3))
    for k in np.where(ch > 0)[0][:3]:
        before, after = cur.iloc[k], new.iloc[k]
        bt, at = before.split(" "), after.split(" ")
        diff = [(a, b) for a, b in zip(bt, at) if a != b][:1] if len(bt) == len(at) else [(before[:30], after[:30])]
        examples[rid].append(f"{cur.index[k]}: {diff[0][0]} -> {diff[0][1]}" if diff else cur.index[k])
    for k in np.where(ex > 0)[0]:
        exc_detail[rid] += [(tx.loc[cur.index[k], "sample_id"], d) for d in res[k][3]]
    cur = new

DRY = tx[["sample_id", "t"]].copy(); DRY["dry_run_text"] = cur.values
S = pd.DataFrame(stats).T.astype({"rows": int, "occurrences": int, "exceptions": int, "exception_rows": int})
S.index.name = "rule"
print("Dry-run result per rule (rows = transcripts changed; occurrences = symbols changed; hours = audio of those transcripts counted once):")
display(S)
for rid in ("NORM_005", "NORM_006", "NORM_007", "NORM_008"):
    print(f"\nExceptions that the rule {rid} deliberately left unchanged ({len(exc_detail[rid])}):")
    for sid, d in exc_detail[rid]: print("   ", sid, "|", d)
print("\nREVIEW AID - tokens made only of ordinary letters that contain a g and look like English spelling (extend G_EXCEPTION_WORDS above if you find more):")
_vocab = collections.Counter(w for t in tx["t"] for w in t.split(" ") if w)
_cand = sorted([(w, n) for w, n in _vocab.items() if "g" in w and w.isascii() and w.isalpha() and (re.search(r"[wyq]", w) or re.search(r"(ment|tion|ing$|ground|exerc|gener)", w.lower()))], key=lambda x: -x[1])
print("   ", [(w, n, "EXCEPTION" if w.lower() in G_EXCEPTION_WORDS else "") for w, n in _cand][:40])
print("\nBefore -> after examples (the changed word only):")
for rid in ("NORM_003", "NORM_005", "NORM_006", "NORM_007", "NORM_008", "NORM_009"):
    print(" ", rid, "|", " ;  ".join(examples[rid][:3]))


# Original zero-based notebook cell 4
print("=" * 100)
print("CELL T4 - VALIDATE THE DRY RUN (nothing may be lost or invented by accident)")
print("=" * 100)

before_all, after_all = "".join(DRY["t"]), "".join(DRY["dry_run_text"])
comb_before = sum(ud.category(c).startswith("M") for c in before_all); comb_after = sum(ud.category(c).startswith("M") for c in after_all)
checks = {
  "no transcript became empty": bool((DRY["dry_run_text"].str.len() > 0).all()),
  "no leading/trailing/double spaces remain": bool(not DRY["dry_run_text"].str.contains(r"^ | $|  ", regex=True).any()),
  "text is NFD everywhere": bool((DRY["dry_run_text"] == DRY["dry_run_text"].map(lambda s: ud.normalize("NFD", s))).all()),
  "no ASCII capital letter remains": bool(not DRY["dry_run_text"].str.contains(r"[A-Z]", regex=True).any()),
  "no ':' remains except the exceptions": after_all.count(":") == stats["NORM_005"]["exceptions"],
  "no ';' remains except the exceptions": after_all.count(";") == stats["NORM_006"]["exceptions"],
  "no '[' remains except the exceptions": after_all.count("[") == stats["NORM_007"]["exceptions"],
  "combining marks were NOT removed (only added by the '[' rule)": comb_after == comb_before + stats["NORM_007"]["occurrences"],
  "word count is unchanged (no word merged or split)": int(DRY["t"].str.split().str.len().sum()) == int(DRY["dry_run_text"].str.split().str.len().sum()),
  "running the whole pipeline again on its own output changes nothing (stable)": all(normalize_text(t) == t for t in DRY["dry_run_text"]),
}
for k, v in checks.items(): print(("PASS  " if v else "FAIL  ") + k)
print(f"\nCombining marks before: {comb_before:,} | after: {comb_after:,} (difference = marks created from '[' = {stats['NORM_007']['occurrences']})")


# Original zero-based notebook cell 5
print("=" * 100)
print("CELL T5 - THE POLICY TABLE: every rule, its condition, evidence and status  ->  cleaning/normalization_policy_A_v1.csv")
print("=" * 100)
print("""Statuses:  Approved = safe technical clean-up | Provisional = supported by evidence, NOT confirmed by the maintainers
           Pending review = not enough evidence to decide | Rejected = this transformation must not be applied
Nothing in the transcripts is changed here. The table records the decisions, the conditions and the evidence.
""")
REVIEWER = ""                      # <- type your name here: the person who approves these decisions (it is written into the table)
POLICY_VERSION = "A_v1"
POLICY_FILE = CLEAN / "normalization_policy_A_v1.csv"

vocab = collections.Counter(w for t in tx["t"] for w in t.split(" ") if w)
def alt_evidence(sym, alt):
    """How many distinct words contain sym, and for how many of them the same word ALSO occurs elsewhere with alt in its place."""
    ws = [w for w in vocab if sym in w]
    return len(ws), len([w for w in ws if vocab.get(w.replace(sym, alt), 0) > 0])
def ctx_stats(sym):
    """For a symbol: occurrences, % at the end of a word, % at the start of a word, % right after a vowel."""
    n = fin = ini = aft = 0
    for t in tx["t"]:
        for w in t.split(" "):
            for i, c in enumerate(w):
                if c == sym:
                    n += 1; fin += (i == len(w) - 1); ini += (i == 0); aft += (prev_base(w, i) in VOWELS)
    return n, round(100 * fin / max(n, 1)), round(100 * ini / max(n, 1)), round(100 * aft / max(n, 1))
def sym_info(chars):
    """Transcripts containing any of the symbols, total occurrences, and the audio hours of those transcripts."""
    joined = "".join(tx["t"]); mask = tx["t"].map(lambda s: any(c in s for c in chars))
    return int(mask.sum()), sum(joined.count(c) for c in chars), round(float(tx.loc[mask, "hours"].sum()), 3)

# --- extra evidence -------------------------------------------------------------------------------------------------
w_col, ok_col = alt_evidence(":", "\u02d0"); w_sem, ok_sem = alt_evidence(";", "\u02d0"); w_brk, ok_brk = alt_evidence("[", "\u032a")
tx["g_plain"] = tx["t"].str.count("g"); tx["g_ipa"] = tx["t"].str.count("\u0261")
bs = tx.groupby("speaker_group_id")[["g_plain", "g_ipa"]].sum(); bs = bs[bs.sum(axis=1) >= 30]
share = bs["g_plain"] / (bs["g_plain"] + bs["g_ipa"]); one_form = int(((share >= 0.98) | (share <= 0.02)).sum())
mixed_tx = int(((tx["g_plain"] > 0) & (tx["g_ipa"] > 0)).sum())
n_not_nfc = int((tx["t"] != tx["t"].map(lambda s: ud.normalize("NFC", s))).sum())
invis = int(tx["t"].map(lambda s: any(ud.category(c) in ("Cf", "Cc") or c == "\u00a0" for c in s)).sum())
dbl_speakers = tx.loc[tx["t"].str.contains("  ", regex=False), "speaker_group_id"].nunique()
cnt = collections.Counter("".join(tx["t"]))
months = {}
for m, g in tx.groupby("recording_month"):
    s = "".join(g["t"])
    if len(s) >= 20000: months[m] = s
def rng(sym):
    v = {m: 1000 * s.count(sym) / len(s) for m, s in months.items()}
    lo, hi = min(v, key=v.get), max(v, key=v.get)
    return f"{v[lo]:.1f} ({lo}) to {v[hi]:.1f} ({hi}) per 1000 symbols"
dot, qm = ctx_stats("."), ctx_stats("?")
rare = {c for c, n in cnt.items() if n < 5 and c != " " and (not c.isascii() or c.islower())}      # rare letters and marks (capitals/digits/punctuation have their own rules)
rare_info = sym_info(rare)
other_punct = [",", '"', "=", "'", "\\"]
OP = sym_info(other_punct)
S5, S6, S7, S8, S9 = (stats[r] for r in ("NORM_005", "NORM_006", "NORM_007", "NORM_008", "NORM_009"))   # plain dicts from the dry run: the counts stay integers

def exa(rid, k=2): return " ; ".join(examples[rid][:k])
VOWEL_TXT = "".join(sorted(VOWELS))
R = [
 # (id, issue, source, replacement, condition, examples, evidence, risk, status, action, dry-run (rows, occurrences, exceptions, hours))
 ("NORM_001", "Unicode representation", "any text", "NFD (decomposed form)", "all transcripts", "", f"text is already 100% NFD (0 transcripts change); converting to NFC would change {n_not_nfc} transcripts and create new symbols", "low", "Approved", "Apply", (0, 0, 0, 0.0)),
 ("NORM_002", "Leading/trailing whitespace", "space at either end", "(removed)", "all transcripts", "", "0 transcripts have leading or trailing whitespace", "none", "Approved", "Apply", (0, 0, 0, 0.0)),
 ("NORM_003", "Repeated spaces", "two or more spaces in a row", "one space", "ordinary spaces only; word boundaries are not otherwise touched", exa("NORM_003", 1),
  f"{int(S.loc['NORM_003','rows'])} transcripts ({int(S.loc['NORM_003','occurrences'])} extra spaces) spread over {dbl_speakers} speaker groups and all recording months: typing noise, no sign that it carries meaning", "low", "Approved", "Apply",
  (int(S.loc["NORM_003", "rows"]), int(S.loc["NORM_003", "occurrences"]), 0, float(S.loc["NORM_003", "hours"]))),
 ("NORM_004", "Invisible characters", "zero-width, control, non-breaking space", "(none)", "detect only", "", f"{invis} transcripts contain any invisible, control or non-breaking-space character", "none", "Approved", "Check only", (0, 0, 0, 0.0)),
 ("NORM_005", "Colon typed instead of length mark", ":", "\u02d0 (U+02D0)", f"only right after a vowel (nearest preceding letter, accents skipped); vowels: {VOWEL_TXT}", exa("NORM_005"),
  f"{S5['occurrences'] + S5['exceptions']} occurrences, all follow a vowel; {ok_col} of {w_col} distinct words also occur elsewhere with the real length mark", "medium (could be a deliberate mark)", "Provisional", "Apply and log",
  (int(S5["rows"]), int(S5["occurrences"]), int(S5["exceptions"]), float(S5["hours"]))),
 ("NORM_006", "Semicolon typed instead of length mark", ";", "\u02d0 (U+02D0)", f"only right after a vowel; any other semicolon is left unchanged and flagged; vowels: {VOWEL_TXT}", exa("NORM_006"),
  f"{S6['occurrences'] + S6['exceptions']} occurrences, {S6['occurrences']} follow a vowel; {ok_sem} of {w_sem} distinct words also occur with the real length mark; exceptions: " + ", ".join(sid for sid, _ in exc_detail["NORM_006"]),
  "medium", "Provisional", "Apply and log", (int(S6["rows"]), int(S6["occurrences"]), int(S6["exceptions"]), float(S6["hours"]))),
 ("NORM_007", "Square bracket typed instead of dental mark", "[", "combining dental mark U+032A", "only right after t or d (the mark is attached to that letter)", exa("NORM_007"),
  f"{S7['occurrences'] + S7['exceptions']} occurrences, all directly after t or d (the dental mark follows t/d in 12,300 of its 12,317 uses); for {ok_brk} of {w_brk} distinct words the same word occurs with the real mark", "medium (inference)", "Provisional", "Apply and log",
  (int(S7["rows"]), int(S7["occurrences"]), int(S7["exceptions"]), float(S7["hours"]))),
 ("NORM_008", "Latin g vs IPA script g", "g (U+0067)", "\u0261 (U+0261)", "every g, except in the words of the exception list (ordinary English spellings): " + ", ".join(sorted(G_EXCEPTION_WORDS)), exa("NORM_008"),
  f"{one_form} of {len(bs)} speaker groups with at least 30 g symbols use only ONE of the two forms (>=98% or <=2% plain g); only {mixed_tx} transcripts contain both: a transcriber habit, not two sounds",
  "medium (could be two real sounds; an unmapped comparison version is kept for an ablation)", "Provisional", "Apply and log", (int(S8["rows"]), int(S8["occurrences"]), int(S8["exceptions"]), float(S8["hours"]))),
 ("NORM_009", "ASCII capital letters", "A-Z", "a-z", "ASCII capitals only (no other letter is touched)", exa("NORM_009"),
  f"{int(S9['rows'])} transcripts: typing slips (hoɡE, iskuLho) and English words typed in Latin letters; capitals carry no sound meaning in IPA", "low", "Provisional", "Apply and log",
  (int(S9["rows"]), int(S9["occurrences"]), 0, float(S9["hours"]))),
 ("NORM_010", "Full stop inside words", ".", "(unchanged)", "all occurrences", "hok.kə ; cʰot̪.t̪aːme",
  f"{dot[0]} occurrences, {dot[1]}% word-final, {dot[2]}% word-initial, {dot[3]}% after a vowel: it sits between consonants, which fits an IPA syllable boundary", "medium", "Pending review", "Preserve", (*sym_info(["."])[:2], 0, sym_info(["."])[2])),
 ("NORM_011", "Question mark inside words", "?", "(unchanged)", "all occurrences", "həb? ; ak?",
  f"{qm[0]} occurrences, {qm[1]}% word-final, {qm[3]}% after a vowel: could be punctuation or a typed glottal stop (ʔ)", "medium", "Pending review", "Preserve", (*sym_info(["?"])[:2], 0, sym_info(["?"])[2])),
 ("NORM_012", "Other ASCII punctuation", ", \" = ' \\", "(unchanged)", "all occurrences", "bag,a ; bə=paŋe", "meaning not established", "medium", "Pending review", "Preserve and flag", (*OP[:2], 0, OP[2])),
 ("NORM_013", "Digit inside a transcription", "0", "(unchanged)", "occurrence in A_003884 (d̪0)", "d̪0 mad̪ən", "one transcript; probably a typo for o, but not corrected automatically", "low", "Pending review", "Flag for manual review", (*sym_info(["0"])[:2], 0, sym_info(["0"])[2])),
 ("NORM_014", "Rare letters and marks", f"{len(rare)} symbols used fewer than 5 times", "(unchanged)", "all occurrences", "", "rarity alone does not show an error", "medium", "Pending review", "Preserve and flag", (*rare_info[:2], 0, rare_info[2])),
 ("NORM_015", "Dental mark / affricate / fricative conventions", "U+032A, dʒ ʒ ʃ ɟ c ...", "(unchanged)", "all transcripts", "",
  f"usage varies by recording month: dental mark {rng(chr(0x032a))}; ɟ {rng(chr(0x025f))}; ʃ {rng(chr(0x0283))}: transcriber habit or loanwords, only an expert can tell", "high", "Pending review", "Preserve", (*sym_info([chr(0x032a)])[:2], 0, sym_info([chr(0x032a)])[2])),
 ("NORM_016", "Vowel conventions", "ɨ ɛ ɔ ʈ ...", "(unchanged)", "all transcripts", "", f"usage varies by recording month: ɨ {rng(chr(0x0268))}; ɛ {rng(chr(0x025b))}; ʈ {rng(chr(0x0288))}", "high", "Pending review", "Preserve", (*sym_info([chr(0x0268)])[:2], 0, sym_info([chr(0x0268)])[2])),
 ("NORM_017", "Removing combining diacritics", "combining marks", "(removed)", "never", "", "would destroy phonetic information", "high", "Rejected", "Do not apply", (0, 0, 0, 0.0)),
 ("NORM_018", "Automatic correction of apparent transcription errors", "any", "(corrected)", "never", "", "a correction needs a reviewer and the audio; it is not a normalization", "high", "Rejected", "Do not apply", (0, 0, 0, 0.0)),
]
POL = pd.DataFrame([dict(rule_id=r[0], issue=r[1], source_pattern=r[2], replacement=r[3], condition=r[4], examples=r[5], evidence=r[6], linguistic_risk=r[7],
                         status=r[8], action=r[9], reviewer=REVIEWER, review_date=TODAY, policy_version=POLICY_VERSION,
                         dry_run_rows=r[10][0], dry_run_occurrences=r[10][1], dry_run_exceptions=r[10][2], dry_run_hours=r[10][3]) for r in R])

KEY = ["rule_id", "source_pattern", "replacement", "condition", "status", "action"]
if POLICY_FILE.exists():
    old = pd.read_csv(POLICY_FILE, encoding="utf-8-sig", keep_default_na=False)
    if old[KEY].astype(str).reset_index(drop=True).equals(POL[KEY].astype(str).reset_index(drop=True)):
        print(f"{POLICY_FILE.name} already exists and its rules are identical: NOT rewritten (a frozen version is never edited).")
    else:
        alt = CLEAN / "normalization_policy_A_v1_PROPOSED_CHANGES.csv"; POL.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"{POLICY_FILE.name} exists and the rules DIFFER. It was not overwritten. The new version was saved as {alt.name}; if the policy really changed, name it A_v2.")
else:
    POL.to_csv(POLICY_FILE, index=False, encoding="utf-8-sig"); print("Saved:", POLICY_FILE)
print("\nThe policy table (key columns):")
display(POL[["rule_id", "issue", "source_pattern", "replacement", "status", "action", "dry_run_rows", "dry_run_occurrences", "dry_run_exceptions", "dry_run_hours"]])
print("\nThe evidence behind each rule:")
for r in R: print(f"  {r[0]} [{r[8]}] {r[6]}")
print("\nStatus counts:", POL["status"].value_counts().to_dict())
print("The reviewer column is empty. To record your name, type it in REVIEWER at the top of this cell, delete the file cleaning/normalization_policy_A_v1.csv and re-run (this version is not frozen until you start applying the rules).")


# Original zero-based notebook cell 6
print("=" * 100)
print("CELL T6 - RECORD THE TRANSCRIPTION DECISIONS IN THE DECISION LOG (safe to run several times)")
print("=" * 100)
log_decision("4", "Normalization policy A_v1 adopted (cleaning/normalization_policy_A_v1.csv, 18 rules): transcript_raw is never modified; the normalized text goes into transcript_normalized",
             "versioned, reversible rules; every rule has a status, condition and evidence", len(tx), round(float(tx["hours"].sum()), 3))
log_decision("4", "Approved technical rules: NFD (text already NFD, 0 changes), trim (0), collapse repeated spaces (640 transcripts, 847 spaces), invisible-character check (none found)",
             "formatting only; double spaces are spread over many speakers and months", int(S.loc["NORM_003", "rows"]), float(S.loc["NORM_003", "hours"]))
log_decision("4", "Provisional rule NORM_005: ':' -> 'ː' only right after a vowel (all 20 occurrences, 7 transcripts)",
             "same position pattern as the length mark; 7 of 14 words also occur with the real mark; to be confirmed by the maintainers", int(S5["rows"]), float(S5["hours"]))
log_decision("4", "Provisional rule NORM_006: ';' -> 'ː' only right after a vowel (26 of 28); the 2 others (A_001165, A_005631) stay unchanged and are flagged",
             "same pattern as ':'; exceptions are not after a vowel", int(S6["rows"]), float(S6["hours"]))
log_decision("4", "Provisional rule NORM_007: '[' -> combining dental mark U+032A only right after t or d (all 28 occurrences, 26 transcripts)",
             "every '[' sits exactly where the dental mark goes; the same words occur with the real mark; an inference, to be confirmed", int(S7["rows"]), float(S7["hours"]))
log_decision("4", "Provisional rule NORM_008: Latin g -> IPA ɡ (U+0261) everywhere except a short explicit list of words in ordinary English spelling; an unmapped comparison version is kept for an ablation",
             "most speaker groups use only one of the two forms, so it is a transcriber habit; affects the most data of all rules", int(S8["rows"]), float(S8["hours"]))
log_decision("4", "The NORM_008 exception is an explicit word list, not a capital-letter test",
             "a capital-letter test is erased by the lowercase rule, so a second pass (or a model's lowercase output) would be treated differently; the list keeps the pipeline stable", int(S8["exception_rows"]), None)
log_decision("4", "Provisional rule NORM_009: ASCII capitals A-Z -> lowercase (13 transcripts); rows with mixed-language words get language_status='mixed' only where the content justifies it",
             "capitals carry no sound meaning in IPA; nothing but A-Z is changed", int(S9["rows"]), float(S9["hours"]))
log_decision("4", f"Preserved and flagged, not changed: '.', '?', ',', '\"', '=', \"'\", '\\', the digit in A_003884, {len(rare)} rare letters and marks; dental, affricate and vowel conventions pending expert review",
             "meaning not established; rarity or inconsistency alone does not justify a change", rare_info[0] + OP[0], None)
log_decision("4", "Rejected: removing combining diacritics; automatic correction of apparent transcription errors",
             "they would destroy phonetic information or mix correction with normalization", None, None)
log_decision("4", "Three transcript layers: raw (never changed), normalized (rules above), reviewed reference (checked against the audio by a competent reviewer); normalization and correction are separate operations",
             "keeps every change traceable", None, None)
log_decision("4", "Transcripts with unresolved symbols get transcript_review_status='needs_review' and stay out of validation and test; the preliminary count of 168 is recalculated after the rules are applied",
             "uncertain references would make evaluation scores hard to interpret", None, None)
log_decision("4", "Evaluation convention: references and model outputs both NFD; primary CER on the normalized convention; secondary convention-tolerant CER with a documented merge list (each merge justified); WER only once word boundaries are reliable",
             "the tolerant CER estimates the effect of annotation inconsistency and must never replace the primary CER", None, None)
print("\nThe decision log now has", len(pd.read_csv(LOG, encoding="utf-8-sig")), "entries. The newest ones:")
display(pd.read_csv(LOG, encoding="utf-8-sig")[["step", "decision", "rows_affected", "hours_affected"]].tail(13))


# Original zero-based notebook cell 7
print("=" * 100)
print("CELL T7 - FINAL CHECKS ON TASK 1 (policy table and decision log)")
print("=" * 100)
pol = pd.read_csv(POLICY_FILE, encoding="utf-8-sig", keep_default_na=False)
log = pd.read_csv(LOG, encoding="utf-8-sig")
tests = {
  "policy file has 18 rules with unique rule_ids": len(pol) == 18 and pol["rule_id"].is_unique,
  "every status is Approved / Provisional / Pending review / Rejected": set(pol["status"]) <= {"Approved", "Provisional", "Pending review", "Rejected"},
  "every Provisional rule has a condition and evidence": bool(pol.loc[pol["status"] == "Provisional", ["condition", "evidence"]].apply(lambda c: c.str.len() > 0).all().all()),
  "dry run: ':' changes 20 symbols, no exception": int(S5["occurrences"]) == 20 and int(S5["exceptions"]) == 0,
  "dry run: ';' changes 26 symbols, 2 exceptions": int(S6["occurrences"]) == 26 and int(S6["exceptions"]) == 2,
  "dry run: '[' changes 28 symbols, no exception": int(S7["occurrences"]) == 28 and int(S7["exceptions"]) == 0,
  "dry run: 13 transcripts have capitals": int(S9["rows"]) == 13,
  "the pipeline is stable (running it twice gives the same text)": all(normalize_text(t) == t for t in DRY["dry_run_text"].head(3000)),
  "NOTHING was applied: transcript_normalized is still empty in the master table": int(pd.read_parquet(MAN / "master_A_v1_1.parquet")["transcript_normalized"].notna().sum()) == 0,
  "transcript_raw is unchanged": pd.read_parquet(MAN / "master_A_v1_1.parquet")["transcript_raw"].equals(M["transcript_raw"]),
  "decision log has no duplicate decisions": not log["decision"].duplicated().any(),
}
for k, v in tests.items(): print(("PASS  " if v else "FAIL  ") + k)
assert all(tests.values()), "at least one check failed - read the FAIL lines above"
print("\nAll checks passed. Task 1 (policy table + decision log) is complete. The rules are defined and tested but NOT yet applied.")


# Original zero-based notebook cell 10
print("=" * 100)
print("CELL A0 - BEFORE WE APPLY ANYTHING: is the policy file sound, do the rule functions match it, and do YOU confirm the provisional rules?")
print("=" * 100)
needed = ["PIPELINE", "normalize_text", "VOWELS", "G_EXCEPTION_WORDS", "log_decision", "prev_base", "TODAY", "MAN", "CLEAN", "LOG"]
missing = [n for n in needed if n not in globals()]
assert not missing, f"Run cells T0 to T4 first in this session (missing: {missing})"

POLICY_VERSION = "A_v1"
POLICY_FILE = CLEAN / "normalization_policy_A_v1.csv"
assert POLICY_FILE.exists(), "the policy file is missing: run cell T5 first"
pol = pd.read_csv(POLICY_FILE, encoding="utf-8-sig", keep_default_na=False)

# --- the policy file itself must be consistent ---
assert pol["rule_id"].is_unique, "duplicate rule_id in the policy file"
assert set(pol["policy_version"]) == {POLICY_VERSION}, "the policy file mixes versions"
assert set(pol["status"]) <= {"Approved", "Provisional", "Pending review", "Rejected"}, "unknown status in the policy file"
assert (pol["review_date"].astype(str).str.len() > 0).all(), "a rule has no review_date"

# --- the rules the policy says to APPLY must be exactly the rules in the pipeline (NORM_001b is just the final NFD pass) ---
to_apply = set(pol.loc[pol["action"].isin(["Apply", "Apply and log"]), "rule_id"])
in_pipeline = {rid for rid, _ in PIPELINE if rid != "NORM_001b"}
print("Rules the policy says to apply :", sorted(to_apply))
print("Rules in the pipeline functions:", sorted(in_pipeline))
assert to_apply == in_pipeline, "the pipeline and the policy file disagree: stop and check"
assert not (set(pol.loc[pol["status"].isin(["Pending review", "Rejected"]), "rule_id"]) & in_pipeline), "a Pending or Rejected rule is in the pipeline"
print("\nOK: they are identical, and no Pending or Rejected rule is in the pipeline.")
print("Reviewer recorded in the policy:", repr(pol["reviewer"].iloc[0]), "(empty only means no name was typed in T5)")

# --- your explicit decision about the PROVISIONAL rules ---
prov = pol[pol["rule_id"].isin(to_apply) & (pol["status"] == "Provisional")]
print("\nThese PROVISIONAL rules will change the transcripts. They are supported by evidence but NOT confirmed by the dataset maintainers:")
for _, r in prov.iterrows():
    print(f"  {r['rule_id']}  {r['issue']}:  {r['source_pattern']}  ->  {r['replacement']}")
    print(f"        condition: {r['condition'][:110]}")
    print(f"        evidence : {r['evidence'][:150]}")
CONFIRM_PROVISIONAL_RULES = True       # <- change to True ONLY after reading the list above, then run this cell again
assert CONFIRM_PROVISIONAL_RULES, "Read the provisional rules above. If you accept applying them (the original text stays safe in transcript_raw), set CONFIRM_PROVISIONAL_RULES = True and re-run this cell."
print("\nConfirmed: the provisional rules will be applied. They stay marked 'Provisional' in the policy and in the log.")


# Original zero-based notebook cell 11
print("=" * 100)
print("CELL A1 - APPLY THE RULES: build transcript_normalized (transcript_raw is NOT touched)  ->  master_A_v1_2")
print("=" * 100)
print("""For every clip that has a transcript we run the pipeline step by step on a working copy and remember which rule changed what.
Clips with the transcript N/A stay empty. The result goes into a NEW file (master_A_v1_2); master_A_v1_1 is not modified.

What the count 'n' means in the change list (it is the same meaning in every report):
  NORM_001 / NORM_002 : 1 if the transcript changed, otherwise 0       NORM_003 : number of spaces removed
  NORM_005 ... NORM_009: number of SYMBOLS changed                       (an exception line counts symbols deliberately left alone)
""")
assert CONFIRM_PROVISIONAL_RULES, "run cell A0 and confirm the provisional rules first"
M0 = pd.read_parquet(MAN / "master_A_v1_1.parquet")
M0["hours"] = M0["duration_decoded"] / 3600

# --- input integrity: what A1 trusts must really be true ---
assert M0["sample_id"].is_unique, "duplicate sample_id"
assert int(M0["transcript_normalized"].notna().sum()) == 0, "transcript_normalized is already filled in v1.1: stop"
raw_is_na = M0["transcript_raw"].isna() | M0["transcript_raw"].astype(str).str.strip().isin(["N/A", ""])      # in this dataset a missing transcript is the TEXT 'N/A'
assert (M0["transcript_missing"].astype(bool) == raw_is_na).all(), "the flag transcript_missing and the text in transcript_raw disagree"
assert M0.loc[~M0["transcript_missing"], "transcript_raw"].map(lambda x: isinstance(x, str)).all(), "a transcript is not a text"
print(f"Input checked: {len(M0)} unique rows; transcript_missing agrees with the text of transcript_raw for every row.")

def example_pair(rid, before, after):
    """A short before/after example: the first word that changed (or, for repeated spaces, the text around the first double space)."""
    if rid == "NORM_003":
        i = before.find("  "); b = before[max(0, i - 12): i + 14]; return b.replace("  ", "\u2423\u2423"), re.sub(" {2,}", " ", b)
    bt, at = before.split(" "), after.split(" ")
    if len(bt) == len(at):
        d = [(x, y) for x, y in zip(bt, at) if x != y]
        if d: return d[0]
    return before[:30], after[:30]

norm, applied, excs, changes = [], [], [], []
for sid, raw, miss in zip(M0["sample_id"], M0["transcript_raw"], M0["transcript_missing"]):
    if miss:
        norm.append(None); applied.append(None); excs.append(None); continue
    cur, a, e = raw, [], []
    for rid, fn in PIPELINE:
        new, ch, ex, det = fn(cur)
        if ch > 0:
            a.append(rid); b_, a_ = example_pair(rid, cur, new)
            changes.append(dict(sample_id=sid, rule_id=rid, kind="changed", n=ch, before=b_, after=a_))
        if ex > 0:
            e.append(f"{rid}:{ex}"); changes.append(dict(sample_id=sid, rule_id=rid, kind="exception", n=ex, before=" | ".join(det), after=""))
        cur = new
    norm.append(cur); applied.append(";".join(a) or None); excs.append(";".join(e) or None)

CH = pd.DataFrame(changes)
M2 = M0.drop(columns=["hours"]).copy()
M2["transcript_normalized"] = norm
M2["normalization_policy_version"] = np.where(M2["transcript_missing"], None, POLICY_VERSION)
M2["normalization_rules_applied"] = applied          # which rules changed this transcript (empty = unchanged)
M2["normalization_exceptions"] = excs                # rules that deliberately left a symbol alone in this transcript

# --- a frozen version is never overwritten with different content ---
OUT12 = MAN / "master_A_v1_2.parquet"; NEWCOLS = ["sample_id", "transcript_normalized", "normalization_policy_version", "normalization_rules_applied", "normalization_exceptions"]
if OUT12.exists():
    prev = pd.read_parquet(OUT12)
    same = prev[NEWCOLS].fillna("<NA>").astype(str).equals(M2[NEWCOLS].fillna("<NA>").astype(str))
    assert same, "master_A_v1_2 already exists with DIFFERENT content. Policy A_v1 is frozen: do not overwrite it. If the rules changed, create policy A_v2."
    print("master_A_v1_2 already exists and is identical to the result of this run: nothing was rewritten.")
else:
    M2.to_parquet(OUT12, index=False)
    M2.to_csv(MAN / "master_A_v1_2.csv", index=False, encoding="utf-8-sig")
    CH.to_csv(CLEAN / "normalization_row_changes_A_v1.csv", index=False, encoding="utf-8-sig")
    print("Saved: master_A_v1_2.parquet / .csv and cleaning/normalization_row_changes_A_v1.csv")

has_text = ~M2["transcript_missing"]
changed_rows = has_text & (M2["transcript_normalized"] != M2["transcript_raw"])
print(f"\nTranscripts normalized           : {int(has_text.sum())}")
print(f"  of which changed by >= 1 rule  : {int(changed_rows.sum())}  ({M0.loc[changed_rows, 'hours'].sum():.3f} h)")
print(f"  unchanged                      : {int((has_text & ~changed_rows).sum())}")
print(f"Clips without transcript (empty) : {int((~has_text).sum())}")
print(f"Change list: {len(CH)} lines (one for every row x rule that changed something or left an exception)")
print("\nFive random changed transcripts, before and after:")
for _, r in M2[changed_rows].sample(5, random_state=2).iterrows():
    print(f"  {r['sample_id']} | rules: {r['normalization_rules_applied']}")
    print("     raw :", r["transcript_raw"][:110]); print("     norm:", r["transcript_normalized"][:110])


# Original zero-based notebook cell 12
print("=" * 100)
print("CELL A2 - VALIDATE THE RESULT (every check must pass before we trust the new column)")
print("=" * 100)
print("""Two kinds of checks. (1) Consistency checks on the whole table. (2) A CHARACTER-LEVEL AUDIT: every single changed character must be one that
a rule is allowed to change, in a place where that rule applies - and every place where a rule applies must really have been changed.
Passing these checks shows that the rules were applied correctly. It does NOT show that the rules are linguistically right: that needs the
maintainers or a Nyishi speaker.
""")
chk = pd.read_parquet(MAN / "master_A_v1_2.parquet")
old = pd.read_parquet(MAN / "master_A_v1_1.parquet")
t_raw, t_new = chk.loc[has_text, "transcript_raw"], chk.loc[has_text, "transcript_normalized"]
allnew, allraw = "".join(t_new), "".join(t_raw)
comb = lambda s: sum(ud.category(c).startswith("M") for c in s)
exc_counts = {r: int(CH[(CH.rule_id == r) & (CH.kind == "exception")]["n"].sum()) for r in ("NORM_005", "NORM_006", "NORM_007", "NORM_008")}

# --- the character-level audit -------------------------------------------------------------------------------
ALLOWED = {(":", "\u02d0"), (";", "\u02d0"), ("[", "\u032a"), ("g", "\u0261")}          # the only substitutions the policy allows (besides capitals -> lowercase)
def char_audit(raw, new):
    """Audit ONE transcript. Returns a list of problems (empty list = every change is allowed and nothing that should change was missed)."""
    base = re.sub(" {2,}", " ", ud.normalize("NFD", raw).strip())       # the only mechanical step that changes the length is removing spaces
    if len(base) != len(new): return ["length differs after removing repeated spaces"]
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^ ]+", base)]
    def in_exception(i): return any(s <= i < e and w.lower() in G_EXCEPTION_WORDS for s, e, w in spans)
    problems = []
    for i, (a, b) in enumerate(zip(base, new)):
        if a == b:                                                       # unchanged: make sure no rule SHOULD have changed it
            if a in ":;" and prev_base(base, i) in VOWELS: problems.append(f"{a!r} after a vowel was not converted (position {i})")
            elif a == "[" and i > 0 and base[i - 1] in "td": problems.append(f"'[' after t/d was not converted (position {i})")
            elif a == "g" and not in_exception(i): problems.append(f"g outside the exception list was not converted (position {i})")
            elif "A" <= a <= "Z": problems.append(f"a capital letter was not lowercased (position {i})")
        elif (a, b) in ALLOWED:                                          # changed: make sure the rule's condition really held
            if a in ":;" and prev_base(base, i) not in VOWELS: problems.append(f"{a!r} converted although no vowel precedes it (position {i})")
            elif a == "[" and not (i > 0 and base[i - 1] in "td"): problems.append(f"'[' converted although t/d does not precede it (position {i})")
            elif a == "g" and in_exception(i): problems.append(f"g converted inside an exception word (position {i})")
        elif "A" <= a <= "Z" and b == a.lower(): pass                    # capital -> lowercase is allowed
        else: problems.append(f"UNEXPECTED change {a!r} -> {b!r} (position {i})")
    return problems
audit = {sid: char_audit(r, n) for sid, r, n in zip(chk.loc[has_text, "sample_id"], t_raw, t_new)}
bad_audit = {k: v for k, v in audit.items() if v}

# --- exception words that have punctuation stuck to them would NOT be exempted by the rule: look for them ---
strip_p = lambda w: w.strip(".,;:?!\"'()[]")
glued = [(sid, w) for sid, t in zip(chk.loc[has_text, "sample_id"], t_raw) for w in t.split(" ") if "g" in w and strip_p(w).lower() in G_EXCEPTION_WORDS and w.lower() not in G_EXCEPTION_WORDS]

rule_ok = {
  "NORM_001": bool((t_new == t_new.map(lambda s: ud.normalize("NFD", s))).all()),
  "NORM_002": bool(not t_new.str.contains(r"^ | $", regex=True).any()),
  "NORM_003": bool(not t_new.str.contains("  ", regex=False).any()),
  "NORM_005": allnew.count(":") == exc_counts["NORM_005"],
  "NORM_006": allnew.count(";") == exc_counts["NORM_006"],
  "NORM_007": allnew.count("[") == exc_counts["NORM_007"] and comb(allnew) == comb(allraw) + int(CH[(CH.rule_id == "NORM_007") & (CH.kind == "changed")]["n"].sum()),
  "NORM_008": all(w.lower() in G_EXCEPTION_WORDS for t in t_new for w in t.split(" ") if "g" in w) and not glued,
  "NORM_009": bool(not t_new.str.contains(r"[A-Z]", regex=True).any()),
}
# --- do the numbers agree with the dry run that is written in the policy file? (all 8 rules) ---
agree = {}
for r in sorted(in_pipeline):
    c = CH[(CH.rule_id == r) & (CH.kind == "changed")]; p = pol[pol.rule_id == r].iloc[0]
    hrs = float(M0.loc[M0["sample_id"].isin(c["sample_id"]), "hours"].sum())
    agree[r] = bool(c["sample_id"].nunique() == p["dry_run_rows"] and int(c["n"].sum()) == p["dry_run_occurrences"]
                    and int(CH[(CH.rule_id == r) & (CH.kind == "exception")]["n"].sum()) == p["dry_run_exceptions"] and abs(hrs - float(p["dry_run_hours"])) < 0.0015)
checks = {
  "transcript_raw is identical to version 1.1": chk["transcript_raw"].equals(old["transcript_raw"]),
  "every transcript has a normalized version": bool(t_new.notna().all() and (t_new.str.len() > 0).all()),
  "clips without a transcript stay empty": bool(chk.loc[~has_text, "transcript_normalized"].isna().all()),
  "all other columns of version 1.1 are unchanged": all(chk[c].fillna("<NA>").astype(str).equals(old[c].fillna("<NA>").astype(str)) for c in old.columns if c != "transcript_normalized"),
  "the text is valid Unicode (UTF-8 round trip)": all(s.encode("utf-8").decode("utf-8") == s for s in t_new),
  "combining marks: none removed, exactly the ones from '[' added": comb(allnew) == comb(allraw) + int(CH[(CH.rule_id == "NORM_007") & (CH.kind == "changed")]["n"].sum()),
  "word count unchanged: no word was merged or split": int(t_raw.str.split().str.len().sum()) == int(t_new.str.split().str.len().sum()),
  "CHARACTER-LEVEL AUDIT: every changed character is allowed, and nothing that should change was missed": len(bad_audit) == 0,
  "no exception word has punctuation glued to it (the rule would not exempt it)": len(glued) == 0,
  "every changed transcript is traceable to a rule": bool(chk.loc[changed_rows, "normalization_rules_applied"].notna().all()),
  "every transcript with a rule recorded really changed": bool(((chk["normalization_rules_applied"].notna()) <= changed_rows).all()),
  "re-running the pipeline on the raw text gives identical text": all(normalize_text(a) == b for a, b in zip(t_raw, t_new)),
  "running it again on the normalized text changes nothing": all(normalize_text(b) == b for b in t_new),
  "the policy version is recorded on every normalized row": bool((chk.loc[has_text, "normalization_policy_version"] == POLICY_VERSION).all()),
  "all 8 rule-level checks pass": len(rule_ok) == 8 and all(rule_ok.values()),
  "the numbers of all 8 rules agree with the dry run stored in the policy file": len(agree) == 8 and all(agree.values()),
}
for k, v in checks.items(): print(("PASS  " if v else "FAIL  ") + k)
print("\nRule-level checks:", rule_ok)
print("Agreement with dry run per rule:", agree)
if bad_audit:
    print(f"\nCharacter-level problems found in {len(bad_audit)} transcripts; the first 5:")
    for k in list(bad_audit)[:5]: print("  ", k, bad_audit[k][:3])
if glued: print("\nException words with glued punctuation:", glued[:10])
assert all(checks.values()), "at least one check failed - read the FAIL lines above and do not use the new column yet"

before_sym, after_sym = collections.Counter(allraw), collections.Counter(allnew)
print(f"\nDistinct symbols: before {len(before_sym)} -> after {len(after_sym)}")
print("Symbols that disappeared:", {c: n for c, n in before_sym.items() if c not in after_sym})
print("Symbols that are NEW      :", {c: n for c, n in after_sym.items() if c not in before_sym} or "(none)")
print("Plain g left (should be only the English words):", sum(1 for c in allnew if c == "g"), "| IPA ɡ now:", after_sym["\u0261"], "(before:", before_sym["\u0261"], ")")


# Original zero-based notebook cell 13
print("=" * 100)
print("CELL A3 - THE CHANGE REPORT  ->  cleaning/normalization_change_report_A_v1.csv")
print("=" * 100)
print("""Three different numbers are reported for every rule:
  rows_affected        = how many DIFFERENT transcripts the rule changed
  occurrences_changed  = how many symbols the rule changed (one transcript can contain many; for NORM_003 it is the number of spaces removed)
  hours_affected       = the audio of those transcripts, each transcript counted ONCE
validation_passed is 'Yes' only if the rule has BOTH a rule-level check and a dry-run agreement check and both passed;
'No' if one failed; 'Not checked' if a check is missing.
""")
def validation_of(r):
    if r not in rule_ok or r not in agree: return "Not checked"
    return "Yes" if (rule_ok[r] and agree[r]) else "No"
rep = []
for _, p in pol[pol["rule_id"].isin(sorted(in_pipeline))].iterrows():
    r = p["rule_id"]
    c = CH[(CH.rule_id == r) & (CH.kind == "changed")]; e = CH[(CH.rule_id == r) & (CH.kind == "exception")]
    rep.append(dict(rule_id=r, issue=p["issue"], policy_version=POLICY_VERSION, status=p["status"], rows_affected=int(c["sample_id"].nunique()),
                    occurrences_changed=int(c["n"].sum()), hours_affected=round(float(M0.loc[M0["sample_id"].isin(c["sample_id"]), "hours"].sum()), 3),
                    exceptions_flagged=int(e["n"].sum()), exception_rows=int(e["sample_id"].nunique()),
                    examples_before=" | ".join(f"{s}: {b}" for s, b in zip(c["sample_id"].head(3), c["before"].head(3))),
                    examples_after=" | ".join(f"{s}: {a}" for s, a in zip(c["sample_id"].head(3), c["after"].head(3))),
                    exception_examples=" | ".join(f"{s}: {b}" for s, b in zip(e["sample_id"].head(3), e["before"].head(3))),
                    validation_passed=validation_of(r)))
REP = pd.DataFrame(rep)
anyc = CH[CH.kind == "changed"]["sample_id"].unique()
total = dict(rule_id="ALL RULES (distinct)", issue="transcripts changed by at least one rule", policy_version=POLICY_VERSION, status="", rows_affected=len(anyc),
             occurrences_changed=int(CH[CH.kind == "changed"]["n"].sum()), hours_affected=round(float(M0.loc[M0["sample_id"].isin(anyc), "hours"].sum()), 3),
             exceptions_flagged=int(CH[CH.kind == "exception"]["n"].sum()), exception_rows=int(CH[CH.kind == "exception"]["sample_id"].nunique()),
             examples_before="", examples_after="", exception_examples="", validation_passed="Yes" if all(checks.values()) else "No")
REP = pd.concat([REP, pd.DataFrame([total])], ignore_index=True)
REP.to_csv(CLEAN / "normalization_change_report_A_v1.csv", index=False, encoding="utf-8-sig")
display(REP[["rule_id", "issue", "status", "rows_affected", "occurrences_changed", "hours_affected", "exceptions_flagged", "validation_passed"]])
print("\nNote: the last line counts every transcript once, so it is NOT the sum of the lines above (many transcripts are changed by several rules).")
print("\nBefore -> after examples (look at them: the checks cannot judge whether a change is linguistically right):")
for _, r in REP.iloc[:-1].iterrows():
    if r["examples_before"]: print(f"  {r['rule_id']}\n      before: {r['examples_before'][:150]}\n      after : {r['examples_after'][:150]}")
print("\nExceptions that were deliberately left unchanged:")
for _, r in REP.iloc[:-1].iterrows():
    if r["exceptions_flagged"]: print(f"  {r['rule_id']} ({r['exceptions_flagged']} symbols in {r['exception_rows']} transcripts): {r['exception_examples'][:150]}")
print("\nSaved: cleaning/normalization_change_report_A_v1.csv")


# Original zero-based notebook cell 14
print("=" * 100)
print("CELL A4 - RECOUNT THE TRANSCRIPTS THAT STILL NEED A HUMAN LOOK (the earlier number, 168, was only a first estimate)")
print("=" * 100)
print("""Three separate categories are counted:
  punctuation    : ASCII punctuation that is still in the text and whose meaning is not established (needs review)
  digit          : a digit inside a transcription (needs review)
  rare symbol    : a letter or mark that occurs fewer than 5 times. This is only a REVIEW SIGNAL, not proof of an error: a rare symbol can be correct IPA.
When a reviewer decides that a rare symbol is fine, add it to ACCEPTED_SYMBOLS below and re-run: it is then no longer flagged.
IMPORTANT: this cell only counts and saves a list. Nothing is flagged in the master table yet and nothing is excluded from validation/test yet.
""")
ACCEPTED_SYMBOLS = set()           # e.g. {"ɜ", "ɬ"} after a reviewer has looked at the examples and accepted them
cn = collections.Counter(allnew)
rare_n = {c for c, n in cn.items() if n < 5 and c != " " and (not c.isascii() or c.islower()) and c not in ACCEPTED_SYMBOLS}
def reasons(s):
    r = []
    for c in sorted(set(s)):
        if c in ACCEPTED_SYMBOLS: continue
        if c.isascii() and c.isdigit(): r.append(("digit", c))
        elif c.isascii() and not c.isalnum() and c != " ": r.append(("punctuation", c))
        elif c in rare_n: r.append(("rare symbol", c))
    return r
U = chk.loc[has_text, ["sample_id", "original_split", "speaker_status", "duration_decoded", "transcript_normalized"]].copy()
U["why"] = U["transcript_normalized"].map(reasons)
U = U[U["why"].map(len) > 0].copy()
U["categories"] = U["why"].map(lambda r: "; ".join(sorted({k for k, _ in r})))
U["symbols"] = U["why"].map(lambda r: "; ".join(f"{k} {c} (U+{ord(c):04X})" for k, c in r)); U["hours"] = U["duration_decoded"] / 3600
print(f"Transcripts that still need a look: {len(U)}  ({U['hours'].sum():.2f} h = {100 * len(U) / int(has_text.sum()):.1f}% of the transcribed clips)")
print("Earlier first estimate (before the rules): 168 transcripts, 1.24 h, 3.0%")
print("\nBy category (a transcript can be in several):")
display(pd.Series(collections.Counter(k for r in U["why"] for k in {x for x, _ in r})).rename("transcripts").to_frame())
print("By original split (they must stay out of validation/test):"); display(U["original_split"].value_counts().rename("transcripts").to_frame())
print("By speaker status:"); display(U["speaker_status"].value_counts().rename("transcripts").to_frame())
sym_rows = collections.defaultdict(list)
for sid, why, txt in zip(U["sample_id"], U["why"], U["transcript_normalized"]):
    for k, c in why: sym_rows[(k, c)].append((sid, txt))
S4 = pd.DataFrame([dict(category=k, symbol=c, code=f"U+{ord(c):04X}", transcripts=len(v), example=f"{v[0][0]}: {v[0][1][:60]}") for (k, c), v in sym_rows.items()]).sort_values(["category", "transcripts"], ascending=[True, False])
print("\nEvery symbol that causes a flag, with one example (use this to decide what to accept):")
display(S4)
U[["sample_id", "original_split", "speaker_status", "hours", "categories", "symbols", "transcript_normalized"]].to_csv(CLEAN / "unresolved_symbol_rows_A_v1.csv", index=False, encoding="utf-8-sig")
S4.to_csv(CLEAN / "unresolved_symbol_summary_A_v1.csv", index=False, encoding="utf-8-sig")
print("Saved: cleaning/unresolved_symbol_rows_A_v1.csv and cleaning/unresolved_symbol_summary_A_v1.csv (the review lists)")


# Original zero-based notebook cell 15
print("=" * 100)
print("CELL A5 - RECORD WHAT WAS DONE IN THE DECISION LOG (safe to run several times)")
print("=" * 100)
print("The log keeps three things apart: (1) what was APPLIED, (2) what is NOT linguistically confirmed, (3) what review is still PENDING.\n")
log_decision("4", f"Policy A_v1 applied: transcript_normalized filled for {int(has_text.sum())} transcripts in master_A_v1_2 ({len(anyc)} changed by at least one rule); transcript_raw untouched; master_A_v1_1 kept",
             "approved and provisional rules applied in the planned order; all consistency and character-level checks passed", int(has_text.sum()), round(float(M0.loc[has_text, "hours"].sum()), 3))
log_decision("4", "User confirmed applying the PROVISIONAL rules NORM_005 to NORM_009 (cell A0). They stay marked Provisional: NOT linguistically confirmed by the maintainers or a Nyishi speaker",
             "applying them is reversible because transcript_raw is unchanged; the checks prove the rules were applied correctly, not that the rules are right", None, None)
log_decision("4", "Policy A_v1 is FROZEN: any change to a rule, condition or exception word list means a new version (A_v2), never an edit; master_A_v1_2 is never overwritten with different content",
             "keeps the normalized text reproducible and every change traceable", None, None)
log_decision("4", "Change report and per-row change list saved (cleaning/normalization_change_report_A_v1.csv, cleaning/normalization_row_changes_A_v1.csv)",
             "every changed transcript can be traced to one or more rule IDs", len(CH), None)
log_decision("4", f"Recount after normalization: {len(U)} transcripts ({U['hours'].sum():.2f} h) still contain punctuation, a digit or a rare symbol (a review signal, not proof of error); replaces the preliminary estimate of 168; lists in cleaning/unresolved_symbol_*_A_v1.csv",
             "they are NOT yet flagged in the master table and NOT yet excluded from validation/test; that happens in the next step", len(U), round(float(U["hours"].sum()), 3))
log_decision("4", "REVIEW STILL PENDING: maintainers' answers on the provisional rules; manual listening audit; review of the unresolved transcripts; Roman-script availability",
             "automated checks cannot establish linguistic correctness", None, None)
print("\nThe decision log now has", len(pd.read_csv(LOG, encoding="utf-8-sig")), "entries.")


# Original zero-based notebook cell 16
print("=" * 100)
print("CELL A6 - FINAL CHECKS ON WHAT IS SAVED ON DISK")
print("=" * 100)
v12 = pd.read_parquet(MAN / "master_A_v1_2.parquet"); v11 = pd.read_parquet(MAN / "master_A_v1_1.parquet")
rep_disk = pd.read_csv(CLEAN / "normalization_change_report_A_v1.csv", encoding="utf-8-sig", keep_default_na=False)
rows_disk = pd.read_csv(CLEAN / "normalization_row_changes_A_v1.csv", encoding="utf-8-sig", keep_default_na=False)
unres_disk = pd.read_csv(CLEAN / "unresolved_symbol_rows_A_v1.csv", encoding="utf-8-sig", keep_default_na=False)
log = pd.read_csv(LOG, encoding="utf-8-sig")
J = v11.merge(v12, on="sample_id", how="outer", suffixes=("_11", "_12"), indicator=True, validate="one_to_one")      # compare by sample_id, never by row order
n_text = int((~v11["transcript_missing"]).sum())
ok = {
  "every sample_id is in both versions (compared by sample_id)": bool((J["_merge"] == "both").all()),
  "transcript_raw identical to v1.1, row by row": bool((J["transcript_raw_11"] == J["transcript_raw_12"]).all()),
  "all 54 columns of v1.1 are still in v1.2, plus exactly the 3 new ones": set(v12.columns) == set(v11.columns) | {"normalization_policy_version", "normalization_rules_applied", "normalization_exceptions"},
  "transcript_normalized is filled for exactly the clips that have a transcript": int(v12["transcript_normalized"].notna().sum()) == n_text and bool(v12.loc[v12["transcript_missing"], "transcript_normalized"].isna().all()),
  "master_A_v1_1 was not modified (transcript_normalized still empty there)": int(v11["transcript_normalized"].notna().sum()) == 0,
  "the saved change report is complete: 8 rule lines + 1 total line, same columns as written": len(rep_disk) == 9 and set(["rule_id", "rows_affected", "occurrences_changed", "hours_affected", "exceptions_flagged", "validation_passed"]) <= set(rep_disk.columns),
  "the saved change report agrees with the validation of this run": (rep_disk["validation_passed"] == "Yes").all() == all(checks.values()) and (rep_disk["validation_passed"] != "Not checked").all(),
  "the change list on disk is consistent with the change report": int(rows_disk[rows_disk["kind"] == "changed"]["sample_id"].nunique()) == int(rep_disk.iloc[-1]["rows_affected"]),
  "the review list on disk is readable and has the expected columns": {"sample_id", "categories", "symbols", "transcript_normalized"} <= set(unres_disk.columns) and len(unres_disk) == len(U),
  "decision log has no duplicate decisions": not log["decision"].duplicated().any(),
}
for k, v in ok.items(): print(("PASS  " if v else "FAIL  ") + k)
print("\nRegression checks for THIS data version (they would legitimately change if the dataset changed):")
reg = {"6718 rows": len(v12) == 6718, "57 columns": v12.shape[1] == 57, "5651 transcripts normalized": n_text == 5651}
for k, v in reg.items(): print(("PASS  " if v else "FAIL  ") + k)
assert all(ok.values()) and all(reg.values()), "at least one check failed - read the FAIL lines above"
print("\nAll checks passed. master_A_v1_2 is the table that has the normalized transcripts.")
print("Reminder: 'passed' means the rules were applied correctly. The linguistic review of the provisional rules is still pending.")
