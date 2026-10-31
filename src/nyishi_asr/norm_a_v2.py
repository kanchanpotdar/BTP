"""Historical a_v2 functions; provisional research policy, not expert corrections."""
import re
import unicodedata as ud

VOWELS2 = set("aeiouəɛɔɪʊɯɨʌæɑɒɐɜɘɵœʏʉ")                      # ONE shared vowel set (ʉ added). 'y' (19x) and 'ᵾ' (1x) are NOT added: their role is not established


G_EXCEPTION_WORDS2 = {"english", "government", "governmente", "assignment", "bekgraound", "generationga", "generationŋul", "exercisega", "exerciseiga"}   # unchanged list


PUNCT_EDGE = ".,;:?!\"'()[]=\\"                                # punctuation that may be glued to a word; used only to MATCH the exception words


def prev_base2(t, i):
    """The nearest letter before position i, skipping combining marks (accents)."""
    j = i - 1
    while j >= 0 and ud.category(t[j]).startswith("M"): j -= 1
    return t[j] if j >= 0 else ""


def r_nfd2(t):                                         # NORM_001: one Unicode form
    new = ud.normalize("NFD", t); return new, int(new != t), 0, []


def r_trim2(t):                                        # NORM_002: remove leading/trailing whitespace
    new = t.strip(); return new, int(new != t), 0, []


def r_collapse2(t):                                    # NORM_003: several ordinary spaces in a row -> one space
    new = re.sub(r" {2,}", " ", t); return new, len(t) - len(new), 0, []


def r_dot2(t):                                         # NORM_010 (NEW in A_v2): delete a full stop that stands between two symbols of one word
    out, ch, exc, det = [], 0, 0, []
    for i, c in enumerate(t):
        if c == ".":
            if 0 < i < len(t) - 1 and t[i - 1] != " " and t[i + 1] != " ": ch += 1               # inside a word: delete
            else: out.append(c); exc += 1; det.append(t[max(0, i - 8): i + 8])                     # anywhere else: keep it and flag it
        else: out.append(c)
    return "".join(out), ch, exc, det


def r_comma2(t):                                       # NORM_019 (NEW in A_v2): delete every comma
    n = t.count(","); return t.replace(",", ""), n, 0, []


def make_length_rule2(sym):                            # NORM_005 (':') and NORM_006 (';'): -> ː  only right after a vowel (shared set, accents skipped)
    def rule(t):
        out, ch, exc, det = [], 0, 0, []
        for i, c in enumerate(t):
            if c == sym:
                if prev_base2(t, i) in VOWELS2: out.append("\u02d0"); ch += 1
                else: out.append(c); exc += 1; det.append(t[max(0, i - 8): i + 8])
            else: out.append(c)
        return "".join(out), ch, exc, det
    return rule


def r_bracket2(t):                                     # NORM_007: '[' -> dental mark U+032A right after t or d; accents in between are skipped
    out, ch, exc, det = [], 0, 0, []
    for i, c in enumerate(t):
        if c == "[":
            j = i - 1
            while j >= 0 and ud.category(t[j]).startswith("M"): j -= 1
            between = t[j + 1: i]                      # accents between the letter and '['
            if j >= 0 and t[j] in ("t", "d") and "\u032a" not in between: out.append("\u032a"); ch += 1      # never create a second dental mark on the same letter
            else: out.append(c); exc += 1; det.append(t[max(0, i - 8): i + 8])
        else: out.append(c)
    return "".join(out), ch, exc, det


def r_g2(t):                                           # NORM_008: Latin g -> IPA script ɡ (U+0261), except in the English words of the list
    out, ch, exc, det = [], 0, 0, []                   # the match ignores punctuation glued to the word ('english,' is still 'english')
    for w in t.split(" "):
        n = w.count("g")
        if n and w.strip(PUNCT_EDGE).lower() in G_EXCEPTION_WORDS2: out.append(w); exc += n; det.append(w)
        else: out.append(w.replace("g", "\u0261")); ch += n
    return " ".join(out), ch, exc, det


def r_lower2(t):                                       # NORM_009: ASCII capital letters A-Z -> lowercase
    new = re.sub(r"[A-Z]", lambda m: m.group().lower(), t); return new, sum(a != b for a, b in zip(t, new)), 0, []


PIPELINE2 = [("NORM_001", r_nfd2), ("NORM_010", r_dot2), ("NORM_019", r_comma2), ("NORM_002", r_trim2), ("NORM_003", r_collapse2),
             ("NORM_005", make_length_rule2(":")), ("NORM_006", make_length_rule2(";")), ("NORM_007", r_bracket2), ("NORM_008", r_g2), ("NORM_009", r_lower2), ("NORM_001b", r_nfd2)]


def normalize_text2(t):
    """The whole A_v2 pipeline for ONE transcript."""
    for rid, fn in PIPELINE2: t = fn(t)[0]
    return t


def v1_to_v2(s):
    s = re.sub(r"(?<=\S)\.(?=\S)", "", s).replace(",", "")
    return re.sub(" {2,}", " ", s.strip())
