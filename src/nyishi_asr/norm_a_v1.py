"""Historical a_v1 functions; provisional research policy, not expert corrections."""
import re
import unicodedata as ud

VOWELS = set("aeiouəɛɔɪʊɯɨʌæɑɒɐɜɘɵœʏ")                        # vowel letters used in the transcripts (extend this set if you find a missing vowel)


def prev_base(t, i):
    """The nearest letter before position i, skipping combining marks (accents)."""
    j = i - 1
    while j >= 0 and ud.category(t[j]).startswith("M"): j -= 1
    return t[j] if j >= 0 else ""


def r_nfd(t):                                         # NORM_001: one Unicode form
    new = ud.normalize("NFD", t); return new, int(new != t), 0, []


def r_trim(t):                                        # NORM_002: remove leading/trailing whitespace
    new = t.strip(); return new, int(new != t), 0, []


def r_collapse(t):                                    # NORM_003: several ordinary spaces in a row -> one space
    new = re.sub(r" {2,}", " ", t); return new, len(t) - len(new), 0, []      # 'changed' = number of spaces removed


def make_length_rule(sym):                            # NORM_005 (':') and NORM_006 (';'): -> ː  only right after a vowel
    def rule(t):
        out, ch, exc, det = [], 0, 0, []
        for i, c in enumerate(t):
            if c == sym:
                if prev_base(t, i) in VOWELS: out.append("\u02d0"); ch += 1
                else: out.append(c); exc += 1; det.append(t[max(0, i - 8): i + 8])
            else: out.append(c)
        return "".join(out), ch, exc, det
    return rule


def r_bracket(t):                                     # NORM_007: '[' -> combining dental mark U+032A, only right after t or d
    out, ch, exc, det = [], 0, 0, []
    for i, c in enumerate(t):
        if c == "[":
            if i > 0 and t[i - 1] in "td": out.append("\u032a"); ch += 1
            else: out.append(c); exc += 1; det.append(t[max(0, i - 8): i + 8])
        else: out.append(c)
    return "".join(out), ch, exc, det


G_EXCEPTION_WORDS = {"english", "government", "governmente", "assignment", "bekgraound", "generationga", "generationŋul", "exercisega", "exerciseiga"}


def r_g(t):                                           # NORM_008: Latin g -> IPA script ɡ (U+0261), except in the words listed above
    out, ch, exc, det = [], 0, 0, []                  # An explicit word list (not a capital-letter test) keeps the rule stable:
    for w in t.split(" "):                            # applying it twice, or to a model's lowercase output, gives the same result.
        n = w.count("g")
        if n and w.lower() in G_EXCEPTION_WORDS: out.append(w); exc += n; det.append(w)
        else: out.append(w.replace("g", "ɡ")); ch += n
    return " ".join(out), ch, exc, det


def r_lower(t):                                       # NORM_009: ASCII capital letters A-Z -> lowercase (nothing else is touched)
    new = re.sub(r"[A-Z]", lambda m: m.group().lower(), t); return new, sum(a != b for a, b in zip(t, new)), 0, []


PIPELINE = [("NORM_001", r_nfd), ("NORM_002", r_trim), ("NORM_003", r_collapse), ("NORM_005", make_length_rule(":")), ("NORM_006", make_length_rule(";")),
            ("NORM_007", r_bracket), ("NORM_008", r_g), ("NORM_009", r_lower), ("NORM_001b", r_nfd)]        # the planned order; NFD is applied again at the end


def normalize_text(text):
    for _, rule in PIPELINE:
        text = rule(text)[0]
    return text
