import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Grapheme classes at the Unicode codepoint level. These are ORTHOGRAPHIC sets
# chosen to track the visually-ambiguous phoneme contrasts. Bengali orthography
# is not phonology (inherent vowel, conjuncts, context-dependent readings), so
# state this framing in the paper and verify the sets against a Bengali phonetics
# reference before the camera-ready.
# ---------------------------------------------------------------------------
CLASSES = {
    "bilabial": list("পফবভম"),
    "retroflex_stop": list("টঠডঢণ"),
    "dental_stop": list("তথদধন"),
    "control_vowel": list("অআইঈউঊএঐওঔ"),  # visually distinct control set
}

# Aspirated / unaspirated minimal pairs (invisible on the lips -> LM should help).
ASPIRATION_PAIRS = [
    ("ক", "খ"), ("গ", "ঘ"), ("চ", "ছ"), ("জ", "ঝ"), ("ট", "ঠ"),
    ("ড", "ঢ"), ("ত", "থ"), ("দ", "ধ"), ("প", "ফ"), ("ব", "ভ"),
]


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def edit_distance(a, b):
    m = len(b)
    dp = list(range(m + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return dp[m]


def cer(hyps, refs):
    e = n = 0
    for h, r in zip(hyps, refs):
        h, r = _nfc(h), _nfc(r)
        e += edit_distance(list(h), list(r))
        n += len(r)
    return e / max(n, 1)


def wer(hyps, refs):
    e = n = 0
    for h, r in zip(hyps, refs):
        hw, rw = _nfc(h).split(), _nfc(r).split()
        e += edit_distance(hw, rw)
        n += len(rw)
    return e / max(n, 1)


def align(ref, hyp):
    """Levenshtein backtrace. Returns [(op, ref_char|None, hyp_char|None)] with
    op in {eq, sub, del, ins}."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _class_of(ch):
    for name, chars in CLASSES.items():
        if ch in chars:
            return name
    return None


def class_error_rates(hyps, refs):
    """Per-class error rate among REFERENCE graphemes: (sub+del) / occurrences."""
    counts, errs = Counter(), Counter()
    for h, r in zip(hyps, refs):
        for op, rc, _ in align(list(_nfc(r)), list(_nfc(h))):
            if rc is None:
                continue
            cls = _class_of(rc)
            if cls is None:
                continue
            counts[cls] += 1
            if op in ("sub", "del"):
                errs[cls] += 1
    return {c: errs[c] / counts[c] for c in counts}, dict(counts)


def pair_confusion(hyps, refs, pairs=ASPIRATION_PAIRS):
    """Substitution counts within minimal pairs, e.g. how often প is read as ব."""
    members = set()
    for a, b in pairs:
        members.update((a, b))
    conf = Counter()
    for h, r in zip(hyps, refs):
        for op, rc, hc in align(list(_nfc(r)), list(_nfc(h))):
            if op == "sub" and rc in members and hc is not None:
                conf[(rc, hc)] += 1
    return conf


def report(hyps, refs, label=""):
    rates, counts = class_error_rates(hyps, refs)
    return {
        "label": label,
        "cer": cer(hyps, refs),
        "wer": wer(hyps, refs),
        "class_error": rates,
        "class_counts": counts,
    }
