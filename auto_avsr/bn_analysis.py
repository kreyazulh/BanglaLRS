"""
Visual-confusion analysis for BanglaVSR.

Separates two questions the scalar per-class error rate conflates:
  (1) VISIBILITY  - is a place-of-articulation class reproduced at all?
                    measured by per-group hit rate and deletion rate.
  (2) CONFUSION   - WHEN the model substitutes, does it swap within the same
                    place group (i.e. recover the visible viseme but lose the
                    voicing/aspiration/nasal contrast that shares it)?
                    measured by within-place substitution concentration vs a
                    frequency-matched permutation null.

Input: decode_out.jsonl from bn_decode.py, one obj per line with "ref","hyp".
Caveats to state in the paper: Bengali orthography != phonology (inherent vowel,
conjuncts, matras); this works at the grapheme level over base consonants and
treats place groups as orthographic proxies for viseme classes. Verify the sets
against a Bengali phonetics reference for the camera-ready.
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from bn_eval import align, _nfc

# Base consonants grouped by place of articulation. Within each group the
# members differ by voicing / aspiration / nasality - features that are largely
# invisible on the lips. Place itself (esp. labial) is far more visible.
PLACE_GROUPS = {
    "velar":     list("কখগঘঙ"),
    "palatal":   list("চছজঝঞ"),
    "retroflex": list("টঠডঢণ"),
    "dental":    list("তথদধন"),
    "labial":    list("পফবভম"),
}
CHAR2GROUP = {c: g for g, cs in PLACE_GROUPS.items() for c in cs}
CONS = set(CHAR2GROUP)
GROUPS = list(PLACE_GROUPS)

# romanized labels so the char-level heatmap renders without a Bengali font and
# is readable by reviewers. Order within each group = the place block order.
ROMAN = {
    "ক": "ka", "খ": "kha", "গ": "ga", "ঘ": "gha", "ঙ": "nga",
    "চ": "ca", "ছ": "cha", "জ": "ja", "ঝ": "jha", "ঞ": "nya",
    "ট": "Ta", "ঠ": "Tha", "ড": "Da", "ঢ": "Dha", "ণ": "Na",
    "ত": "ta", "থ": "tha", "দ": "da", "ধ": "dha", "ন": "na",
    "প": "pa", "ফ": "pha", "ব": "ba", "ভ": "bha", "ম": "ma",
}
# consonants in place-block order, so within-viseme confusion shows as block-diagonal
CONS_ORDER = [c for g in GROUPS for c in PLACE_GROUPS[g]]


def load_pairs(path, field):
    refs, hyps = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            refs.append(_nfc(o["ref"]))
            hyps.append(_nfc(o[field]))
    return refs, hyps


def collect(refs, hyps):
    """Walk alignments once; gather substitutions and per-consonant op counts."""
    subs = []                 # list of (ref_char, hyp_char) for substitutions
    occ, hit, dele = Counter(), Counter(), Counter()
    for r, h in zip(refs, hyps):
        for op, rc, hc in align(list(r), list(h)):
            if rc in CONS:
                occ[rc] += 1
                if op == "eq":
                    hit[rc] += 1
                elif op == "del":
                    dele[rc] += 1
                elif op == "sub":
                    subs.append((rc, hc))
    return subs, occ, hit, dele


def visibility_table(occ, hit, dele):
    rows = {}
    for g in GROUPS:
        o = sum(occ[c] for c in PLACE_GROUPS[g])
        if o == 0:
            continue
        rows[g] = {
            "occ": o,
            "hit_rate": sum(hit[c] for c in PLACE_GROUPS[g]) / o,
            "del_rate": sum(dele[c] for c in PLACE_GROUPS[g]) / o,
        }
    return rows


def group_matrix(subs):
    """Row = ref place group, col = hyp place group ('other' if hyp not a base
    consonant). Only substitutions whose REF char is a grouped consonant."""
    M = {g: Counter() for g in GROUPS}
    for rc, hc in subs:
        gr = CHAR2GROUP.get(rc)
        if gr is None:
            continue
        M[gr][CHAR2GROUP.get(hc, "other")] += 1
    return M


def within_concentration(subs, n_perm=5000, seed=0):
    """Observed P(hyp group == ref group | substitution, ref is grouped cons),
    vs a permutation null that keeps both marginals (shuffles hyp labels)."""
    ref_g, hyp_g = [], []
    for rc, hc in subs:
        gr = CHAR2GROUP.get(rc)
        if gr is None:
            continue
        ref_g.append(gr)
        hyp_g.append(CHAR2GROUP.get(hc, "other"))
    tot = len(ref_g)
    if tot == 0:
        return None
    obs = sum(a == b for a, b in zip(ref_g, hyp_g)) / tot
    rng = random.Random(seed)
    pool = hyp_g[:]
    ge = 0
    null_vals = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        s = sum(a == b for a, b in zip(ref_g, pool)) / tot
        null_vals.append(s)
        if s >= obs:
            ge += 1
    null_mean = sum(null_vals) / len(null_vals)
    return {"observed": obs, "null_mean": null_mean,
            "ratio": obs / null_mean if null_mean else float("inf"),
            "p_value": (ge + 1) / (n_perm + 1), "n_subs": tot}


def per_group_retention(M):
    """For each ref group, P(hyp same group | substitution out of that group)."""
    out = {}
    for g in GROUPS:
        tot = sum(M[g].values())
        if tot:
            out[g] = {"same_place": M[g][g] / tot, "n": tot}
    return out


def char_matrix(subs):
    """Per-character substitution counts: M[ref_char][hyp_char], consonants only."""
    M = {c: Counter() for c in CONS_ORDER}
    for rc, hc in subs:
        if rc in M and hc in CONS:
            M[rc][hc] += 1
    return M


def write_char_csv(M, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ref\\hyp," + ",".join(CONS_ORDER) + "\n")
        for rc in CONS_ORDER:
            f.write(rc + "," + ",".join(str(M[rc][c]) for c in CONS_ORDER) + "\n")


def char_heatmap(M, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    mat = np.array([[M[r][c] for c in CONS_ORDER] for r in CONS_ORDER], float)
    rn = mat / mat.sum(1, keepdims=True).clip(min=1)        # row-normalize each ref
    labels = [ROMAN[c] for c in CONS_ORDER]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rn, cmap="magma", vmin=0, vmax=rn.max() or 1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("hypothesis"); ax.set_ylabel("reference")
    # draw lines between place blocks so within-viseme blocks are obvious
    b = 0
    for g in GROUPS[:-1]:
        b += len(PLACE_GROUPS[g])
        ax.axhline(b - 0.5, color="cyan", lw=0.8); ax.axvline(b - 0.5, color="cyan", lw=0.8)
    fig.colorbar(im, label="row-normalized substitution share")
    fig.tight_layout(); fig.savefig(path, dpi=200)
    print(f"wrote char heatmap -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", required=True, help="decode_out.jsonl")
    ap.add_argument("--field", default="hyp", choices=["hyp", "hyp_rescored"])
    ap.add_argument("--csv", default="confusion_groups.csv")
    ap.add_argument("--heatmap", default=None, help="optional group-level .png")
    ap.add_argument("--char-csv", default="confusion_chars.csv")
    ap.add_argument("--char-heatmap", default=None, help="optional per-character .png")
    a = ap.parse_args()

    refs, hyps = load_pairs(a.decode, a.field)
    subs, occ, hit, dele = collect(refs, hyps)

    vis = visibility_table(occ, hit, dele)
    M = group_matrix(subs)
    conc = within_concentration(subs)
    ret = per_group_retention(M)

    print("=== VISIBILITY (is the place class reproduced at all) ===")
    print(f"{'group':<10}{'occ':>8}{'hit':>9}{'del':>9}")
    for g in GROUPS:
        if g in vis:
            v = vis[g]
            print(f"{g:<10}{v['occ']:>8}{v['hit_rate']:>9.3f}{v['del_rate']:>9.3f}")

    print("\n=== WITHIN-PLACE SUBSTITUTION CONCENTRATION (the viseme test) ===")
    if conc:
        print(f"observed within-place rate : {conc['observed']:.3f}")
        print(f"frequency-null mean        : {conc['null_mean']:.3f}")
        print(f"ratio (obs/null)           : {conc['ratio']:.2f}x")
        print(f"permutation p-value        : {conc['p_value']:.4f}  (n_subs={conc['n_subs']})")

    print("\n=== per ref group: P(hyp same place | sub) ===")
    for g in GROUPS:
        if g in ret:
            print(f"{g:<10}{ret[g]['same_place']:.3f}   (n={ret[g]['n']})")

    cols = GROUPS + ["other"]
    with open(a.csv, "w", encoding="utf-8") as f:
        f.write("ref_group," + ",".join(cols) + "\n")
        for g in GROUPS:
            f.write(g + "," + ",".join(str(M[g][c]) for c in cols) + "\n")
    print(f"\nwrote group confusion matrix -> {a.csv}")

    CM = char_matrix(subs)
    write_char_csv(CM, a.char_csv)
    print(f"wrote per-character confusion matrix -> {a.char_csv}")
    if a.char_heatmap:
        try:
            char_heatmap(CM, a.char_heatmap)
        except Exception as e:
            print("char heatmap skipped:", e)

    if a.heatmap:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            mat = np.array([[M[r][c] for c in GROUPS] for r in GROUPS], float)
            rn = mat / mat.sum(1, keepdims=True).clip(min=1)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(rn, cmap="magma")
            ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, rotation=45, ha="right")
            ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS)
            ax.set_xlabel("hypothesis place"); ax.set_ylabel("reference place")
            for i in range(len(GROUPS)):
                for j in range(len(GROUPS)):
                    ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center",
                            color="white" if rn[i, j] < 0.5 else "black", fontsize=8)
            fig.colorbar(im, label="row-normalized substitution share")
            fig.tight_layout(); fig.savefig(a.heatmap, dpi=200)
            print(f"wrote heatmap -> {a.heatmap}")
        except Exception as e:
            print("heatmap skipped:", e)


if __name__ == "__main__":
    main()