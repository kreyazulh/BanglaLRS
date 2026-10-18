#!/usr/bin/env python3
"""
BanglaVSR dataset statistics.

Reads the transcript manifest (one JSON object per line) and reports the
corpus statistics a dataset paper needs:

  * total duration and per-clip duration distribution
  * vocabulary size, running-word (token) count, type/token ratio, hapax
  * words-per-utterance and characters-per-utterance distributions
  * speaking rate (words/sec, chars/sec) overall and per utterance
  * pseudo-label confidence distribution and low-confidence counts
  * source-video and face-track diversity (parsed from clip_id)
  * the character set (relevant to the char-level vocab / odim)

Each clip is counted once: one manifest record is one clip, so the mp4-vs-wav
duplication never arises. Durations come from the manifest `duration` field;
pass --verify-media to spot-check a random sample against ffprobe.

Usage:
    python dataset_stats.py /path/to/transcripts_bn.jsonl
    python dataset_stats.py transcripts_bn.jsonl --out-json stats.json --out-vocab vocab.tsv
    python dataset_stats.py transcripts_bn.jsonl --clips-dir /path/to/clips --verify-media 50
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter

try:
    import regex  # optional: proper Unicode grapheme-cluster counting
    _HAS_REGEX = True
except ImportError:
    _HAS_REGEX = False

# Characters stripped from token edges when building the word vocabulary.
# Bengali danda/double-danda and common ASCII/Unicode punctuation.
_STRIP = "।॥…“”‘’\"'()[]{}<>/\\|`~@#$%^&*_=+.,!?;:" + " \t"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def fmt_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m:02d}m {s:04.1f}s"


def grapheme_len(s):
    """Length in user-perceived characters (grapheme clusters) if `regex` is
    available, else falls back to Unicode code points."""
    if _HAS_REGEX:
        return len(regex.findall(r"\X", s))
    return len(s)


def tokenize(text):
    """Whitespace split, then strip edge punctuation. Bengali separates words
    with spaces, so this is the standard corpus-stat tokenization."""
    out = []
    for raw in text.split():
        t = raw.strip(_STRIP)
        if t:
            out.append(t)
    return out


def split_clip_id(cid):
    """clip_id == {video_id}_{track}_{frame}; video_id itself may contain
    underscores/hyphens, so split the two trailing fields off from the right."""
    parts = cid.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return cid, "", ""


def describe(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s),
        "sum": float(sum(s)),
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min": float(s[0]),
        "p5": percentile(s, 5),
        "p25": percentile(s, 25),
        "median": percentile(s, 50),
        "p75": percentile(s, 75),
        "p95": percentile(s, 95),
        "max": float(s[-1]),
    }


def print_describe(name, d, unit=""):
    if d is None:
        print(f"  {name}: (no data)")
        return
    u = f" {unit}" if unit else ""
    print(
        f"  {name}: mean {d['mean']:.3f}{u}  median {d['median']:.3f}  "
        f"std {d['std']:.3f}  min {d['min']:.3f}  max {d['max']:.3f}"
    )
    print(
        f"       p5 {d['p5']:.3f}  p25 {d['p25']:.3f}  "
        f"p75 {d['p75']:.3f}  p95 {d['p95']:.3f}"
    )


def histogram(vals, edges):
    counts = [0] * (len(edges) + 1)
    for v in vals:
        for i, e in enumerate(edges):
            if v < e:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    rows, prev = [], 0
    for e, c in zip(edges, counts[:-1]):
        rows.append((f"[{prev:g}, {e:g}) s", c))
        prev = e
    rows.append((f"[{prev:g}, inf) s", counts[-1]))
    return rows


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_records(path):
    records, n_lines, n_bad, n_dup = {}, 0, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            cid = obj.get("clip_id")
            if not cid:
                p = obj.get("mp4_path") or obj.get("wav_path") or ""
                cid = os.path.splitext(os.path.basename(p))[0] or f"_row{n_lines}"
            if cid in records:
                n_dup += 1
                continue
            records[cid] = obj
    return records, n_lines, n_bad, n_dup


def ffprobe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="transcripts_bn.jsonl")
    ap.add_argument("--top", type=int, default=25, help="how many frequent words to print")
    ap.add_argument("--conf-thresholds", default="0.5,0.9,0.95",
                    help="comma-separated confidence thresholds to count below")
    ap.add_argument("--out-json", help="write the full summary as JSON")
    ap.add_argument("--out-vocab", help="write 'word<TAB>count' sorted by frequency")
    ap.add_argument("--clips-dir", help="cross-check manifest against files on disk")
    ap.add_argument("--verify-media", type=int, metavar="K", default=0,
                    help="ffprobe K random clips to check manifest durations")
    args = ap.parse_args()

    records, n_lines, n_bad, n_dup = load_records(args.manifest)
    if not records:
        sys.exit("No valid records found.")

    durations, confidences = [], []
    word_counts, char_counts, char_rates = [], [], []
    vocab, charset = Counter(), Counter()
    videos, tracks = Counter(), Counter()
    empty_text = no_duration = 0

    for cid, obj in records.items():
        vid, trk, _ = split_clip_id(cid)
        videos[vid] += 1
        tracks[(vid, trk)] += 1

        dur = obj.get("duration")
        if isinstance(dur, (int, float)):
            durations.append(float(dur))
        else:
            no_duration += 1
            dur = None

        conf = obj.get("confidence")
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))

        text = unicodedata.normalize("NFC", obj.get("text", "") or "")
        toks = tokenize(text)
        if not toks:
            empty_text += 1
        word_counts.append(len(toks))
        vocab.update(toks)

        nchar = grapheme_len(text.replace(" ", ""))
        char_counts.append(nchar)
        for ch in text:
            if not ch.isspace():
                charset[ch] += 1
        if dur and dur > 0:
            char_rates.append(nchar / dur)

    total_dur = sum(durations)
    total_tokens = sum(vocab.values())
    n_types = len(vocab)
    hapax = sum(1 for c in vocab.values() if c == 1)

    # ----------------------------------------------------------------- report
    print("=" * 70)
    print("BanglaVSR dataset statistics")
    print("=" * 70)
    print(f"manifest                : {args.manifest}")
    print(f"grapheme counting       : {'regex \\X clusters' if _HAS_REGEX else 'code points (pip install regex for clusters)'}")
    print(f"lines read              : {n_lines}")
    print(f"malformed JSON lines    : {n_bad}")
    print(f"duplicate clip_ids      : {n_dup}")
    print(f"unique clips (records)  : {len(records)}")
    if no_duration:
        print(f"records missing duration: {no_duration}")
    if empty_text:
        print(f"records with empty text : {empty_text}")

    print("\n--- Duration ---")
    print(f"  total : {total_dur:.1f} s  =  {total_dur/3600:.2f} h  ({fmt_hms(total_dur)})")
    print_describe("per clip", describe(durations), "s")
    print("  histogram:")
    for label, c in histogram(durations, [1, 2, 3, 4, 5, 6, 8]):
        bar = "#" * int(50 * c / max(1, len(durations)))
        print(f"    {label:>14}: {c:>7}  {bar}")

    print("\n--- Text / vocabulary ---")
    print(f"  running words (tokens)   : {total_tokens}")
    print(f"  vocabulary (word types)  : {n_types}")
    print(f"  type/token ratio         : {n_types/total_tokens:.4f}" if total_tokens else "  type/token ratio: n/a")
    print(f"  hapax legomena (freq==1) : {hapax}"
          f"  ({100*hapax/max(1,n_types):.1f}% of vocab)")
    print(f"  unique characters        : {len(charset)}")
    print_describe("words / utterance", describe(word_counts), "w")
    print_describe("chars / utterance", describe(char_counts), "ch")

    print("\n--- Speaking rate ---")
    if total_dur > 0:
        print(f"  overall words/sec : {total_tokens/total_dur:.2f}")
        print(f"  overall chars/sec : {sum(char_counts)/total_dur:.2f}")
    print_describe("chars/sec per utt", describe(char_rates), "ch/s")

    print("\n--- Pseudo-label confidence ---")
    print_describe("confidence", describe(confidences))
    if confidences:
        for thr in [float(x) for x in args.conf_thresholds.split(",") if x.strip()]:
            n = sum(1 for c in confidences if c < thr)
            print(f"  clips with confidence < {thr:g}: {n}  ({100*n/len(confidences):.2f}%)")

    print("\n--- Source diversity ---")
    print(f"  distinct source videos   : {len(videos)}")
    print(f"  distinct face tracks     : {len(tracks)}  (video_id + track)")
    print_describe("clips / video", describe(list(videos.values())))
    print_describe("clips / track", describe(list(tracks.values())))
    print("  NOTE: speaker counts come from the ArcFace speaker_split, not the")
    print("        manifest. video/track counts above are upper-bound proxies.")

    print(f"\n--- Top {args.top} words ---")
    for w, c in vocab.most_common(args.top):
        print(f"  {c:>7}  {w}")

    # ----------------------------------------------------------------- disk check
    if args.clips_dir:
        on_disk = set()
        for fn in os.listdir(args.clips_dir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in (".mp4", ".wav"):
                on_disk.add(stem)
        ids = set(records)
        missing = ids - on_disk
        orphan = on_disk - ids
        print("\n--- Manifest vs disk ---")
        print(f"  clip stems on disk          : {len(on_disk)}")
        print(f"  in manifest, no file on disk: {len(missing)}")
        print(f"  on disk, not in manifest    : {len(orphan)}")
        for x in list(missing)[:5]:
            print(f"    missing: {x}")
        for x in list(orphan)[:5]:
            print(f"    orphan : {x}")

    # ----------------------------------------------------------------- ffprobe
    if args.verify_media:
        sample = random.sample(list(records.values()),
                               min(args.verify_media, len(records)))
        diffs = []
        checked = 0
        for obj in sample:
            path = obj.get("mp4_path") or obj.get("wav_path")
            man = obj.get("duration")
            if not path or not os.path.exists(path) or not isinstance(man, (int, float)):
                continue
            probed = ffprobe_duration(path)
            if probed is not None:
                diffs.append(abs(probed - man))
                checked += 1
        print("\n--- Media duration check (ffprobe) ---")
        if checked:
            print(f"  checked {checked} clips; mean |manifest - probed| = "
                  f"{statistics.fmean(diffs):.4f} s, max = {max(diffs):.4f} s")
        else:
            print("  no clips could be probed (ffprobe missing or paths absent)")

    # ----------------------------------------------------------------- outputs
    if args.out_vocab:
        with open(args.out_vocab, "w", encoding="utf-8") as f:
            for w, c in vocab.most_common():
                f.write(f"{w}\t{c}\n")
        print(f"\nwrote vocabulary -> {args.out_vocab}")

    if args.out_json:
        summary = {
            "n_clips": len(records),
            "total_duration_sec": total_dur,
            "total_duration_hours": total_dur / 3600 if total_dur else 0,
            "duration": describe(durations),
            "running_words": total_tokens,
            "vocab_size": n_types,
            "type_token_ratio": (n_types / total_tokens) if total_tokens else None,
            "hapax": hapax,
            "unique_chars": len(charset),
            "words_per_utt": describe(word_counts),
            "chars_per_utt": describe(char_counts),
            "chars_per_sec": describe(char_rates),
            "confidence": describe(confidences),
            "distinct_videos": len(videos),
            "distinct_tracks": len(tracks),
            "clips_per_video": describe(list(videos.values())),
            "clips_per_track": describe(list(tracks.values())),
            "n_malformed_lines": n_bad,
            "n_duplicate_ids": n_dup,
            "n_empty_text": empty_text,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"wrote summary -> {args.out_json}")


if __name__ == "__main__":
    main()