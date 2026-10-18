import json
import unicodedata


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def char_rate(text, duration):
    return len(text) / duration if duration and duration > 0 else 0.0


def filter_rows(rows, min_conf=0.5, min_cr=2.0, max_cr=25.0):
    """Drop empty text, low confidence, implausible char-rate. Returns kept rows with NFC text."""
    kept, dropped = [], {"empty": 0, "conf": 0, "char_rate": 0}
    for r in rows:
        text = unicodedata.normalize("NFC", (r.get("text") or "").strip())
        if not text:
            dropped["empty"] += 1
            continue
        if float(r.get("confidence", 0.0)) < min_conf:
            dropped["conf"] += 1
            continue
        cr = char_rate(text, float(r.get("duration", 0.0)))
        if cr < min_cr or cr > max_cr:
            dropped["char_rate"] += 1
            continue
        r = dict(r)
        r["text"] = text
        kept.append(r)
    return kept, dropped


def dual_asr_agreement(rows, second_text_key="text_b", max_cer=0.25):
    """Optional: keep clips where two ASR transcripts agree (low CER). Needs a second pass field."""
    from evaluate import cer
    kept = []
    for r in rows:
        b = r.get(second_text_key)
        if b is None:
            kept.append(r)
            continue
        if cer([r["text"]], [b]) <= max_cer:
            kept.append(r)
    return kept


if __name__ == "__main__":
    import argparse
    from vocab import GraphemeVocab

    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--out_vocab", required=True)
    a = ap.parse_args()

    rows = load_jsonl(a.transcripts)
    kept, dropped = filter_rows(rows)
    print(f"kept {len(kept)} / {len(rows)}  dropped={dropped}")
    write_jsonl(kept, a.out_manifest)
    GraphemeVocab.build([r["text"] for r in kept]).save(a.out_vocab)
