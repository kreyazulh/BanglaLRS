import argparse
import json

from bn_text_transform import build_token_list


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True,
                    help="jsonl splits to collect characters from (use train only, or all)")
    ap.add_argument("--out", default="bn_tokens.json")
    a = ap.parse_args()

    texts = []
    for m in a.manifests:
        texts += [r["text"] for r in load_jsonl(m)]
    tl = build_token_list(texts, a.out)
    print(f"{len(texts)} transcripts -> {len(tl)} tokens (incl <blank>,<unk>,<eos>) -> {a.out}")
    print("blank id 0, eos id", len(tl) - 1, ", unk id", len(tl) - 2)


if __name__ == "__main__":
    main()
