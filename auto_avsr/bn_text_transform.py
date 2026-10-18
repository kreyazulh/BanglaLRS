import json
import unicodedata

import torch


class BnTextTransform:
    """Character-level Bengali tokenizer compatible with auto_avsr's E2E model.
    Mirrors their token_list contract: index 0 is <blank> (CTC), the last index is
    <eos> (the model sets sos == eos == odim-1). <unk> sits just before <eos>."""

    def __init__(self, token_list_path):
        with open(token_list_path, encoding="utf-8") as f:
            self.token_list = json.load(f)
        self.stoi = {t: i for i, t in enumerate(self.token_list)}
        self.blank = 0
        self.eos = len(self.token_list) - 1
        self.unk = self.stoi["<unk>"]
        self.ignore_id = -1

    def tokenize(self, text):
        text = unicodedata.normalize("NFC", text)
        ids = [self.stoi.get(ch, self.unk) for ch in text]
        return torch.tensor(ids, dtype=torch.long)

    def post_process(self, token_ids):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        skip = {self.blank, self.eos, self.unk, self.ignore_id}
        return "".join(self.token_list[i] for i in token_ids if i not in skip).strip()


def build_token_list(texts, out_path):
    """Build a character vocab from transcripts. Includes space as a real unit so
    word boundaries survive char-level decoding."""
    chars = set()
    for t in texts:
        for ch in unicodedata.normalize("NFC", t):
            chars.add(ch)
    units = sorted(chars) + ["<unk>"]
    token_list = ["<blank>"] + units + ["<eos>"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(token_list, f, ensure_ascii=False)
    return token_list
