import json
import unicodedata

# blank MUST be index 0 for CTC. pad MUST be a fixed index for padding_idx in embeddings.
BLANK, PAD, UNK, MASK, BOS, EOS = "<blank>", "<pad>", "<unk>", "<mask>", "<bos>", "<eos>"
SPECIALS = [BLANK, PAD, UNK, MASK, BOS, EOS]


class GraphemeVocab:
    def __init__(self, chars):
        self.itos = list(SPECIALS) + [c for c in chars if c not in SPECIALS]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.blank_id = self.stoi[BLANK]
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.mask_id = self.stoi[MASK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    @classmethod
    def build(cls, texts):
        chars = set()
        for t in texts:
            for ch in unicodedata.normalize("NFC", t):
                chars.add(ch)
        return cls(sorted(chars))

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if self.itos[i] not in SPECIALS)

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            itos = json.load(f)
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = {c: i for i, c in enumerate(itos)}
        obj.blank_id = obj.stoi[BLANK]
        obj.pad_id = obj.stoi[PAD]
        obj.unk_id = obj.stoi[UNK]
        obj.mask_id = obj.stoi[MASK]
        obj.bos_id = obj.stoi[BOS]
        obj.eos_id = obj.stoi[EOS]
        return obj
