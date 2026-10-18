import random

import torch
from torch.utils.data import Dataset, Sampler

# AV-HuBERT grayscale normalization (verify against your checkpoint's data config).
MEAN, STD = 0.421, 0.165


def transform(roi_u8, train, crop=88):
    """roi_u8: uint8 [T,96,96] -> float [T,1,88,88], normalized.
    Train: random crop + horizontal flip. Eval: center crop. Matches AV-HuBERT."""
    x = roi_u8.float() / 255.0
    _, h, w = x.shape
    if train:
        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
    else:
        top = (h - crop) // 2
        left = (w - crop) // 2
    x = x[:, top:top + crop, left:left + crop]
    if train and random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    x = (x - MEAN) / STD
    return x.unsqueeze(1)  # [T,1,88,88]


class VSRDataset(Dataset):
    """Manifest rows need: roi_cache (path to cached [T,96,96] uint8) and text."""

    def __init__(self, rows, vocab, train=True):
        self.rows = rows
        self.vocab = vocab
        self.train = train

    def __len__(self):
        return len(self.rows)

    def lengths(self):
        return [int(r.get("n_frames", r.get("duration", 4.0) * 25)) for r in self.rows]

    def __getitem__(self, i):
        r = self.rows[i]
        roi = torch.load(r["roi_cache"])  # [T,96,96] uint8
        video = transform(roi, self.train)  # [T,1,88,88]
        ids = self.vocab.encode(r["text"])
        return video, torch.tensor(ids, dtype=torch.long)


def collate(batch, pad_id, bos_id, eos_id):
    vids, idss = zip(*batch)
    T = max(v.size(0) for v in vids)
    B = len(vids)
    video = vids[0].new_zeros(B, T, 1, 88, 88)
    vpad = torch.ones(B, T, dtype=torch.bool)  # True = pad
    for b, v in enumerate(vids):
        video[b, :v.size(0)] = v
        vpad[b, :v.size(0)] = False
    video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B,1,T,88,88]

    in_lens = torch.tensor([v.size(0) for v in vids], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in idss], dtype=torch.long)
    Smax = int(tgt_lens.max())
    ctc_tgt = torch.full((B, Smax), pad_id, dtype=torch.long)
    for b, x in enumerate(idss):
        ctc_tgt[b, :len(x)] = x

    # attention decoder teacher forcing: [bos] + ids , ids + [eos]
    S2 = Smax + 1
    dec_in = torch.full((B, S2), pad_id, dtype=torch.long)
    dec_out = torch.full((B, S2), pad_id, dtype=torch.long)
    for b, x in enumerate(idss):
        dec_in[b, 0] = bos_id
        dec_in[b, 1:1 + len(x)] = x
        dec_out[b, :len(x)] = x
        dec_out[b, len(x)] = eos_id
    dec_pad = dec_in.eq(pad_id)
    return {
        "video": video, "vpad": vpad, "in_lens": in_lens,
        "ctc_tgt": ctc_tgt, "tgt_lens": tgt_lens,
        "dec_in": dec_in, "dec_out": dec_out, "dec_pad": dec_pad,
    }


class LengthBucketSampler(Sampler):
    """Group similar-length clips so short clips are not padded to 8 s."""

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.bs = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs
