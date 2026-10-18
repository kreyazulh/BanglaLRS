import argparse
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_prep import load_jsonl
from dataset import VSRDataset, collate, LengthBucketSampler
from encoder import AVHubertVisualEncoder
from lora import trainable_parameters
from recognizer import Recognizer
from vocab import GraphemeVocab


def ctc_loss(ctc_logits, vpad, ctc_tgt, tgt_lens, blank_id):
    # ctc_logits [B,T,V] -> log_probs [T,B,V]
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    in_lens = (~vpad).sum(1)
    return F.ctc_loss(log_probs, ctc_tgt, in_lens, tgt_lens,
                      blank=blank_id, zero_infinity=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)      # has roi_cache + text per row
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--ckpt", required=True)          # AV-HuBERT Base LRS3-433h
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--ctc_weight", type=float, default=0.3)
    ap.add_argument("--out", default="pilot.pt")
    a = ap.parse_args()

    dev = "cuda"
    vocab = GraphemeVocab.load(a.vocab)
    rows = load_jsonl(a.manifest)
    ds = VSRDataset(rows, vocab, train=True)
    sampler = LengthBucketSampler(ds.lengths(), batch_size=1, shuffle=True)
    coll = functools.partial(collate, pad_id=vocab.pad_id,
                             bos_id=vocab.bos_id, eos_id=vocab.eos_id)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=coll, num_workers=4)

    encoder = AVHubertVisualEncoder(a.ckpt, lora_rank=a.lora_rank).to(dev)
    recog = Recognizer(encoder.output_dim, len(vocab), use_attn=True,
                       pad_id=vocab.pad_id).to(dev)

    params = trainable_parameters(encoder) + list(recog.parameters())
    print(f"trainable tensors: {len(params)}  lora modules: {encoder.n_lora}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    for ep in range(a.epochs):
        encoder.train(); recog.train()
        opt.zero_grad()
        running = 0.0
        for step, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev, dtype=torch.bfloat16):
                feats, fpad = encoder(b["video"], b["vpad"])
                ctc_logits, attn_logits = recog(
                    feats, feat_pad=fpad, tgt_in=b["dec_in"], tgt_pad=b["dec_pad"])
                lc = ctc_loss(ctc_logits, fpad if fpad is not None else b["vpad"],
                              b["ctc_tgt"], b["tgt_lens"], vocab.blank_id)
                la = F.cross_entropy(attn_logits.reshape(-1, attn_logits.size(-1)),
                                     b["dec_out"].reshape(-1),
                                     ignore_index=vocab.pad_id)
                loss = (a.ctc_weight * lc + (1 - a.ctc_weight) * la) / a.accum
            loss.backward()
            running += loss.item() * a.accum
            if (step + 1) % a.accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
        print(f"epoch {ep} loss {running / max(1, len(dl)):.4f}")
        torch.save({"encoder": encoder.state_dict(),
                    "recog": recog.state_dict()}, a.out)


if __name__ == "__main__":
    main()
