import torch
import torch.nn as nn


class AttnDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers=2, n_heads=4, d_ff=1024,
                 pad_id=1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_in, memory, memory_pad=None, tgt_pad=None):
        t = tgt_in.size(1)
        x = self.embed(tgt_in) + self.pos[:, :t]
        causal = torch.triu(torch.ones(t, t, device=tgt_in.device, dtype=torch.bool), 1)
        h = self.dec(x, memory, tgt_mask=causal,
                     tgt_key_padding_mask=tgt_pad,
                     memory_key_padding_mask=memory_pad)
        return self.out(h)


class Recognizer(nn.Module):
    """Sits on top of the frozen+LoRA visual encoder. Owns CTC head and optional decoder."""

    def __init__(self, d_in, vocab_size, use_attn=True, pad_id=1):
        super().__init__()
        self.ctc = nn.Linear(d_in, vocab_size)
        self.use_attn = use_attn
        if use_attn:
            self.decoder = AttnDecoder(d_in, vocab_size, pad_id=pad_id)

    def forward(self, feats, feat_pad=None, tgt_in=None, tgt_pad=None):
        ctc_logits = self.ctc(feats)  # [B, T, V]; apply log_softmax in the loss
        attn_logits = None
        if self.use_attn and tgt_in is not None:
            attn_logits = self.decoder(tgt_in, feats, feat_pad, tgt_pad)
        return ctc_logits, attn_logits
