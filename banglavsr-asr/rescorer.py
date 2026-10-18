import torch
import torch.nn.functional as F


class MLMRescorer:
    """Pseudo-log-likelihood scorer (Salazar et al. 2020): mask each token in turn,
    sum the log-prob of the true token. Requires a real masked-LM with a fill-mask head.
    csebuetnlp/banglabert is ELECTRA and will NOT work here. Use e.g.
    sagorsarker/bangla-bert-base, ai4bharat/IndicBERTv2, or xlm-roberta-base."""

    def __init__(self, model_name, device="cuda"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        if self.tok.mask_token_id is None:
            raise ValueError(f"{model_name} has no mask token; not a fill-mask model.")

    @torch.no_grad()
    def pll(self, text, batch_size=64):
        enc = self.tok(text, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        n = ids.size(1)
        positions = list(range(1, n - 1))  # skip [CLS]/[SEP]-style specials
        if not positions:
            return -1e9
        total = 0.0
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            batch = ids.repeat(len(chunk), 1)
            for i, pos in enumerate(chunk):
                batch[i, pos] = self.tok.mask_token_id
            logits = self.model(input_ids=batch,
                                attention_mask=attn.repeat(len(chunk), 1)).logits
            lp = F.log_softmax(logits, dim=-1)
            for i, pos in enumerate(chunk):
                total += lp[i, pos, ids[0, pos]].item()
        return total


def rescore(nbest_ids, acoustic_scores, vocab, rescorer, lam, pll_cache=None):
    """Pick the best candidate by acoustic + lam * PLL. nbest_ids are grapheme id tuples."""
    best, best_s = None, -1e30
    for k, (ids, ac) in enumerate(zip(nbest_ids, acoustic_scores)):
        text = vocab.decode(list(ids))
        pl = pll_cache[k] if pll_cache is not None else rescorer.pll(text)
        s = ac + lam * pl
        if s > best_s:
            best_s, best = s, text
    return best


def tune_lambda(val_items, vocab, rescorer, cer_fn, grid=None):
    """val_items: [{'nbest_ids': [...], 'scores': [...], 'ref': str}].
    Returns (best_lambda, best_cer). PLLs are computed once and cached."""
    if grid is None:
        grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    pll_cache = [[rescorer.pll(vocab.decode(list(ids))) for ids in it["nbest_ids"]]
                 for it in val_items]
    best_lam, best_cer = 0.0, 1e9
    for lam in grid:
        hyps, refs = [], []
        for it, plls in zip(val_items, pll_cache):
            scores = [ac + lam * pl for ac, pl in zip(it["scores"], plls)]
            j = max(range(len(scores)), key=lambda i: scores[i])
            hyps.append(vocab.decode(list(it["nbest_ids"][j])))
            refs.append(it["ref"])
        c = cer_fn(hyps, refs)
        if c < best_cer:
            best_cer, best_lam = c, lam
    return best_lam, best_cer
