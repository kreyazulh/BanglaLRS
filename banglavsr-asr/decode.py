import math
from collections import defaultdict

NEG = -1e30


def _lse(*xs):
    xs = [x for x in xs if x > NEG]
    if not xs:
        return NEG
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def ctc_beam_search(log_probs, blank_id, beam_size=25, n_best=10):
    """Standard prefix beam search (Hannun et al. formulation).
    log_probs: [T][V] log-softmax over the grapheme vocab for one utterance.
    Returns up to n_best (id_tuple, acoustic_log_score), best first.

    This is a correct reference implementation. For large-scale runs you can swap
    in torchaudio.models.decoder.ctc_decoder; the rescorer only needs (ids, score)."""
    T, V = len(log_probs), len(log_probs[0])
    beam = {(): [0.0, NEG]}  # prefix -> [p_blank, p_nonblank]
    for t in range(T):
        lp = log_probs[t]
        cand = sorted(range(V), key=lambda s: lp[s], reverse=True)[:beam_size]
        nxt = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beam.items():
            ptot = _lse(pb, pnb)
            b0, nb0 = nxt[prefix]
            nxt[prefix] = [_lse(b0, ptot + lp[blank_id]), nb0]  # emit blank, stay
            last = prefix[-1] if prefix else None
            for c in cand:
                if c == blank_id:
                    continue
                pc = lp[c]
                if c == last:
                    np_ = prefix + (c,)
                    b1, nb1 = nxt[np_]
                    nxt[np_] = [b1, _lse(nb1, pb + pc)]      # repeat only via blank
                    b2, nb2 = nxt[prefix]
                    nxt[prefix] = [b2, _lse(nb2, pnb + pc)]  # merge repeat, stay
                else:
                    np_ = prefix + (c,)
                    b3, nb3 = nxt[np_]
                    nxt[np_] = [b3, _lse(nb3, ptot + pc)]
        beam = dict(sorted(nxt.items(), key=lambda kv: _lse(*kv[1]), reverse=True)[:beam_size])
    out = [(p, _lse(b, nb)) for p, (b, nb) in beam.items()]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:n_best]
