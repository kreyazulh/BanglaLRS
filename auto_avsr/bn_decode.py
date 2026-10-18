import argparse
import json

import torch

import bn_eval as evaluate  # our metrics module, renamed to avoid the HF `evaluate` package
from bn_dataset import BnVideoDataset
from bn_text_transform import BnTextTransform
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from lightning import get_beam_search_decoder


def load_model(ckpt_path, token_list_path, ctc_weight, device):
    tt = BnTextTransform(token_list_path)
    model = E2E(len(tt.token_list), "video", ctc_weight=ctc_weight)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)  # Lightning checkpoint stores under state_dict
    msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(msd, strict=False)
    if missing:
        print("missing keys (first few):", missing[:5])
    model.eval().to(device)
    return model, tt


@torch.no_grad()
def decode_one(model, beam, video, device, nbest):
    x = model.frontend(video.unsqueeze(0).to(device))
    x = model.proj_encoder(x)
    enc, _ = model.encoder(x, None)
    enc = enc.squeeze(0)
    hyps = beam(enc)
    out = []
    for h in hyps[:nbest]:
        ids = h.yseq[1:].tolist()  # drop sos; post_process drops eos
        out.append((ids, float(h.score)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)            # exp_bn/last.ckpt or best epoch
    ap.add_argument("--manifest", required=True)        # val or test split jsonl
    ap.add_argument("--token-list", required=True)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--beam-size", type=int, default=20)
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)     # 0 = whole set; small N for a quick look
    ap.add_argument("--out", default="decode_out.jsonl")
    ap.add_argument("--mlm", default=None, help="optional HF Bengali masked-LM for rescoring")
    ap.add_argument("--lam", type=float, default=0.3)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tt = load_model(a.ckpt, a.token_list, a.ctc_weight, device)
    beam = get_beam_search_decoder(model, tt.token_list,
                                   ctc_weight=a.ctc_weight, beam_size=a.beam_size)
    ds = BnVideoDataset(a.manifest, a.token_list, "test")  # CenterCrop, deterministic
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))

    rescorer = None
    if a.mlm:
        from rescorer import MLMRescorer  # copy our rescorer.py into the repo root
        rescorer = MLMRescorer(a.mlm, device=device)

    refs, hyps, hyps_rs, rows = [], [], [], []
    for i in range(n):
        item = ds[i]
        ref = tt.post_process(item["target"])
        nb = decode_one(model, beam, item["input"], device, a.nbest)
        hyp = tt.post_process(torch.tensor(nb[0][0]))
        refs.append(ref); hyps.append(hyp)

        rs = hyp
        if rescorer is not None:
            best, best_s = None, -1e30
            for ids, ac in nb:
                text = tt.post_process(torch.tensor(ids))
                s = ac + a.lam * rescorer.pll(text)
                if s > best_s:
                    best_s, best = s, text
            rs = best
        hyps_rs.append(rs)
        rows.append({"ref": ref, "hyp": hyp, "hyp_rescored": rs})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n} decoded")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n--- top-1 beam search ---")
    print("CER", round(evaluate.cer(hyps, refs), 4), "| WER", round(evaluate.wer(hyps, refs), 4))
    rates, counts = evaluate.class_error_rates(hyps, refs)
    print("phoneme-class error:", {k: round(v, 3) for k, v in rates.items()})
    if rescorer is not None:
        print("\n--- after MLM rescoring ---")
        print("CER", round(evaluate.cer(hyps_rs, refs), 4), "| WER", round(evaluate.wer(hyps_rs, refs), 4))
        r2, _ = evaluate.class_error_rates(hyps_rs, refs)
        print("phoneme-class error:", {k: round(v, 3) for k, v in r2.items()})
    print("\nsamples:")
    for r in rows[:5]:
        print(" ref:", r["ref"])
        print(" hyp:", r["hyp"])


if __name__ == "__main__":
    main()