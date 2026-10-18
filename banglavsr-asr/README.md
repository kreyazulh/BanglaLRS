# BanglaVSR pilot

Frozen AV-HuBERT encoder (+LoRA) -> grapheme CTC (+ small attention decoder)
-> n-best -> Bengali MLM rescorer. The result is the phoneme-class error gap, not WER.

## Model choice
Use the **Base LRS3-433h finetuned** checkpoint as the frozen feature extractor.
At 10-20 h it is a stronger frozen extractor than pretrained-only: its features are
already lipreading-discriminative, and visemes transfer across languages. Keep the
pretrained-only checkpoint for a sensitivity row (init effect). Optionally sweep
`feature_layer` in encoder.py: the final layer is the most English-specialized, a
mid layer can transfer better.

Base hidden size is **768**, not 1024 (the spec assumed Large). All heads are sized
from `encoder.output_dim`, so this is handled. Bonus: in Stage 2 the MLM hidden is
also 768, so the visual projection becomes near-identity.

## Run order
1. `python data_prep.py --transcripts data/transcripts_bn.jsonl --out_manifest m.jsonl --out_vocab vocab.json`
2. ROI cache: wire `roi_preprocess.align_fn` to AV-HuBERT (or auto_avsr) landmark+align,
   run over clips, add `roi_cache` and `n_frames` to each manifest row.
3. Speaker-disjoint split via ArcFace face clustering (not video_id), then split.
4. `python train.py --manifest m.jsonl --vocab vocab.json --ckpt avhubert_base_lrs3_433h.pt`
5. Decode val to n-best (decode.ctc_beam_search), tune lambda (rescorer.tune_lambda).
6. `evaluate.report` + `pair_confusion` with vs without rescorer. That table is the finding.

## What runs where
- vocab / data_prep / evaluate / decode / rescorer logic: standard, low risk.
- encoder.py: the only fairseq-tied module. Verify attribute paths and the
  `extract_features` signature once against your checkpoint (notes in the file).
- ROI must match AV-HuBERT exactly (96x96 grayscale, mean-face align, 25 fps).
  dataset.py applies the 88 crop + (0.421/0.165) normalize to match the data transform.

## Caveats for the writeup
- Classes in evaluate.py are orthographic codepoint sets, not phonemes. State this
  framing and verify against a Bengali phonetics reference.
- MLM must have a fill-mask head. BanglaBERT (csebuetnlp) is ELECTRA and will fail
  the rescorer assert by design.
- decode.ctc_beam_search is a correct reference; swap torchaudio's CTC decoder if you
  scale up. The rescorer only needs (ids, acoustic_score).
