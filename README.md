# BanglaLRS

This repository contains the code, configuration files, and metadata required to reproduce the Bengali audiovisual speech recognition pipeline used in **BanglaLRS**, while excluding large raw media files, generated outputs, and model checkpoints.

Each component has its own dependencies. Install the corresponding `requirements.txt` files and use separate virtual environments to avoid dependency conflicts.

## Folder Layout

- `banglavsr-data/`: video extraction and transcript generation
- `banglavsr-asr/`: transcript filtering, mouth ROI generation, dataset split creation, and token preparation
- `auto_avsr/`: Bengali visual speech recognition training and decoding, based on the [Auto-AVSR](https://github.com/mpc001/auto_avsr) codebase and environment (please clone the original repo for the necessary modules)

## Execution Order

### 1. Extraction and Transcription

```bash
cd banglavsr-data

bash setup.sh
source .venv-extract/bin/activate

python run.py \
  --url "https://www.youtube.com/watch?v=XXXX" \
  --out data \
  --limit 1

python transcribe.py \
  --clips /home/tigerit/Documents/banglavsr-data/data/clips \
  --out /home/tigerit/Documents/banglavsr-data/data/transcripts_bn.jsonl
```

### 2. Transcript Processing, ROI Generation, and Splits

```bash
cd banglavsr-asr

python data_prep.py \
  --transcripts /home/Documents/banglavsr-data/data/transcripts_bn.jsonl \
  --out_manifest m.jsonl \
  --out_vocab vocab.json

python build_roi.py \
  --manifest m.jsonl \
  --clips_dir /home/Documents/banglavsr-data/data/clips \
  --roi_dir data/roi \
  --mean_face ./20words_mean_face.npy \
  --out_manifest m_roi.jsonl

python split.py \
  --manifest m_roi.jsonl \
  --clips_dir /home/Documents/banglavsr-data/data/clips \
  --out_dir data/splits \
  --threshold 0.5
```

### 3. Model Training and Decoding

Set up the `auto_avsr` environment following the instructions in the upstream [Auto-AVSR repository](https://github.com/mpc001/auto_avsr), then run:

```bash
cd auto_avsr

python bn_train.py \
  --train-file data/splits/train.jsonl \
  --val-file data/splits/val.jsonl \
  --token-list bn_tokens.json \
  --pretrained-model-path vsr_trlrs3_base.pth \
  --freeze-encoder \
  --max-frames 800 \
  --precision bf16-mixed

python bn_decode.py \
  --ckpt exp_bn/last-v1.ckpt \
  --manifest data/splits/val.jsonl \
  --token-list bn_tokens.json \
  --beam-size 20 \
  --limit 20
```

## Notes

- Large media files, generated output directories, and model checkpoints are intentionally excluded from this repository.
- This repository is provided as a **reproducible code bundle**, rather than a complete archive of the BanglaLRS dataset.
- Paths shown above correspond to the original development environment and may need to be adjusted for a local installation.
