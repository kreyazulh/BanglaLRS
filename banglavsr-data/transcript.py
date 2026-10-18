#!/usr/bin/env python3
"""Transcribe clip audio to Bengali text with a Bengali-specialized Whisper model.

Uses bengaliAI/tugstugi_bengaliai-asr_whisper-medium, which an independent 2026
benchmark found to be the strongest Bengali ASR backbone. This fixes the
orthography errors that stock multilingual Whisper produces on Bengali.

Reads every .wav in the clips folder, writes one JSONL row per clip with the
transcript and a confidence (mean per-token probability, 0..1). Resumable.

Install:
    pip install transformers soundfile
(torch is assumed already present. Clips must be 16 kHz mono, which the data
pipeline already produces.)

Usage:
    python transcribe_bn.py --clips data/clips --out data/transcripts.jsonl
"""
from __future__ import annotations
import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

MODEL_ID = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"


def load_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["clip_id"])
                except Exception:
                    pass
    return done


def read_wav_16k(wav_path: Path):
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)          # to mono
    return audio, sr


@torch.no_grad()
def transcribe_one(processor, model, device, dtype, wav_path: Path, lang: str):
    audio, sr = read_wav_16k(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path.name} is {sr} Hz, expected 16000")
    feats = processor(audio, sampling_rate=16000,
                      return_tensors="pt").input_features.to(device, dtype)
    out = model.generate(
        feats,
        language=lang, task="transcribe",
        return_dict_in_generate=True, output_scores=True,
        max_new_tokens=200,
    )
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
    text = unicodedata.normalize("NFC", text).strip()
    # confidence: mean of the top softmax probability across generated steps
    if out.scores:
        probs = torch.stack(out.scores, dim=1).softmax(-1).max(-1).values
        conf = float(probs.mean())
    else:
        conf = float("nan")
    return text, conf, float(len(audio) / 16000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="data/transcripts.jsonl")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lang", default="bn", help="if generate rejects 'bn', use 'bengali'")
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(clips_dir.glob("*.wav"))
    if not wavs:
        print(f"no .wav files found in {clips_dir}")
        return

    done = load_done(out_path)
    todo = [w for w in wavs if w.stem not in done]
    print(f"{len(wavs)} clips, {len(done)} already done, {len(todo)} to transcribe")
    if not todo:
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {args.model} on {device}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype).to(device).eval()
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained("openai/whisper-medium")

    n_empty = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for wav in tqdm(todo, desc="transcribing"):
            clip_id = wav.stem
            mp4 = wav.with_suffix(".mp4")
            try:
                text, conf, dur = transcribe_one(processor, model, device, dtype, wav, args.lang)
            except Exception as e:
                print(f"[error] {clip_id}: {e}")
                continue
            if not text:
                n_empty += 1
            row = {
                "clip_id": clip_id,
                "wav_path": str(wav),
                "mp4_path": str(mp4) if mp4.exists() else None,
                "text": text,
                "confidence": round(conf, 4),
                "duration": round(dur, 3),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"done -> {out_path}  (empty transcripts: {n_empty})")


if __name__ == "__main__":
    main()