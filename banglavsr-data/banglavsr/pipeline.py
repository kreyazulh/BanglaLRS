"""End-to-end processing of a single video.

download -> audio + VAD -> shots -> detect/embed/aperture -> track ->
ASD score -> anchor cluster -> spans -> export. Resumable: a video with a
`.done` marker is skipped.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .utils import Config
from . import media, faces, tracking, asd as asd_mod, anchor, spans, export


def process_video(video: dict, out_root: Path, cfg: Config,
                  asd_backend: str = "heuristic", max_seconds: int = 0,
                  delete_raw: bool = False) -> int:
    vid = video["id"]
    raw_dir = out_root / "raw"
    clip_dir = out_root / "clips"
    work_dir = out_root / "work"
    manifest = out_root / "manifest.jsonl"
    done_marker = work_dir / f"{vid}.done"
    if done_marker.exists():
        print(f"[skip] {vid} already processed")
        return 0
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] {vid}")
    video_path = media.download_video(video["url"], vid, raw_dir, cfg)

    print(f"[audio] {vid}")
    wav_path = media.extract_audio(video_path, work_dir / f"{vid}.wav")

    print(f"[shots] {vid}")
    shots = faces.detect_shots(video_path)

    print(f"[faces] {vid} (single decode pass)")
    detections, meta = faces.extract_features(video_path, shots, cfg)
    speech_mask = media.speech_mask_per_frame(
        wav_path, meta.fps, meta.n_frames, cfg.vad_aggressiveness)

    print(f"[track] {vid}: {len(detections)} detections")
    tracks = tracking.build_tracks(detections, cfg)
    if not tracks:
        print(f"[empty] {vid}: no tracks")
        done_marker.touch()
        return 0

    detector = asd_mod.make_asd(asd_backend, cfg)
    track_probs, speaking_frames = [], []
    for tr in tracks:
        p = detector.score_track(tr, speech_mask, cfg)
        track_probs.append(p)
        speaking_frames.append(int(np.sum(p >= cfg.asd_high)))

    cluster_ids = anchor.cluster_tracks(tracks, cfg)
    anchor_idx = anchor.select_anchor_tracks(tracks, cluster_ids, speaking_frames, cfg)
    print(f"[anchor] {vid}: {len(anchor_idx)}/{len(tracks)} tracks are the anchor")

    rows = []
    for i in anchor_idx:
        tr = tracks[i]
        for sf, ef in spans.build_spans(tr, track_probs[i], speech_mask, meta.fps, meta.height, cfg):
            rows.append(export.export_span(
                video_path, wav_path, tr, sf, ef, meta, clip_dir, vid, cfg))

    export.append_manifest(manifest, rows)
    done_marker.touch()

    if delete_raw:
        for f in raw_dir.glob(f"{vid}.*"):
            if f.suffix.lower() != ".vtt":      # keep the .bn.vtt captions
                f.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)       
    total = sum(r["end_sec"] - r["start_sec"] for r in rows)
    print(f"[done] {vid}: {len(rows)} clips, {total/60:.1f} min kept")
    return len(rows)
