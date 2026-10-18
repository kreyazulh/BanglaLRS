"""Write the final per-span deliverables: a face-cropped mp4, its audio slice,
and a manifest row. The manifest is the handoff to the ASR/alignment stage.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2

from .utils import square_crop_box, Config


def _stable_box(track, sf: int, ef: int, frame_w: int, frame_h: int, cfg: Config):
    """One fixed square crop box for the whole span: median centre, max side,
    padded. A stable box avoids per-frame jitter, which matters for lip reading."""
    boxes = [b for f, b in zip(track.frames, track.bboxes) if sf <= f < ef]
    if not boxes:
        boxes = track.bboxes
    arr = np.asarray(boxes, dtype=np.float32)
    cx = np.median(0.5 * (arr[:, 0] + arr[:, 2]))
    cy = np.median(0.5 * (arr[:, 1] + arr[:, 3]))
    side = float(np.max(np.maximum(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])))
    pseudo = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return square_crop_box(pseudo, cfg.crop_margin, frame_w, frame_h)


def export_span(video_path: Path, wav_path: Path, track, sf: int, ef: int,
                meta, out_dir: Path, video_id: str, cfg: Config) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = f"{video_id}_{track.track_id:04d}_{sf:07d}"
    clip_path = out_dir / f"{clip_id}.mp4"
    audio_path = out_dir / f"{clip_id}.wav"

    x1, y1, x2, y2 = _stable_box(track, sf, ef, meta.width, meta.height, cfg)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, meta.fps,
                             (cfg.crop_size, cfg.crop_size))
    for _ in range(ef - sf):
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        writer.write(cv2.resize(crop, (cfg.crop_size, cfg.crop_size)))
    writer.release()
    cap.release()

    t0, t1 = sf / meta.fps, ef / meta.fps
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ss", f"{t0:.3f}",
         "-to", f"{t1:.3f}", "-c", "copy", str(audio_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return {
        "clip_id": clip_id,
        "video_id": video_id,
        "clip_path": str(clip_path),
        "audio_path": str(audio_path),
        "track_id": int(track.track_id),
        "start_frame": int(sf),
        "end_frame": int(ef),
        "start_sec": round(t0, 3),
        "end_sec": round(t1, 3),
        "fps": round(float(meta.fps), 3),
        "crop_box": [int(x1), int(y1), int(x2), int(y2)],
    }


def append_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
