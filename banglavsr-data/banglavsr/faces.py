"""Single decode pass over a video.

Produces lightweight per-detection features (bbox, 5-pt kps, ArcFace embedding,
inner-lip aperture) so every later stage works on small arrays instead of
re-decoding frames. Lip aperture comes from insightface's own 68-point landmark
model, so there is no MediaPipe / jax dependency. Heavy deps: opencv,
insightface, scenedetect.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from .utils import Config

# iBUG 68 inner-lip landmarks: top row 61,62,63 sit above bottom row 67,66,65.
_INNER_TOP = [61, 62, 63]
_INNER_BOT = [67, 66, 65]


@dataclass
class Detection:
    frame_idx: int
    shot_id: int
    bbox: tuple              # (x1, y1, x2, y2) in pixels
    kps: np.ndarray          # (5, 2)
    det_score: float
    emb: np.ndarray          # (512,) ArcFace, L2-normalised
    aperture: float          # inner-lip gap normalised by face height


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int


def detect_shots(video_path: Path) -> list[tuple[int, int]]:
    """Return (start_frame, end_frame) per shot using content-based cuts."""
    from scenedetect import detect, ContentDetector
    scenes = detect(str(video_path), ContentDetector())
    if not scenes:
        return []
    return [(s.get_frames(), e.get_frames()) for s, e in scenes]


def _shot_id_for_frame(frame_idx: int, shots: list[tuple[int, int]]) -> int:
    for sid, (s, e) in enumerate(shots):
        if s <= frame_idx < e:
            return sid
    return -1


def _build_face_app(cfg: Config):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "landmark_3d_68", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def _lip_aperture(face, bbox) -> float:
    """Mean inner-lip vertical gap (normalised by face height) from 68 landmarks."""
    lm = getattr(face, "landmark_3d_68", None)
    if lm is None or len(lm) < 68:
        return 0.0
    lm = np.asarray(lm, dtype=np.float32)[:, :2]
    gap = float(np.mean(np.abs(lm[_INNER_TOP, 1] - lm[_INNER_BOT, 1])))
    face_h = float(bbox[3] - bbox[1]) + 1e-6
    return gap / face_h


def extract_features(video_path: Path, shots: list[tuple[int, int]],
                     cfg: Config) -> tuple[list[Detection], VideoMeta]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = VideoMeta(fps=fps, width=width, height=height, n_frames=n_frames)

    if not shots:
        shots = [(0, n_frames)]

    app = _build_face_app(cfg)

    detections: list[Detection] = []
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if cfg.frame_stride > 1 and frame_idx % cfg.frame_stride != 0:
            continue
        sid = _shot_id_for_frame(frame_idx, shots)

        for f in app.get(frame):
            if f.det_score < cfg.det_score_min:
                continue
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            detections.append(Detection(
                frame_idx=frame_idx,
                shot_id=sid,
                bbox=(x1, y1, x2, y2),
                kps=np.asarray(f.kps, dtype=np.float32),
                det_score=float(f.det_score),
                emb=np.asarray(f.normed_embedding, dtype=np.float32),
                aperture=_lip_aperture(f, (x1, y1, x2, y2)),
            ))

    cap.release()
    return detections, meta