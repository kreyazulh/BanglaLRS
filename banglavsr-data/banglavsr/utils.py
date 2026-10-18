"""Small dependency-light helpers (numpy only)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml

EPS = 1e-8


@dataclass
class Config:
    # ingest
    max_height: int = 720
    sub_lang: str = "bn"
    # detection / tracking
    frame_stride: int = 1            # process every Nth frame (1 = all)
    det_score_min: float = 0.55
    iou_thresh: float = 0.45
    track_max_age: int = 8           # frames a track may go unmatched before closing
    track_min_len: int = 8           # discard tracks shorter than this many frames
    # quality gates
    min_face_frac: float = 0.06      # min face height as fraction of frame height
    frontal_min: float = 0.55        # 0..1, higher = stricter frontal requirement
    # active speaker (heuristic)
    asd_motion_win: int = 7          # frames for rolling lip-motion energy
    asd_motion_ref: float = 0.012    # energy that maps to prob ~1 (TUNE per channel)
    vad_aggressiveness: int = 2      # webrtcvad 0..3
    # span extraction
    asd_high: float = 0.55           # hysteresis: start a span above this
    asd_low: float = 0.35            # hysteresis: keep a span until below this
    smooth_k: int = 5                # median smoothing window (frames)
    merge_gap_sec: float = 0.20      # bridge speaking spans closer than this
    min_span_sec: float = 0.40       # drop spans shorter than this
    max_span_sec: float = 6.0        # hard cap, matches LRS3 convention
    # anchor identity clustering
    id_sim_thresh: float = 0.50      # cosine sim above which two tracks are same person
    # export
    crop_margin: float = 0.4         # extra padding around face box (fraction)
    crop_size: int = 224             # output square size before later mouth-ROI step

    min_active_frac: float = 0.5   # fraction of a voiced run the anchor's lips must move
    span_pad_sec: float = 0.06     # small padding so onset/offset consonants aren't clipped

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        data = {}
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
        known = {f for f in Config.__dataclass_fields__}
        return Config(**{k: v for k, v in data.items() if k in known})


def iou(a, b) -> float:
    """IoU of two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + EPS)


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + EPS)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2norm(a), l2norm(b)))


def frontal_score(kps: np.ndarray) -> float:
    """0..1 frontal score from insightface 5-point kps.

    kps order: [left_eye, right_eye, nose, left_mouth, right_mouth], each (x, y).
    Uses horizontal nose offset from the eye midpoint, normalised by inter-ocular
    distance, as a cheap yaw proxy.
    """
    kps = np.asarray(kps, dtype=np.float32)
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid_x = 0.5 * (le[0] + re[0])
    interocular = abs(re[0] - le[0]) + EPS
    yaw = abs(nose[0] - eye_mid_x) / interocular     # 0 frontal, grows when turned
    return float(np.clip(1.0 - yaw / 0.5, 0.0, 1.0))


def face_height_frac(bbox, frame_h: int) -> float:
    return (bbox[3] - bbox[1]) / (frame_h + EPS)


def square_crop_box(bbox, margin: float, frame_w: int, frame_h: int):
    """Padded square crop centred on the face, clamped to the frame."""
    x1, y1, x2, y2 = bbox
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    nx1 = int(max(0, round(cx - half)))
    ny1 = int(max(0, round(cy - half)))
    nx2 = int(min(frame_w, round(cx + half)))
    ny2 = int(min(frame_h, round(cy + half)))
    return nx1, ny1, nx2, ny2


def median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Odd-window median filter, numpy only, edge-replicated."""
    x = np.asarray(x, dtype=np.float32)
    if k <= 1 or x.size == 0:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.median(xp[i:i + k])
    return out


def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling standard deviation, numpy only."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.std(xp[i:i + win])
    return out
