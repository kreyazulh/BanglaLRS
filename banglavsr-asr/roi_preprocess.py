import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Mouth-ROI preprocessing must EXACTLY match the AV-HuBERT checkpoint convention,
# or transfer degrades badly. Do NOT reimplement face detection / landmarking here.
# Wire `align_fn` to one of:
#   - av_hubert/avhubert/preparation  (RetinaFace + FAN 68 landmarks -> affine
#     align to their 20-frame mean face -> crop 96x96 -> grayscale), or
#   - auto_avsr's preprocessing (cleaner PyTorch, same convention).
# This module only handles decode -> 25 fps resample -> delegate -> cache.
# Output cache: uint8 grayscale [T, 96, 96]. Train-time crop to 88 + normalize
# happens in dataset.py to match AV-HuBERT's data transform.
# ---------------------------------------------------------------------------


def decode_video_25fps(path):
    """Return RGB frames [T, H, W, 3] uint8, resampled to 25 fps. Uses OpenCV,
    which is already required by the aligner, so no torchvision/decord dependency."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # guard 0 / NaN
        fps = 25.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    frames = np.asarray(frames, dtype=np.uint8)
    idx = _resample_indices(len(frames), fps, 25.0)
    return frames[idx]


def _resample_indices(n, src_fps, dst_fps):
    if n == 0:
        return [0]
    dur = n / src_fps
    m = max(1, int(round(dur * dst_fps)))
    return [min(n - 1, int(round(k * src_fps / dst_fps))) for k in range(m)]


def preprocess_clip(video_path, align_fn, cache_path):
    """align_fn(frames_rgb [T,H,W,3] uint8) -> grayscale mouth ROI [T,96,96] uint8."""
    if os.path.exists(cache_path):
        return cache_path
    frames = decode_video_25fps(video_path)
    roi = align_fn(frames)  # AV-HuBERT / auto_avsr recipe
    roi = np.asarray(roi, dtype=np.uint8)
    assert roi.ndim == 3 and roi.shape[1:] == (96, 96), f"bad ROI shape {roi.shape}"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(torch.from_numpy(roi), cache_path)
    return cache_path