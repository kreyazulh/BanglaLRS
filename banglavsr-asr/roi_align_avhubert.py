"""Faithful reimplementation of AV-HuBERT's mouth-ROI crop (avhubert/preparation/
align_mouth.py). Keeping the exact constants is what makes the ROIs match the
checkpoint: STD_SIZE 256, stable points [33,36,39,42,45], mouth landmarks 48:68,
window_margin 12, 96x96 grayscale crop. The only external artifact you need is the
repo's 20words_mean_face.npy. Verify by saving a few output frames as PNG and
checking they are centered mouths."""

import numpy as np
from skimage import transform as tf

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX, STOP_IDX = 48, 68
WINDOW_MARGIN = 12
CROP = 96  # final patch is CROP x CROP


def linear_interpolate(landmarks, start_idx, stop_idx):
    start = landmarks[start_idx]
    stop = landmarks[stop_idx]
    delta = stop - start
    for i in range(1, stop_idx - start_idx):
        landmarks[start_idx + i] = start + i / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    """landmarks: list of (68,2) arrays or None per frame. Fills gaps and pads ends."""
    valid = [i for i, l in enumerate(landmarks) if l is not None]
    if not valid:
        return None
    for i in range(1, len(valid)):
        if valid[i] - valid[i - 1] > 1:
            landmarks = linear_interpolate(landmarks, valid[i - 1], valid[i])
    valid = [i for i, l in enumerate(landmarks) if l is not None]
    landmarks[:valid[0]] = [landmarks[valid[0]]] * valid[0]
    landmarks[valid[-1] + 1:] = [landmarks[valid[-1]]] * (len(landmarks) - valid[-1] - 1)
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform("similarity", src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    return (warped * 255).astype("uint8"), tform


def cut_patch(img, landmarks, height, width, threshold=5):
    cy, cx = np.mean(landmarks, axis=0)[1], np.mean(landmarks, axis=0)[0]
    if cy - height < 0:
        cy = height
        if cy - height < -threshold:
            raise ValueError("too much bias in height")
    if cx - width < 0:
        cx = width
        if cx - width < -threshold:
            raise ValueError("too much bias in width")
    if cy + height > img.shape[0]:
        cy = img.shape[0] - height
        if cy + height > img.shape[0] + threshold:
            raise ValueError("too much bias in height")
    if cx + width > img.shape[1]:
        cx = img.shape[1] - width
        if cx + width > img.shape[1] + threshold:
            raise ValueError("too much bias in width")
    return np.copy(img[int(round(cy) - round(height)): int(round(cy) + round(height)),
                       int(round(cx) - round(width)): int(round(cx) + round(width))])


def crop_patch(gray_frames, landmarks, mean_face,
               stable_ids=STABLE_PNTS_IDS, std_size=STD_SIZE,
               window_margin=WINDOW_MARGIN, start_idx=START_IDX, stop_idx=STOP_IDX,
               crop=CROP):
    """gray_frames: list/array of 2D grayscale frames. landmarks: list of (68,2).
    Returns [T, crop, crop] uint8."""
    seq = []
    n = len(landmarks)
    for i in range(n):
        half = min(window_margin // 2, i, n - 1 - i)
        smoothed = np.mean(np.stack([landmarks[j] for j in range(i - half, i + half + 1)]), axis=0)
        warped, trans = warp_img(smoothed[stable_ids, :], mean_face[stable_ids, :],
                                 gray_frames[i], std_size)
        trans_lm = trans(smoothed)
        patch = cut_patch(warped, trans_lm[start_idx:stop_idx], crop // 2, crop // 2)
        seq.append(patch)
    return np.asarray(seq, dtype=np.uint8)