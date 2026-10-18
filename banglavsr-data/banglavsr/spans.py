"""Convert speaking activity into clean spans whose boundaries sit in audio pauses.

Boundaries come from the VAD speech mask, not lip motion, so clips do not cut
mid-word on closed-mouth consonants. Lip motion is used only to confirm the
anchor is the speaker over each candidate run. Numpy only.
"""
from __future__ import annotations
import numpy as np

from .utils import median_smooth, frontal_score, face_height_frac, Config


def frame_validity(track, frame_h: int, cfg: Config) -> np.ndarray:
    n = len(track.frames)
    valid = np.ones(n, dtype=bool)
    for i in range(n):
        ok = track.det_scores[i] >= cfg.det_score_min
        ok &= face_height_frac(track.bboxes[i], frame_h) >= cfg.min_face_frac
        ok &= frontal_score(track.kps[i]) >= cfg.frontal_min
        valid[i] = ok
    return valid


def _bridge(voiced: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill internal silence gaps shorter than max_gap so they do not split a span."""
    v = voiced.copy()
    n = len(v); i = 0
    while i < n:
        if not v[i]:
            j = i
            while j < n and not v[j]:
                j += 1
            if 0 < i and j < n and (j - i) < max_gap:
                v[i:j] = True
            i = j
        else:
            i += 1
    return v


def _contig_true(v: np.ndarray) -> list[tuple[int, int]]:
    runs = []; n = len(v); i = 0
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def _split_long(s: int, e: int, voiced: np.ndarray, max_len: int, min_len: int):
    """Split a too-long run at its longest internal pause; recurse."""
    if e - s <= max_len:
        return [(s, e)]
    best_mid, best_len = None, 0
    i = s
    while i < e:
        if not voiced[i]:
            j = i
            while j < e and not voiced[j]:
                j += 1
            if i > s and j < e and (j - i) > best_len:
                best_len, best_mid = j - i, (i + j) // 2
            i = j
        else:
            i += 1
    cut = best_mid if best_mid is not None else s + max_len
    out = []
    for a, b in ((s, cut), (cut, e)):
        if b - a >= min_len:
            out.extend(_split_long(a, b, voiced, max_len, min_len))
    return out


def build_spans(track, speaking_prob: np.ndarray, speech_mask: np.ndarray,
                fps: float, frame_h: int, cfg: Config) -> list[tuple[int, int]]:
    n = len(track.frames)
    valid = frame_validity(track, frame_h, cfg)
    prob = np.asarray(speaking_prob, dtype=np.float32).copy()
    prob[~valid] = 0.0
    prob = median_smooth(prob, cfg.smooth_k)
    active = (prob >= cfg.asd_low) & valid          # frames where the anchor is talking

    voiced = np.array(
        [bool(speech_mask[fi]) if 0 <= fi < speech_mask.size else False
         for fi in track.frames], dtype=bool)

    max_gap = int(round(cfg.merge_gap_sec * fps))   # silence shorter than this is not a boundary
    bridged = _bridge(voiced, max_gap)

    min_len = int(round(cfg.min_span_sec * fps))
    max_len = int(round(cfg.max_span_sec * fps))
    pad = int(round(cfg.span_pad_sec * fps))
    lo, hi = track.frames[0], track.frames[-1] + 1

    out = []
    for s, e in _contig_true(bridged):
        if active[s:e].mean() < cfg.min_active_frac:   # anchor not the speaker (voiceover)
            continue
        for a, b in _split_long(s, e, voiced, max_len, min_len):
            if b - a < min_len:
                continue
            fa = max(lo, track.frames[a] - pad)
            fb = min(hi, track.frames[b - 1] + 1 + pad)
            out.append((fa, fb))
    return out