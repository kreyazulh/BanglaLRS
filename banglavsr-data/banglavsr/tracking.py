"""Greedy IoU tracking. Tracks never cross shot boundaries.

Depends only on numpy so it can be unit-tested without the CV stack.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .utils import iou, l2norm, Config


@dataclass
class Track:
    track_id: int
    shot_id: int
    frames: list = field(default_factory=list)       # frame indices
    bboxes: list = field(default_factory=list)
    kps: list = field(default_factory=list)
    det_scores: list = field(default_factory=list)
    apertures: list = field(default_factory=list)
    embs: list = field(default_factory=list)

    def __len__(self):
        return len(self.frames)

    def mean_embedding(self) -> np.ndarray:
        return l2norm(np.mean(np.stack(self.embs), axis=0))


def build_tracks(detections, cfg: Config) -> list[Track]:
    """Detections must carry .frame_idx and .shot_id. Returns closed tracks
    of at least cfg.track_min_len frames."""
    by_shot: dict[int, list] = {}
    for d in detections:
        by_shot.setdefault(d.shot_id, []).append(d)

    next_id = 0
    finished: list[Track] = []

    for shot_id, dets in by_shot.items():
        if shot_id < 0:
            continue
        frames = sorted({d.frame_idx for d in dets})
        per_frame: dict[int, list] = {}
        for d in dets:
            per_frame.setdefault(d.frame_idx, []).append(d)

        active: list[Track] = []
        last_seen: dict[int, int] = {}

        for fi in frames:
            cur = per_frame[fi]
            used = set()
            # Greedy: each active track grabs its best unused detection.
            for tr in active:
                best, best_iou = None, cfg.iou_thresh
                for j, d in enumerate(cur):
                    if j in used:
                        continue
                    v = iou(tr.bboxes[-1], d.bbox)
                    if v >= best_iou:
                        best, best_iou = j, v
                if best is not None:
                    d = cur[best]
                    used.add(best)
                    tr.frames.append(d.frame_idx)
                    tr.bboxes.append(d.bbox)
                    tr.kps.append(d.kps)
                    tr.det_scores.append(d.det_score)
                    tr.apertures.append(d.aperture)
                    tr.embs.append(d.emb)
                    last_seen[tr.track_id] = fi

            for j, d in enumerate(cur):
                if j in used:
                    continue
                tr = Track(track_id=next_id, shot_id=shot_id)
                next_id += 1
                tr.frames.append(d.frame_idx)
                tr.bboxes.append(d.bbox)
                tr.kps.append(d.kps)
                tr.det_scores.append(d.det_score)
                tr.apertures.append(d.aperture)
                tr.embs.append(d.emb)
                last_seen[tr.track_id] = fi
                active.append(tr)

            still: list[Track] = []
            for tr in active:
                if fi - last_seen[tr.track_id] > cfg.track_max_age:
                    if len(tr) >= cfg.track_min_len:
                        finished.append(tr)
                else:
                    still.append(tr)
            active = still

        for tr in active:
            if len(tr) >= cfg.track_min_len:
                finished.append(tr)

    return finished
