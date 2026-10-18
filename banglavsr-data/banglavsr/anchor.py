"""Find the anchor: the identity that recurs across the program and speaks most.

Greedy cosine-similarity clustering over per-track mean embeddings, then pick
the cluster with the most total speaking frames. This drops one-off ad presenters
(non-recurring identities) and B-roll people.
"""
from __future__ import annotations
import numpy as np

from .utils import cosine_sim, Config


def cluster_tracks(tracks, cfg: Config) -> list[int]:
    """Return a cluster id per track (index-aligned to `tracks`)."""
    cluster_ids = [-1] * len(tracks)
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []

    for i, tr in enumerate(tracks):
        emb = tr.mean_embedding()
        best, best_sim = -1, cfg.id_sim_thresh
        for c, cen in enumerate(centroids):
            s = cosine_sim(emb, cen)
            if s >= best_sim:
                best, best_sim = c, s
        if best == -1:
            centroids.append(emb)
            members.append([i])
            cluster_ids[i] = len(centroids) - 1
        else:
            members[best].append(i)
            cluster_ids[i] = best
            # update centroid as running mean (re-normalised)
            m = members[best]
            cen = np.mean([tracks[j].mean_embedding() for j in m], axis=0)
            centroids[best] = cen / (np.linalg.norm(cen) + 1e-8)

    return cluster_ids


def select_anchor_tracks(tracks, cluster_ids, speaking_frames, cfg: Config) -> set[int]:
    """speaking_frames[i] = number of speaking frames in track i.
    Returns the set of track indices belonging to the anchor cluster."""
    totals: dict[int, int] = {}
    for i, cid in enumerate(cluster_ids):
        totals[cid] = totals.get(cid, 0) + int(speaking_frames[i])
    if not totals:
        return set()
    anchor_cluster = max(totals, key=totals.get)
    return {i for i, cid in enumerate(cluster_ids) if cid == anchor_cluster}
