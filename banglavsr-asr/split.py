import argparse
import os
import random

import numpy as np

from data_prep import load_jsonl, write_jsonl
from roi_preprocess import decode_video_25fps


def embed_clips(rows, clips_dir, sample=8, device_id=0):
    """One mean ArcFace embedding per clip (sampled frames). Clips with no detected
    face get a NaN row so they can be isolated as singleton speakers later."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=device_id, det_size=(224, 224))

    embs = []
    for r in rows:
        clip = r.get("clip_id") or r["clip"]
        frames = decode_video_25fps(os.path.join(clips_dir, clip + ".mp4"))
        idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
        vecs = []
        for k in idx:
            faces = app.get(frames[k][:, :, ::-1])  # RGB -> BGR
            if faces:
                vecs.append(faces[0].normed_embedding)
        embs.append(np.mean(vecs, axis=0) if vecs else np.full(512, np.nan, np.float32))
    return np.stack(embs)


def cluster(embs, threshold=0.5):
    """Cosine agglomerative clustering, unknown speaker count. No-face clips
    (NaN rows) each become their own singleton speaker."""
    from sklearn.cluster import AgglomerativeClustering
    good = ~np.isnan(embs).any(axis=1)
    ids = np.full(len(embs), -1, dtype=int)
    if good.sum() >= 2:
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                     linkage="average", distance_threshold=threshold)
        ids[good] = cl.fit_predict(embs[good])
    nxt = (ids[good].max() + 1) if good.any() else 0
    for i in np.where(~good)[0]:  # isolate no-face clips
        ids[i] = nxt; nxt += 1
    return ids


def speaker_split(rows, speaker_ids, ratios=(0.8, 0.1, 0.1), seed=0):
    """Assign whole speakers to one split, balanced by total duration."""
    rng = random.Random(seed)
    by_spk = {}
    for r, s in zip(rows, speaker_ids):
        by_spk.setdefault(int(s), []).append(r)
    spk = list(by_spk); rng.shuffle(spk)
    total = sum(rr.get("duration", 0.0) for rs in by_spk.values() for rr in rs)
    t_train, t_val = ratios[0] * total, (ratios[0] + ratios[1]) * total
    splits = {"train": [], "val": [], "test": []}
    cum = 0.0
    for s in spk:
        bucket = "train" if cum < t_train else ("val" if cum < t_val else "test")
        for rr in by_spk[s]:
            rr = dict(rr); rr["speaker_id"] = int(s)
            splits[bucket].append(rr)
        cum += sum(rr.get("duration", 0.0) for rr in by_spk[s])
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)   # output of build_roi (has roi_cache)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)  # cosine distance; tune
    a = ap.parse_args()

    rows = load_jsonl(a.manifest)
    sid = cluster(embed_clips(rows, a.clips_dir), a.threshold)
    print(f"{len(set(sid))} speaker clusters over {len(rows)} clips")

    splits = speaker_split(rows, sid)
    seen = {}
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rs in splits.items():
        write_jsonl(rs, os.path.join(a.out_dir, name + ".jsonl"))
        spks = {r["speaker_id"] for r in rs}
        seen[name] = spks
        print(f"{name}: {len(rs)} clips, {len(spks)} speakers")
    assert not (seen["train"] & seen["val"]) and not (seen["train"] & seen["test"]) \
        and not (seen["val"] & seen["test"]), "speaker leakage across splits"
    print("speaker-disjoint check passed")


if __name__ == "__main__":
    main()