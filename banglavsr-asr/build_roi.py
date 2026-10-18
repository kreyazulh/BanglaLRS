import argparse
import json
import os

import torch
import numpy as np

from data_prep import load_jsonl
from roi_preprocess import preprocess_clip


def make_align_fn(mean_face_path, device="cuda"):
    """FAN (face_alignment) for 68 landmarks + the vendored AV-HuBERT crop.
    mean_face_path points at the repo's 20words_mean_face.npy.
    Needs: pip install face-alignment scikit-image opencv-python (torch already present)."""
    import cv2
    import face_alignment

    from roi_align_avhubert import crop_patch, landmarks_interpolate

    mean_face = np.load(mean_face_path)
    assert mean_face.shape == (68, 2), f"mean face must be (68,2), got {mean_face.shape}"

    try:
        lm_type = face_alignment.LandmarksType.TWO_D       # newer API
    except AttributeError:
        lm_type = face_alignment.LandmarksType._2D          # older API
    fa = face_alignment.FaceAlignment(lm_type, device=device, flip_input=False)

    def align(frames_rgb):
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
        lms = []
        for f in frames_rgb:
            preds = fa.get_landmarks(f)
            lms.append(preds[0] if preds else None)
        lms = landmarks_interpolate(lms)
        if lms is None:
            raise RuntimeError("no face detected in any frame")
        return crop_patch(gray, lms, mean_face)  # [T,96,96] uint8

    return align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)        # output of data_prep
    ap.add_argument("--clips_dir", required=True)        # dir with {clip_id}.mp4
    ap.add_argument("--roi_dir", required=True)          # where .pt caches go
    ap.add_argument("--mean_face", required=True)        # 20words_mean_face.npy from the repo
    ap.add_argument("--out_manifest", required=True)
    a = ap.parse_args()

    rows = load_jsonl(a.manifest)
    os.makedirs(a.roi_dir, exist_ok=True)

    # resume: skip clips already recorded in the manifest
    done = set()
    if os.path.exists(a.out_manifest):
        for r in load_jsonl(a.out_manifest):
            done.add(r.get("clip_id") or r.get("clip"))
    print(f"resuming with {len(done)} clips already in {a.out_manifest}")

    align = None  # built lazily, only if a clip actually needs ROI computed
    with open(a.out_manifest, "a", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            clip = r.get("clip_id") or r["clip"]
            if clip in done:
                continue
            video_path = os.path.join(a.clips_dir, clip + ".mp4")
            cache = os.path.join(a.roi_dir, clip + ".pt")
            try:
                if not os.path.exists(cache):
                    if align is None:
                        align = make_align_fn(a.mean_face)
                    preprocess_clip(video_path, align, cache)
                n = int(torch.load(cache).shape[0])
            except Exception as e:
                print(f"skip {clip}: {e}")
                continue
            r = dict(r); r["roi_cache"] = cache; r["n_frames"] = n
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(clip)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)} seen, {len(done)} in manifest")
    print(f"manifest now has {len(done)} clips -> {a.out_manifest}")


if __name__ == "__main__":
    main()