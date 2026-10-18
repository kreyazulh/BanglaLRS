# BanglaVSR data pipeline

Extract anchor speech segments from Bengali broadcast news on YouTube, cropped
to the speaker's face and ready for the ASR / forced-alignment / sentence-cut
stage. Works on a single video now and on whole playlists or channels later
through the same command.

## What it does

From one URL it produces face-cropped clips of the news anchor actually
speaking, dropping ads, cutscenes, B-roll, wide shots, and voiceover.

```
URL
 -> download video + Bangla auto-subs (yt-dlp)
 -> 16 kHz mono audio (ffmpeg) + per-frame VAD (webrtcvad)
 -> shot boundaries (PySceneDetect)
 -> one decode pass: face detect + ArcFace embedding (insightface)
                      + lip aperture (mediapipe)
 -> IoU tracking within shots
 -> active-speaker score per track (lip motion gated by VAD)
 -> cluster tracks by face identity, keep the recurring anchor
 -> speaking-probability -> clean spans (smooth, hysteresis, merge, length cap)
 -> export face crops + audio slices + manifest.jsonl
```

The two ideas that make it elegant: active-speaker detection (not plain face
detection) is what rejects B-roll voiceover, and clustering tracks by face
identity is what isolates the anchor and drops one-off ad presenters.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

`ffmpeg` is a system dependency (`sudo apt-get install -y ffmpeg`). On a GPU box,
swap `onnxruntime` for `onnxruntime-gpu` and set the insightface providers to
`CUDAExecutionProvider` in `banglavsr/faces.py` for a large speedup.

## Run

```bash
# single video
python run.py --url "https://www.youtube.com/watch?v=XXXX" --out data --limit 1

# whole playlist or channel later, no code change
python run.py --url "https://www.youtube.com/playlist?list=YYYY" --out data
```

Output:

```
data/
  raw/        downloaded mp4 + .bn.vtt subs
  work/       audio wav, per-video .done markers (resumable)
  clips/      <video>_<track>_<frame>.mp4  +  .wav
  manifest.jsonl   one row per clip
```

Re-running skips finished videos (the `.done` markers), so collection is
resumable across sessions.

## Tuning

Tune on one hand-checked clip before scaling, as recommended:

- `asd_motion_ref` (config.yaml): the single most important knob. Lower it if
  real speech is being missed, raise it if idle faces leak through.
- `asd_high` / `asd_low`: hysteresis thresholds for starting / continuing a span.
- `min_face_frac`, `frontal_min`: stricter values keep only clean frontal faces.
- `id_sim_thresh`: cosine similarity for treating two tracks as the same person.

## Upgrading the active-speaker model

`HeuristicASD` (lip motion + VAD) is the default and runs with no extra weights.
It is good for clean anchor-desk footage, especially with the anchor-identity
filter. For higher precision on the harder cases (anchor's face shown over a
different voice), wire `LightASDAdapter` in `banglavsr/asd.py` to a cloned
Light-ASD or TalkNet checkpoint. The interface is identical, so `pipeline.py`
does not change. Select it with `--asd lightasd`.

## Notes

- Auto-subtitles are downloaded as a free prior, but their timing is approximate.
  Use proper forced alignment (CTC-segmentation or WhisperX) in the next stage,
  not the .vtt timestamps, for training labels.
- Plan to release the pipeline plus video IDs and timestamps rather than the
  videos themselves.
