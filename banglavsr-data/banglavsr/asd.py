"""Active speaker detection.

`HeuristicASD` is the zero-extra-model default: lip-motion energy gated by audio
VAD. It runs out of the box and is good enough for clean anchor-desk footage,
especially combined with the anchor-identity filter. Its weak spot is the
exact case where the anchor's face is shown while a different voice plays
(reaction shots over a clip); forced alignment downstream catches most of those.

`LightASDAdapter` is the precision upgrade. Wire it to a cloned Light-ASD or
TalkNet repo when you want true audio-visual sync scoring. The interface is the
same, so the pipeline does not change.
"""
from __future__ import annotations
import numpy as np

from .utils import Config, rolling_std


class ActiveSpeakerDetector:
    def score_track(self, track, speech_mask: np.ndarray, cfg: Config) -> np.ndarray:
        """Return a per-frame speaking probability aligned to track.frames."""
        raise NotImplementedError


class HeuristicASD(ActiveSpeakerDetector):
    def score_track(self, track, speech_mask: np.ndarray, cfg: Config) -> np.ndarray:
        aperture = np.asarray(track.apertures, dtype=np.float32)
        energy = rolling_std(aperture, cfg.asd_motion_win)
        prob = np.clip(energy / (cfg.asd_motion_ref + 1e-8), 0.0, 1.0)
        # Gate by whether there is actually speech at each frame.
        for i, fi in enumerate(track.frames):
            if not (0 <= fi < speech_mask.size and speech_mask[fi]):
                prob[i] = 0.0
        return prob


class LightASDAdapter(ActiveSpeakerDetector):
    """Skeleton for swapping in a real AV-sync model.

    Expected wiring:
      1. git clone https://github.com/Junhua-Liao/Light-ASD into third_party/
      2. load its checkpoint once in __init__
      3. in score_track, crop the track's faces, resample to 25 fps, pair with
         the matching audio window, and return the model's per-frame scores
         mapped onto track.frames.
    Keeping the same signature means pipeline.py needs no changes.
    """
    def __init__(self, repo_dir: str, ckpt: str):
        self.repo_dir = repo_dir
        self.ckpt = ckpt
        # TODO: import the repo, build the model, load weights.

    def score_track(self, track, speech_mask: np.ndarray, cfg: Config) -> np.ndarray:
        raise NotImplementedError(
            "Wire LightASDAdapter to a cloned Light-ASD/TalkNet checkpoint, "
            "or use HeuristicASD for the pilot.")


def make_asd(name: str, cfg: Config) -> ActiveSpeakerDetector:
    if name == "heuristic":
        return HeuristicASD()
    if name == "lightasd":
        return LightASDAdapter(repo_dir="third_party/Light-ASD", ckpt="weights/finetuning_TalkSet.model")
    raise ValueError(f"unknown asd backend: {name}")
