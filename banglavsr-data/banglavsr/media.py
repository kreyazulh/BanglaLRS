"""Download and audio utilities (heavy deps: yt-dlp, ffmpeg, webrtcvad)."""
from __future__ import annotations
import subprocess
import wave
from pathlib import Path
import numpy as np
from yt_dlp import YoutubeDL

from .utils import Config


def download_video(url: str, video_id: str, out_dir: Path, cfg: Config) -> Path:
    """Download merged mp4 plus Bangla auto-subtitles. Idempotent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{video_id}.mp4"
    if mp4.exists():
        return mp4

    opts = {
        "format": (f"bestvideo[height<=?{cfg.max_height}][vcodec^=avc1]+bestaudio/"
           f"bestvideo[height<=?{cfg.max_height}]+bestaudio/best"),
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "writeautomaticsub": True,
        "subtitleslangs": [cfg.sub_lang],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    return mp4


def extract_audio(video_path: Path, out_wav: Path) -> Path:
    """16 kHz mono PCM wav, the format webrtcvad and sync models expect."""
    if out_wav.exists():
        return out_wav
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-vn", "-f", "wav", str(out_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav


def _read_wav_mono16(wav_path: Path):
    with wave.open(str(wav_path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected 16-bit mono"
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16), sr


def speech_mask_per_frame(wav_path: Path, fps: float, n_frames: int,
                          aggressiveness: int = 2) -> np.ndarray:
    """Boolean speech mask aligned to video frames.

    Runs webrtcvad on 30 ms audio windows, then marks a video frame as speech
    if the audio window covering its midpoint is voiced.
    """
    import webrtcvad
    pcm, sr = _read_wav_mono16(wav_path)
    vad = webrtcvad.Vad(aggressiveness)

    win_ms = 30
    win = int(sr * win_ms / 1000)            # samples per VAD window
    n_win = len(pcm) // win
    voiced = np.zeros(n_win, dtype=bool)
    for i in range(n_win):
        chunk = pcm[i * win:(i + 1) * win].tobytes()
        try:
            voiced[i] = vad.is_speech(chunk, sr)
        except Exception:
            voiced[i] = False

    mask = np.zeros(n_frames, dtype=bool)
    for f in range(n_frames):
        t_mid = (f + 0.5) / fps
        wi = int(t_mid * 1000 / win_ms)
        if 0 <= wi < n_win:
            mask[f] = voiced[wi]
    return mask
