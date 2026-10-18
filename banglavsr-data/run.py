#!/usr/bin/env python3
"""Build anchor speech clips from a YouTube URL.

Single video:
    python run.py --url "https://www.youtube.com/watch?v=XXXX"

Playlist or channel (works unchanged, same flag):
    python run.py --url "https://www.youtube.com/playlist?list=YYYY"
"""
from __future__ import annotations
import argparse
from pathlib import Path

from banglavsr.utils import Config
from banglavsr.sources import enumerate_videos
from banglavsr.pipeline import process_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="video, playlist, or channel URL")
    ap.add_argument("--out", default="data", help="output root")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asd", default="heuristic", choices=["heuristic", "lightasd"])
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    ap.add_argument("--delete-raw", action="store_true",
                    help="delete the source video after processing (keeps clips, audio, captions)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    out_root = Path(args.out)

    videos = enumerate_videos(args.url)
    if args.limit:
        videos = videos[:args.limit]
    print(f"resolved {len(videos)} video(s)")

    total_clips = 0
    for v in videos:
        try:
            total_clips += process_video(v, out_root, cfg, asd_backend=args.asd,
                                          delete_raw=args.delete_raw)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[error] {v['id']}: {e}")
    print(f"finished: {total_clips} clips written to {out_root/'clips'}")
    print(f"manifest: {out_root/'manifest.jsonl'}")


if __name__ == "__main__":
    main()
