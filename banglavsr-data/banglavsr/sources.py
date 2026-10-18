"""Resolve any YouTube URL into a flat list of videos.

A single-video URL yields one entry; a playlist or channel URL yields many.
Everything downstream iterates over the returned list, so adding playlist
support later needs no changes elsewhere.
"""
from __future__ import annotations
from yt_dlp import YoutubeDL


def enumerate_videos(url: str) -> list[dict]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,     # do not resolve each entry fully, just list them
        "skip_download": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries")
    if entries:
        out = []
        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            if not vid:
                continue
            out.append({
                "id": vid,
                "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
                "title": e.get("title"),
            })
        return out

    return [{
        "id": info["id"],
        "url": info.get("webpage_url", url),
        "title": info.get("title"),
    }]
