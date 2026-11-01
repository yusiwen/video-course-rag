"""The one thing a run needs to begin: a video file and its identity.

Identity is content, not the file name: a package is matched to a video by size
plus a hash of its first and last 4 MB, so renaming or moving the .mp4 changes
nothing. The page hashes the same way, so the two cannot disagree.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HEAD_TAIL = 4 * 1024 * 1024


def fingerprint(path: Path) -> dict:
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(HEAD_TAIL))
        if size > HEAD_TAIL:
            f.seek(max(0, size - HEAD_TAIL))
            h.update(f.read(HEAD_TAIL))
    return {"size": size, "sha1": h.hexdigest()[:16]}


def duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def describe(video: str | Path) -> dict:
    p = Path(video).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"no such video: {p}")
    return {"video": str(p), "name": p.name, "duration": duration_s(p), **fingerprint(p)}


def extract_audio(run) -> None:
    src = run.read_json(run.source) or {}
    video = src.get("video")
    if not video:
        raise SystemExit(f"{run.source} has no video path")
    run.audio.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(run.audio)], check=True)
    print(f"audio -> {run.audio}")
