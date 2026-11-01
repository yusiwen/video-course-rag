#!/usr/bin/env python3
"""Emit WebVTT for a run so external players can show the same subtitles.

The viewer draws its own overlay from the sentence table (no CORS, works over
file://), but mpv/IINA/QuickTime want a file. One vtt per run in index/<run>/.

Usage: scripts/subs.py <run-id> [<run-id> ...]
"""
from ..paths import run as _run
import json
import sys
from pathlib import Path


def ts(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main():
    for run in sys.argv[1:]:
        sents = json.load(open(str(_run(run).asr_sent), encoding="utf-8"))["items"]
        out = _run(run).index_dir / "subs.vtt"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["WEBVTT", ""]
        for i, s in enumerate(sents, 1):
            lines.append(str(i))
            lines.append(f"{ts(s['start'] / 1000)} --> {ts(s['end'] / 1000)}")
            lines.append(s["text"])
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"{out}  {len(sents)} cues  {out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
