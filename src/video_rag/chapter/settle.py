#!/usr/bin/env python3
"""Find a readable jump target for each unit.

A unit's `start` is the honest boundary (first frame after the content
changed), but that frame is often mid-transition or a slide whose bullets have
not been animated in yet. `jump` is the first second inside the unit where OCR
recovers the same title we recorded at the unit's midpoint -- i.e. a frame a
human would recognize as "this is the slide I clicked".
"""
from ..ocr import backend
from ..paths import run as _run
import difflib
import json
import os
import re
import subprocess
import sys

REGIONS = {
    "ppt":     (0.132, 0.090, 0.711, 0.745),
    "ppt-fs":  (0.000, 0.000, 1.000, 0.930),
    "browser": (0.000, 0.080, 1.000, 0.840),
}


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def grab(video, t, out, region, size):
    vf = ["null"]
    if region:
        SW, SH = size
        x, y, w, h = region
        vf = [f"crop={int(SW*w)//2*2}:{int(SH*h)//2*2}:{int(SW*x)//2*2}:{int(SH*y)//2*2}"]
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
        "-vf", ",".join(vf), out])
    return out if os.path.exists(out) and os.path.getsize(out) > 0 else None


def ocr(png):
    try:
        return backend.read(png)
    except Exception:
        return []


def title_of(lines):
    cand = [l for l in lines if l["text"].strip()]
    if not cand:
        return ""
    hmax = max(l["h"] for l in cand)
    big = sorted([l for l in cand if l["h"] >= hmax * 0.75], key=lambda l: (-l["y"], -l["w"]))
    return big[0]["text"].strip(" ·|—-")


def sim(a, b):
    return difflib.SequenceMatcher(None, re.sub(r"\W", "", a), re.sub(r"\W", "", b)).ratio()


MATCH = 0.6      # similarity at which a frame counts as "the same slide"
TRIES = 9        # seconds after the boundary that may be scanned


def scan_for_title(start, end, target, read_at, step=1, tries=TRIES, match=MATCH):
    """First frame at or after `start` whose text recovers `target`.

    `jump` exists to answer "which second would a human click and recognise",
    so it only means something for a unit we named. An unnamed unit has no
    frame to recover: the old loop still scanned nine seconds per unit for it,
    and on a lecture where 246 of 249 units have no title that was 2214 OCR
    calls whose similarity was zero by construction — 15 of the stage's 17
    minutes, producing exactly the answer the early return gives for free.

    `read_at(t)` returns the non-empty texts read at second `t`, so a caller
    can count frames without ffmpeg or a model.
    """
    target = (target or "").strip()
    if not target:
        return None
    for k in range(tries):
        t = start + k * step
        if t >= end:
            break
        texts = read_at(t)
        if not texts:
            continue
        # match against any line, and against the joined top-5 block: the
        # "largest font" line at t often differs from the one at the unit
        # midpoint, which made single-line matching report no-match
        block = " ".join(sorted(texts, key=lambda x: -len(x))[:5])
        s = max([sim(x, target) for x in texts] + [sim(block, target)])
        if s >= match:
            return t, texts[0], s
    return None


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out/states"
    d = json.load(open(f"{outdir}/states.json"))
    video = d["video"]
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", video])
    SW, SH = [int(x) for x in r.stdout.strip().split(",")]
    tmp = f"{outdir}/_settle.png"

    for st in d["states"]:
        region = REGIONS.get(st["mode"], REGIONS["ppt-fs"])

        def read_at(t, region=region):
            lines = ocr(grab(video, t, tmp, region, (SW, SH)))
            return [l["text"].strip() for l in lines if l["text"].strip()]

        found = scan_for_title(st["start"], st["end"], st["title"], read_at)
        if found:
            st["jump"] = round(found[0], 2)
            st["jump_lag"] = round(found[0] - st["start"], 2)
            st["jump_title_match"] = round(found[2], 2)
        else:
            st["jump"] = round(st["start"], 2)
            st["jump_lag"] = 0.0
            st["jump_title_match"] = None

    if os.path.exists(tmp):
        os.remove(tmp)
    json.dump(d, open(f"{outdir}/states.json", "w"), ensure_ascii=False, indent=1)

    lagged = [s for s in d["states"] if s["jump_lag"] >= 1]
    print(f"{len(d['states'])} units; {len(lagged)} needed a jump lag >=1s")
    for s in d["states"]:
        mark = "*" if s["jump_lag"] >= 1 else " "
        print(f" {mark}{s['id']:>2} start {s['start']:7.2f}  jump {s['jump']:7.2f} "
              f"(+{s['jump_lag']:.0f}s, match {s['jump_title_match']})  {s['title'][:34]}")


if __name__ == "__main__":
    main()
