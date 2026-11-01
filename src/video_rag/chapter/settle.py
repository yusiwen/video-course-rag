#!/usr/bin/env python3
"""Find a readable jump target for each unit.

A unit's `start` is the honest boundary (first frame after the content
changed), but that frame is often mid-transition or a slide whose bullets have
not been animated in yet. `jump` is the first second inside the unit where OCR
recovers the same title we recorded at the unit's midpoint -- i.e. a frame a
human would recognize as "this is the slide I clicked".
"""
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
        return json.loads(sh(["./scripts/ocr", png, "0.25"]).stdout or "[]")
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
        target = st["title"]
        best = st["start"]
        found = None
        for k in range(0, 9):
            t = st["start"] + k
            if t >= st["end"]:
                break
            lines = ocr(grab(video, t, tmp, region, (SW, SH)))
            if not lines:
                continue
            # match against any line, and against the joined top-5 block: the
            # "largest font" line at t often differs from the one at the
            # unit midpoint, which made single-line matching report no-match
            texts = [l["text"].strip() for l in lines if l["text"].strip()]
            block = " ".join(t2 for t2 in sorted(
                texts, key=lambda x: -len(x))[:5])
            s = max([sim(x, target) for x in texts] + [sim(block, target)]) if target else 0
            if s >= 0.6:
                found = (t, texts[0], s)
                break
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
