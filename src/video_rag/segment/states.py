#!/usr/bin/env python3
"""P1 stage 1 (v4, final for this sample): content units for chaptering.

Key finding that drove this rewrite: the recording's screen content updates
roughly once per second (the capture is not 30fps of real motion), so a
"first frame that differs" segmentation produces hundreds of 1s fragments and
any minimum-duration merge then blends unrelated screens together.

So boundaries are picked as *novelty peaks with a refractory period*:
  1. per-frame change fraction (adjacent frames, taskbar masked)
  2. candidates = frames whose change >= 30% of the frame
  3. non-maximum suppression: keep the strongest candidate, then reject any
     candidate closer than MIN_GAP seconds to an accepted one
  4. units = spans between accepted boundaries; OCR each; merge neighbours
     whose OCR title is identical
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

W, H = 320, 180
FPS = 2.0
PIX_THR = 20
CHANGE_TH = 0.30     # a page/slide-level change
MIN_GAP = 3.0        # refractory: small enough not to swallow a real slash PPT<->browser switch;
                     # coarse chapter merging is the next stage's job
BOTTOM_MASK = 0.08

PPT_TABS = ("文件", "开始", "插入", "设计", "切换", "动画", "幻灯片放映",
            "审阅", "视图", "帮助", "操作说明搜索")
UI_WORDS = set(PPT_TABS) | {"Panseft", "Pansoft", "备注", "批注", "单击此处添加备注"}
URL_RE = re.compile(r"(https?://|www\.)", re.I)
HOST_RE = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.(cn|com|net|org|io)\b", re.I)

REGIONS = {
    "ppt":     (0.132, 0.090, 0.711, 0.745),
    "ppt-fs":  (0.000, 0.000, 1.000, 0.930),
    "browser": (0.000, 0.080, 1.000, 0.840),
}
TOP = (0.0, 0.0, 1.0, 0.115)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration", "-of", "csv=p=0", path])
    nums = [x for x in r.stdout.replace("\n", ",").split(",") if x.strip()]
    return float(nums[-1]), int(nums[0]), int(nums[1])


def load(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS},scale={W}:{H}",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    fr = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(fr) // (W * H)
    fr = fr[: n * W * H].reshape(n, H, W).astype(np.int16).copy()
    fr[:, int(H * (1 - BOTTOM_MASK)):, :] = 0
    return fr


def boundaries(fr, dur):
    n = len(fr)
    ch = np.array([float((np.abs(fr[i] - fr[i - 1]) > PIX_THR).mean()) for i in range(1, n)])
    cand = np.where(ch >= CHANGE_TH)[0] + 1
    order = cand[np.argsort(-ch[cand - 1])]
    accepted = [0]
    for c in order:
        if all(abs(c - a) / FPS >= MIN_GAP for a in accepted):
            accepted.append(int(c))
    accepted.sort()
    times = [a / FPS for a in accepted if a / FPS < dur] + [dur]
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)], ch


def grab(video, t, out, region, size):
    vf = ["null"]
    if region:
        SW, SH = size
        x, y, w, h = region
        vf = [f"crop={int(SW*w)//2*2}:{int(SH*h)//2*2}:{int(SW*x)//2*2}:{int(SH*y)//2*2}"]
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
        "-vf", ",".join(vf), out])
    return out if os.path.exists(out) and os.path.getsize(out) > 0 else None


def ocr(png):
    if not png:
        return []
    try:
        return json.loads(sh(["./scripts/ocr", png, "0.25"]).stdout or "[]")
    except Exception:
        return []


def brightness(video, t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                        "-vf", "scale=8:8", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    return float(np.frombuffer(r.stdout, dtype=np.uint8).mean()) if r.stdout else 0.0


def mode_of(video, t, out, size):
    top = " ".join(l["text"] for l in ocr(grab(video, t, out + "_top.png", TOP, size)))
    if any(tab in top for tab in PPT_TABS) or "PowerPoint" in top or "pptx" in top.lower():
        return "ppt", top
    if URL_RE.search(top) or HOST_RE.search(top):
        return "browser", top
    if brightness(video, t) > 150:
        if len(ocr(grab(video, t, out + "_fs.png", REGIONS["ppt-fs"], size))) >= 5:
            return "ppt-fs", top
    return ("other-app" if top.strip() else "desktop"), top


def pick_title(lines):
    cand = []
    for l in lines:
        txt = l["text"].strip()
        if not txt or txt in UI_WORDS or re.fullmatch(r"[\W_·|—-]+", txt):
            continue
        if l["x"] > 0.78 and l["y"] > 0.82:      # logo corner
            continue
        cand.append(l)
    if not cand:
        return ""
    hmax = max(l["h"] for l in cand)
    big = sorted([l for l in cand if l["h"] >= hmax * 0.75], key=lambda l: (-l["y"], -l["w"]))
    return big[0]["text"].strip(" ·|—-")


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "lecture.mp4"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "out/states"
    os.makedirs(outdir + "/thumbs", exist_ok=True)

    dur, SW, SH = probe(video)
    fr = load(video)
    units, ch = boundaries(fr, dur)
    print(f"{len(fr)} frames @{FPS}fps -> {len(units)} units "
          f"(change>={CHANGE_TH:.0%}, min gap {MIN_GAP:.0f}s)")

    states = []
    for idx, (t0, t1) in enumerate(units):
        t = (t0 + t1) / 2
        base = f"{outdir}/thumbs/{idx:03d}"
        mode, _ = mode_of(video, t, base, (SW, SH))
        full = grab(video, t, base + "_full.png", None, (SW, SH))
        region = REGIONS.get(mode)
        clines = ocr(grab(video, t, base + "_c.png", region, (SW, SH))) if region else ocr(full)
        b = int(round(t0 * FPS))
        states.append({
            "id": idx, "start": round(t0, 2), "end": round(t1, 2), "dur": round(t1 - t0, 2),
            "mode": mode,
            "title": pick_title(clines),
            "change": round(float(ch[b - 1]), 3) if 0 < b - 1 < len(ch) else None,
            "ocr_lines": [l["text"] for l in sorted(clines, key=lambda l: -l["y"])][:16],
            "thumb": f"thumbs/{idx:03d}_full.png",
        })

    merged = []
    for st in states:
        if merged and st["title"] and st["title"] == merged[-1]["title"] and st["mode"] == merged[-1]["mode"]:
            merged[-1]["end"] = st["end"]
            merged[-1]["dur"] = round(merged[-1]["end"] - merged[-1]["start"], 2)
            continue
        merged.append(st)
    for i, st in enumerate(merged):
        st["id"] = i
        st["mode_switch"] = bool(i and st["mode"] != merged[i - 1]["mode"])

    json.dump({"video": video, "duration": dur, "fps": FPS,
               "params": {"pix_thr": PIX_THR, "change_th": CHANGE_TH, "min_gap": MIN_GAP},
               "states": merged}, open(f"{outdir}/states.json", "w"), ensure_ascii=False, indent=1)

    share = {}
    for s in merged:
        share[s["mode"]] = share.get(s["mode"], 0) + s["dur"]
    tot = sum(share.values()) or 1
    print(f"units={len(merged)} (pre-merge {len(units)}) mean_dur={np.mean([s['dur'] for s in merged]):.0f}s")
    print("time share: " + ", ".join(f"{k}={v/tot*100:.0f}%" for k, v in sorted(share.items(), key=lambda x: -x[1])))
    print(f"no-title: {sum(1 for s in merged if not s['title'])}  "
          f"mode-switch: {sum(s['mode_switch'] for s in merged)}")


if __name__ == "__main__":
    main()
