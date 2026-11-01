#!/usr/bin/env python3
"""Refine 2fps unit boundaries to the exact frame at native fps.

2fps sampling puts every boundary within +/-0.5 s. Since the file really is
30fps, each boundary can be pinned by decoding a 1s window around it and
taking the first frame whose change vs the previous frame exceeds ONSET.
"""
import json
import subprocess
import sys

import numpy as np

ONSET = 0.004      # 0.4% of pixels changed at native fps = a real onset
W, H = 320, 180


def frames(path, t0, dur):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", path,
                        "-t", f"{dur:.3f}", "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True)
    fr = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(fr) // (W * H)
    return fr[: n * W * H].reshape(n, H, W).astype(np.int16)


def refine(path, b):
    """Return (content_start, onset). Content start = first frame after the
    transition settles (10 consecutive frames with change < ONSET), because
    jumping to the onset lands mid-fade on a half-drawn slide."""
    fr = frames(path, b - 0.7, 2.0)
    if len(fr) < 12:
        return None
    d = np.array([(np.abs(fr[i] - fr[i - 1]) > 20).mean() for i in range(1, len(fr))])
    hit = np.where(d > ONSET)[0]
    if not len(hit):
        return None
    onset_i = int(hit[0]) + 1
    t0 = b - 0.7
    onset = t0 + onset_i / 30.0
    stable = np.where(d < ONSET)[0]
    for j in stable:
        if j < onset_i:
            continue
        if all((j + k) < len(d) and d[j + k] < ONSET for k in range(10)):
            return t0 + j / 30.0, onset
    return onset + 0.3, onset


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out/states"
    d = json.load(open(f"{outdir}/states.json"))
    src = d["video"]
    shifted = 0
    for i, st in enumerate(d["states"][1:], start=1):
        res = refine(src, st["start"])
        if res is None:
            continue
        nb, onset = res
        delta = nb - st["start"]
        st["start"] = round(nb, 3)
        st["onset"] = round(onset, 3)
        st["refined_delta"] = round(delta, 3)
        if abs(delta) > 0.2:
            shifted += 1
    # recompute ends/durations from refined starts
    for i, st in enumerate(d["states"]):
        end = d["states"][i + 1]["start"] if i + 1 < len(d["states"]) else d["duration"]
        st["end"] = round(end, 3)
        st["dur"] = round(end - st["start"], 3)
    d["params"]["boundary_refined_native_fps"] = True
    json.dump(d, open(f"{outdir}/states.json", "w"), ensure_ascii=False, indent=1)
    print(f"refined {len(d['states']) - 1} boundaries; {shifted} moved >0.2s")
    for st in d["states"]:
        print(f"  {st['id']:>2} {st['start']:8.3f}  dur {st['dur']:6.2f}  "
              f"shift {st.get('refined_delta', 0):+.3f}s  {st['title'][:34]}")


if __name__ == "__main__":
    main()
