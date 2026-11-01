#!/usr/bin/env python3
"""Derive a course's own title geometry instead of shipping hand-tuned constants.

retitle.py carries two kinds of knowledge: mechanism (take the largest line in
the title band) and material facts (where the title band is, which lines are
shell text, what counts as too small to be a title). The second kind is a guess
about the recording setup, and it is why a new course's table of contents can
degrade silently into "未登录" / "分享" / a product header.

This script measures those facts from the material itself:

  sample     one thumbnail per unit, spread over the run, up to N per mode
  grid       bin every OCR line into a coarse cell; per cell record presence,
             distinct texts, turnover = distinct/frames, median line height
  title band the cell row that is present in most frames, changes text often
             (a slide title turns over; a header, a menu bar or a product name
             does not) and holds the tallest text
  floor      0.75 x the median height of the chosen band — small text cannot be
             a slide title, and this is measured, not picked
  chrome     lines that dominate their mode but are not in the title band

Writes data/profiles/<run>.json + out/runs/<run>/calibration.md.

Usage:
    scripts/calibrate.py lecture-1           # derive, write profile, report
    scripts/calibrate.py lecture-1 --per-mode 40
    scripts/calibrate.py lecture-1 --recalibrate   # ignore the OCR cache
"""
from ..paths import run as _run, mkdirs
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

from video_rag.title.gate import WINDOW_RE, norm  # noqa: E402

from ..ocr import backend
BAND_COVERAGE_MIN = 0.30       # the title band must hold a candidate in this share of frames
EXCLUDE_Y = ((0.0, 0.09), (0.955, 1.0))   # taskbar strip, window title bar strip
DOMINANT_MAX = 0.90            # one spelling in ~100% of frames is a watermark or a
                              #    product name; an app nav bar cycles between a few
                              #    entries («表单设计器» / «订单管理») and is worth using
MIN_TEXTS = 3                  # a band must show at least this many different strings
AGREE_MIN = 0.10               # ...and its text must change at some point in the run
CHROME_IN_MODE = 0.40          # a line in this share of a mode's frames is chrome
FLOOR_RATIO = 0.75             # title font floor, relative to the band's median


def hhmm(t):
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def load(run_id):
    st = json.load(open(str(_run(run_id).states)))
    return st


def ocr_frame(path, cache, recalibrate):
    key = os.path.basename(path)
    if not recalibrate and key in cache:
        return cache[key]
    lines = backend.read(path)
    cache[key] = lines
    return lines


def frame_candidates(lineset, top=2):
    """The tallest lines of ONE frame — the title is one of them, by definition.

    Comparing heights across frames needs a threshold nobody can pick; comparing
    inside a frame needs none. A slide title is the tallest text on its slide,
    while ribbon labels, menus and a taskbar clock are small regardless of what
    the rest of the screen is doing. So: per frame, keep the `top` tallest lines
    and let the cross-frame statistics decide where they live.
    """
    if not lineset:
        return []
    good = [l for l in lineset if len(norm(l["text"])) >= 2]
    if not good:
        return []
    hmax = max(l["h"] for l in good)
    return [l for l in good if l["h"] >= 0.80 * hmax][:top]


def cluster_y(cands, nframes, tol=0.05):
    """Group candidate lines by vertical position; keep the bands that behave."""
    buckets = defaultdict(list)
    for i, cs in cands.items():
        for l in cs:
            y = l["y"] + l["h"] / 2
            buckets[round(y / tol)].append((i, l))
    bands = []
    for key, items in buckets.items():
        frames = {i for i, _ in items}
        texts = Counter(norm(l["text"]) for _, l in items)
        if not texts:
            continue
        dom = texts.most_common(1)[0][1] / len(items)
        present = sorted(frames)
        by_frame = defaultdict(set)
        for i, l in items:
            by_frame[i].add(norm(l["text"]))
        pairs = list(zip(present, present[1:]))
        agree = (sum(1 for a, b in pairs if by_frame[a] != by_frame[b]) / len(pairs)
                 if pairs else 0.0)
        bands.append({
            "y": key * tol, "frames": len(frames), "frameset": frames,
            "coverage": len(frames) / max(1, nframes),
            "distinct": len(texts), "agree": agree,
            "h": statistics.median(l["h"] for _, l in items),
            "dominant": dom, "texts": texts,
            "x0": min(l["x"] for _, l in items),
            "x1": max(l["x"] + l["w"] for _, l in items),
            "samples": [t for t, _ in sorted(texts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]],
        })
    return bands


def looks_like_window_title(texts):
    """A band whose text is the window title bar names the app, not the内容.

    Measured on a screen-recording course: the tallest, most stable band of the
    fullscreen-app mode was the browser tab title («产品开发平台 · System
    Dashboard»), which then named 57 chapters after the product. Those strings
    are recognisable by shape, so the band is refused and the mode falls back to
    "no title row" — a neutral segment beats forty rows of the same app name.
    """
    hits = [t for t in texts if WINDOW_RE.search(t)]
    return len(texts) >= 2 and len(hits) / len(texts) >= 0.5


def merge_bands(bands, tol=0.05):
    """Join adjacent buckets: one slide title straddles two of them easily."""
    bands = sorted(bands, key=lambda b: b["y"])
    out = []
    for b in bands:
        if out and b["y"] - out[-1]["y"] <= 1.2 * tol:
            a = out[-1]
            a["texts"].update(b["texts"])
            a["frameset"] |= b["frameset"]
            a["frames"] = len(a["frameset"])
            a["dominant"] = a["texts"].most_common(1)[0][1] / max(1, sum(a["texts"].values()))
            a["h"] = max(a["h"], b["h"])
            a["x0"], a["x1"] = min(a["x0"], b["x0"]), max(a["x1"], b["x1"])
            a["y"] = (a["y"] + b["y"]) / 2
            a["distinct"] = len(a["texts"])
            a["agree"] = max(a["agree"], b["agree"])
            a["samples"] = [t for t, _ in sorted(a["texts"].items(), key=lambda kv: (-kv[1], kv[0]))[:6]]
        else:
            out.append(dict(b))
    return out


def pick_band(cands, nframes):
    """The band that is tall, frequent, and does not repeat one string forever."""
    bands = [b for b in merge_bands(cluster_y(cands, nframes))
             if b["coverage"] >= BAND_COVERAGE_MIN
             and b["dominant"] <= DOMINANT_MAX
             and b["distinct"] >= MIN_TEXTS
             and b["agree"] >= AGREE_MIN
             and not any(lo <= b["y"] <= hi for lo, hi in EXCLUDE_Y)]
    if not bands:
        return None
    # tall and frequent wins; a header has coverage but dominant ~1.0 and is
    # already filtered out, a ribbon strip is short, a clock sits in the taskbar
    best = max(bands, key=lambda b: b["h"] * b["coverage"])
    top = [t for t, _ in sorted(best["texts"].items(), key=lambda kv: (-kv[1], kv[0]))[:6]]
    if looks_like_window_title(top):
        return None
    return best


def calibrate(run_id, per_mode=24, recalibrate=False):
    st = load(run_id)
    units = st["states"]
    by_mode = defaultdict(list)
    for u in units:
        by_mode[u.get("mode", "?")].append(u)

    cache_path = str(_run(run_id).calib_ocr)
    cache = {}
    if os.path.exists(cache_path) and not recalibrate:
        try:
            # A frame with no text is a frame whose OCR failed, not a frame that
            # was empty. Keeping those entries would freeze one broken run into
            # every later calibration, so they are treated as misses.
            cache = {k: v for k, v in json.load(open(cache_path)).items() if v}
        except Exception:
            cache = {}

    profile = {"run_id": run_id, "video": st["video"], "duration": st["duration"],
               "params": {"dominant_max": DOMINANT_MAX,
                          "band_coverage_min": BAND_COVERAGE_MIN,
                          "chrome_in_mode": CHROME_IN_MODE,
                          "floor_ratio": FLOOR_RATIO, "per_mode": per_mode},
               "modes": {}, "chrome": {"global": [], "per_mode": {}},
               "acronyms": [], "uncertain": []}

    all_upper = Counter()
    frames_by_mode = {}
    for mode, us in sorted(by_mode.items()):
        if len(us) < 5:
            profile["uncertain"].append(f"模式 {mode} 只有 {len(us)} 个单元，样本不足")
            continue
        step = max(1, len(us) // per_mode)
        sample = us[::step][:per_mode]
        ocred = []
        for u in sample:
            p = str(_run(run_id).root / u["thumb"])
            if os.path.exists(p):
                ocred.append(ocr_frame(p, cache, recalibrate))
        if len(ocred) < 3:
            profile["uncertain"].append(f"模式 {mode} 只能读到 {len(ocred)} 帧")
            continue
        cands = {i: frame_candidates(l) for i, l in enumerate(ocred)}
        band = pick_band(cands, len(ocred))
        floor = 0.0
        mode_info = {"frames": len(ocred), "units": len(us), "band": None}
        if band:
            floor = round(FLOOR_RATIO * band["h"], 4)
            # How often does this band actually name a frame? Coverage only says
            # "the strip holds a candidate somewhere", so a strip whose one stable
            # string is the product header passes coverage and still names nothing:
            # measured on an application course, 29% of frames yield a title from
            # its band while the figure sits at 58-92% when the band is a real one.
            # Store the number instead of a verdict — 33% is a working band on a
            # lecture that is mostly transitions, so it cannot be a pass/fail gate.
            y0, y1 = band["y"] - 0.04, band["y"] + 0.04
            x0, x1 = band["x0"], band["x1"]
            dom = norm(sorted(band["texts"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
            got = 0
            for lines in ocred:
                if any(y0 <= l["y"] <= y1 and x0 - 0.02 <= l["x"] <= x1 + 0.02
                       and l["h"] >= floor and len(l["text"].strip()) >= 2
                       and norm(l["text"]) != dom for l in lines):
                    got += 1
            band["yield"] = round(got / max(1, len(ocred)), 3)
            mode_info["band"] = {
                "y": round(band["y"], 3),
                "y_range": [round(band["y"] - 0.04, 3), round(band["y"] + 0.04, 3)],
                "x_range": [round(band["x0"], 3), round(band["x1"], 3)],
                "median_h": round(band["h"], 4),
                "coverage": round(band["coverage"], 3),
                "dominant": round(band["dominant"], 3),
                "dominant_text": sorted(band["texts"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0],
                "samples": band["samples"],
                "yield": band.get("yield"),
            }
            mode_info["h_floor"] = floor
        else:
            profile["uncertain"].append(
                f"模式 {mode} 找不到“位置稳定 + 每帧最高 + 不重复同一串”的标题带")
        # chrome inside the mode: frequent lines outside the title band
        # Count once per frame, in the order the lines were read. Iterating a set
        # here made the counter's insertion order depend on string hashing, so the
        # same material produced a different chrome list — and a different table of
        # contents — on every run; the list is also truncated at 40 entries, which
        # made the difference visible.
        freq = Counter()
        for lineset in ocred:
            seen = set()
            for l in lineset:
                t = l["text"].strip()
                if t and t not in seen:
                    seen.add(t)
                    freq[t] += 1
        in_band = set()
        if band:
            y0, y1 = band["y"] - 0.05, band["y"] + 0.05
            for lineset in ocred:
                for l in lineset:
                    if y0 <= l["y"] + l["h"] / 2 <= y1 and l["h"] >= floor:
                        in_band.add(norm(l["text"]))
        mode_chrome = [t for t, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
                       if c / len(ocred) >= CHROME_IN_MODE and norm(t) not in in_band]
        profile["chrome"]["per_mode"][mode] = mode_chrome[:40]
        for t in freq:
            if t and t.isascii() and t.isupper() and len(t) <= 10 and freq[t] >= 3:
                all_upper[t] += freq[t]
        profile["modes"][mode] = mode_info
        frames_by_mode[mode] = len(ocred)

    profile["acronyms"] = [t for t, _ in sorted(all_upper.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    # global chrome: frequent across every mode
    total_frames = sum(frames_by_mode.values()) or 1
    allfreq = Counter()
    for mode, lst in profile["chrome"]["per_mode"].items():
        for t in lst:
            allfreq[t] += 1
    profile["chrome"]["global"] = [t for t, c in allfreq.items() if c >= max(2, len(frames_by_mode) - 1)]

    out = str(_run(run_id).profile)
    json.dump(profile, open(out, "w"), ensure_ascii=False, indent=1)
    json.dump({k: v for k, v in cache.items() if v}, open(cache_path, "w"))

    lines = [f"# 标定报告 · {run_id}", "",
             f"- 视频 {profile['video']}", f"- 时长 {st['duration'] / 60:.0f} 分钟",
             f"- 采样 {sum(frames_by_mode.values())} 帧", ""]
    for mode, info in profile["modes"].items():
        b = info["band"]
        lines.append(f"## 模式 {mode}（{info['units']} 单元，采样 {info['frames']} 帧）")
        lines.append("")
        if b:
            lines.append(f"- 标题带：屏幕 y {b['y_range'][0]:.2f}–{b['y_range'][1]:.2f}"
                         f"，x {b['x_range'][0]:.2f}–{b['x_range'][1]:.2f}"
                         f"，中位字高 {b['median_h']}，覆盖率 {b['coverage']:.2f}"
                         f"，单串占比 {b['dominant']:.2f}")
            lines.append(f"- 字号下限 h ≥ {info['h_floor']}")
            lines.append(f"- 带内文字样例：{'、'.join(b['samples'])}")
        else:
            lines.append("- 没找到标题带，这个模式的章节标题不应来自画面")
        ch = profile["chrome"]["per_mode"].get(mode, [])
        if ch:
            lines.append(f"- 模式内外壳文字（{len(ch)} 条）：{'、'.join(ch[:12])}")
        lines.append("")
    if profile["uncertain"]:
        lines.append("## 需要人看一眼")
        lines.append("")
        for u in profile["uncertain"]:
            lines.append(f"- {u}")
        lines.append("")
    open(str(mkdirs(_run(run_id).report_dir / "calibration.md")), "w").write("\n".join(lines))

    print(f"profile -> {out}")
    for mode, info in profile["modes"].items():
        b = info["band"]
        print(f"  {mode:<9} frames={info['frames']:>2}  "
              f"band={'y%.2f h=%.3f cov=%.2f dom=%.2f yield=%.0f%%' % (b['y'], b['median_h'], b['coverage'], b['dominant'], (b.get('yield') or 0) * 100) if b else 'none'}"
              f"  floor={info.get('h_floor')}")
    if profile["uncertain"]:
        print("  待人工确认：" + "; ".join(profile["uncertain"][:3]))
    return profile


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_id = args[0] if args else "sample"
    per_mode = 24
    for a in sys.argv[1:]:
        if a.startswith("--per-mode="):
            per_mode = int(a.split("=", 1)[1])
    calibrate(run_id, per_mode=per_mode, recalibrate="--recalibrate" in sys.argv)


if __name__ == "__main__":
    main()
