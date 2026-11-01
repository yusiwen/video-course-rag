#!/usr/bin/env python3
"""Re-derive mode + title from the saved full-frame thumbnails.

Why a separate pass: states_v4 OCRs a *cropped* image per mode, so region
filtering and mode detection are entangled and every crop needs its own ffmpeg
call. Vision already returns normalised boxes, so one OCR on the full frame
gives both: filter boxes by region in Python, no crops, no ffmpeg-9 expression
trap.

Rules that fix the five junk titles in this sample:
  - never take a line from the taskbar band (y < 0.10) or a PPT logo corner
  - reject ribbon/menu/taskbar vocabulary and single-character boxes
  - reject short pure-ASCII garbage ("pogle") unless it is a known acronym
  - prefer the largest font inside the top 45% of the content region, because
    that is where slide titles and page H1s live
  - if nothing survives, mark title_ok=False and let the chapter layer merge
    the unit away instead of inventing a title
"""
from ..paths import run as _run
import json
import os
import re
import subprocess
import sys

from video_rag.title.gate import WINDOW_RE, similar  # noqa: E402

UI_BLOCK = {
    "文件", "开始", "插入", "设计", "切换", "动画", "幻灯片放映", "审阅", "视图", "帮助",
    "操作说明搜索", "单击此处添加备注", "备注", "批注", "移动到", "复制到", "全部取消",
    "朗读此页内容", "正编辑", "共享", "演讲者视图", "百度网盘", "腾讯QQ", "微信", "回收站",
    "此电脑", "开始菜单", "任务视图", "删除", "重命名", "公告", "相册",
    "应用设置", "新建", "属性", "打开", "全部选择",
}
ACRONYMS = {"API", "URL", "PDF", "PPT", "SSO", "LDAP", "JIRA", "SQL", "JSON", "HTTP",
            "CSV", "XML", "HTML", "UUID", "GPU", "CPU", "YAML"}
REGIONS = {  # x0, y0, x1, y1 in Vision coords (origin bottom-left)
    "ppt":     (0.132, 0.185, 0.843, 0.910),
    # y1 capped at 0.93: a window title bar sits at ~0.95 and its text is the
    # file name ("...pptx - PowerPoint"), not the slide title
    "ppt-fs":  (0.020, 0.070, 0.980, 0.930),
    "browser": (0.020, 0.080, 0.980, 0.920),
    "other-app": (0.020, 0.100, 0.980, 0.920),
    "desktop": (0.020, 0.100, 0.980, 0.920),
}
CJK = re.compile(r"[\u4e00-\u9fff]")


def ocr(png):
    r = subprocess.run(["./scripts/ocr", png, "0.25"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout or "[]")
    except Exception:
        return []


def inside(l, box):
    cx, cy = l["x"] + l["w"] / 2, l["y"] + l["h"] / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def classify(lines):
    top = " ".join(l["text"] for l in lines if l["y"] > 0.86)
    if any(t in top for t in ("文件", "开始", "插入", "设计", "切换", "动画",
                              "幻灯片放映", "审阅", "视图", "PowerPoint")) or "pptx" in top.lower():
        return "ppt"
    if re.search(r"(https?://|www\.|\b[a-z0-9][a-z0-9.-]*\.(cn|com|net|org|io)\b)", top, re.I):
        return "browser"
    body = [l for l in lines if l["y"] > 0.10]
    if len(body) >= 5:
        return "ppt-fs"
    return "other-app" if body else "desktop"


def ok(text):
    t = text.strip(" ·|—-·")
    if len(t) < 2 or len(t) > 60 or re.fullmatch(r"[\W\d_]+", t):
        return False
    if t in UI_BLOCK:
        return False
    if any(w in t for w in ("删除", "重命名", "全部取消", "全部选择", "移动到", "复制到",
                            "新建", "属性", "打开", "默认值")):
        return False
    if t.isascii() and len(t) <= 8 and t not in ACRONYMS:
        return False
    if re.search(r"\.pptx|PowerPoint|【兼容模式】", t, re.I):
        return False
    if not CJK.search(t) and t not in ACRONYMS:
        return False
    return True


def pick(lines, box):
    """Topmost valid line wins, tie-break on font size.

    Font size alone cannot separate a slide title from body copy in these
    decks (heading h 0.036 vs body 0.032); position does. Anything longer than
    a headline is a paragraph and is skipped.
    """
    valid = []
    for l in lines:
        if not inside(l, box):
            continue
        if l["x"] > 0.78 and l["y"] > 0.82:      # logo corner
            continue
        t = l["text"].strip()
        if not ok(t) or len(t) > 40:
            continue
        valid.append(l)
    if not valid:
        return "", None
    # largest font wins, then highest on screen. Topmost-alone picks the
    # bookmarks bar (a lone product name) and taskbar labels; largest-alone picks
    # body copy.
    valid.sort(key=lambda l: (-l["h"], -l["y"]))
    top = valid[0]
    return top["text"].strip(" ·|—-·"), round(top["h"], 3)


def _norm(t):
    return "".join(ch for ch in (t or "").lower()
                   if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def load_profile(outdir):
    """The course's own measured title geometry, if it has been calibrated."""
    run_id = os.path.basename(os.path.normpath(outdir))
    p = _run(run_id).profile
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out/states"
    path = f"{outdir}/states.json"
    d = json.load(open(path))
    stem = os.path.splitext(os.path.basename(d["video"]))[0]
    profile = load_profile(outdir)
    prof_modes = (profile or {}).get("modes", {})
    no_title_modes = 0
    ov_path = next((p for p in (f"{outdir}/title_overrides.json",
                                str(_run(stem).overrides)) if os.path.exists(p)), None)
    overrides = {}
    if ov_path:
        raw = json.load(open(ov_path))
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            overrides[float(k)] = v
    changed = 0
    for st in d["states"]:
        png = f"{outdir}/{st['thumb']}"
        if not os.path.exists(png):
            continue
        lines = ocr(png)
        mode = classify(lines)
        # Calibrated geometry beats the shipped guess. A mode with no title band
        # has no title band: saying nothing is better than naming a chapter after
        # the product header that happens to sit where a title would be.
        use_title, score = True, None
        pm = prof_modes.get(mode)
        if pm is not None:
            band = pm.get("band")
            if not band:
                use_title = False
                no_title_modes += 1
                title = ""
            else:
                box = (band["x_range"][0], band["y_range"][0],
                       band["x_range"][1], band["y_range"][1])
                # A band can hold the app's product name next to real navigation
                # entries («表单设计器», «订单管理»). The name is the same string in
                # most frames of the band, so naming chapters after it repeats the
                # product forty times; drop it and take the entry behind it.
                if band.get("dominant_text") and band.get("dominant", 0) >= 0.35:
                    dom = _norm(band["dominant_text"])
                    kept = [l for l in lines
                            if not similar(_norm(l["text"]), dom)]
                    lines = kept or lines
                # a window title bar inside the band names the app, not the topic
                notwin = [l for l in lines if not WINDOW_RE.search(l["text"])]
                lines = notwin or lines
                title, score = pick(lines, box)
                if title and score is not None and score < pm.get("h_floor", 0):
                    title, use_title = "", False       # too small to be a title
        else:
            box = REGIONS.get(mode, REGIONS["ppt-fs"])
            title, score = pick(lines, box)
        if title != st["title"] or mode != st["mode"]:
            changed += 1
        st["mode"] = mode
        st["title"] = title
        st["title_ok"] = bool(title) and use_title
        # manual override, matched by nearest start within 0.5 s
        hit = next((k for k in overrides if abs(k - st["start"]) <= 0.5), None)
        if hit is not None:
            if overrides[hit] is None:
                st["title"] = ""
                st["title_ok"] = False
                st["override"] = "merge"
            else:
                st["title"] = overrides[hit]
                st["title_ok"] = True
                st["override"] = True
        st["title_score"] = round(score, 3) if score else None
        st["ocr_lines"] = [l["text"].strip() for l in
                           sorted(lines, key=lambda z: -z["y"]) if l["text"].strip()][:24]
    json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)

    def mmss(x):
        return f"{int(x//60):02d}:{int(x%60):02d}"

    print(f"{changed}/{len(d['states'])} units changed mode or title")
    for st in d["states"]:
        flag = "" if st["title_ok"] else "  <-- no usable title"
        print(f"{st['id']:>3} {mmss(st['start'])} {st['mode']:<10} {st['title'][:46]}{flag}")


if __name__ == "__main__":
    main()
