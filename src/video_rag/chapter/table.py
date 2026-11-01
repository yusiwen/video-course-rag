#!/usr/bin/env python3
"""Collapse fine units into chapters.

Merging rules, in order:
  1. a unit with no usable title is a transition, not a chapter -- fold it into
     the neighbour that shares its mode (previous first, then next); fall back to
     the previous unit. "No usable title" covers title_ok=false AND any title the
     gate rejects: a unit titled after the product header or a menu label is a
     transition too, even though retitle accepted it.
  2. units shorter than MIN_CH behave the same way
  3. neighbours with an identical title and mode merge

A chapter that still has no trustworthy title when everything settles gets a
neutral label derived from what was measured (mode + duration). Inventing a
topic out of OCR debris is worse than saying 「桌面操作（3 分钟）」: the first
looks like information, the second is honest.
"""
from ..paths import run as _run
import json
import os
import sys

from video_rag.title.retitle import load_profile, ok as title_ok  # noqa: E402
from video_rag.title.gate import (SEP, chrome_and_acronyms, chrome_for, dur_txt,
                       neutral_title, norm, similar, verdict)  # noqa: E402


def main():
    MIN_CH = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    rid = sys.argv[2] if len(sys.argv) > 2 else "out/states"
    outdir = str(_run(rid).root) if not os.path.isdir(rid) else rid
    d = json.load(open(f"{outdir}/states.json"))
    units = [dict(u) for u in d["states"]]
    chrome, mode_chrome, acronyms = chrome_and_acronyms(units)
    # Modes the calibration could not give a title band are kept as neutral
    # segments: folding them away would glue hours of screen time into one
    # chapter, and naming them after whatever OCR found would be worse still.
    prof = load_profile(outdir) or {}
    bandless = {m for m, info in prof.get("modes", {}).items() if not info.get("band")}

    def usable(u):
        c = chrome_for(u.get("mode"), chrome, mode_chrome)
        return u.get("title_ok", True) and verdict(u.get("title"), c, acronyms) is None

    def keep_as_segment(u):
        return u.get("mode") in bandless

    def fold_target(i):
        u = units[i]
        if i > 0 and units[i - 1]["mode"] == u["mode"]:
            return units[i - 1]
        if i + 1 < len(units) and units[i + 1]["mode"] == u["mode"]:
            return units[i + 1]
        return units[i - 1] if i > 0 else None

    def fold(i):
        u = units[i]
        tgt = fold_target(i)
        if tgt is not None:
            tgt["start"] = min(tgt["start"], u["start"])
            tgt["end"] = max(tgt["end"], u["end"])
        return units[:i] + units[i + 1:]

    CH_CAP = 300.0     # a fold that would build a chapter longer than this is refused
    i = 0
    while i < len(units) and len(units) > 1:
        u = units[i]
        droppable = ((not usable(u)) or (u["end"] - u["start"] < MIN_CH)) \
            and not keep_as_segment(u)
        if droppable:
            tgt = fold_target(i)
            span = (max(tgt["end"], u["end"]) - min(tgt["start"], u["start"])) if tgt else 0.0
            if tgt is None or span > CH_CAP:
                # Folding is for transitions, not for minutes of unnameable screen:
                # 353 untitled units folded into their neighbours produced a
                # 69-minute "chapter" (185 minutes on a frozen capture). Keep it as
                # its own segment instead and let the neutral pass name it.
                u["title_kind"] = "neutral"
                u["dropped_title"] = u["title"]
                u["title"] = ""          # labelled after merging, from the final span
                i += 1
                continue
            units = fold(i)
            continue
        i += 1

    SEG_MAX = 360.0        # neutral segments are capped so one row never owns an hour
    merged = []
    for c in units:
        if merged:
            last = merged[-1]
            same_mode = c["mode"] == last["mode"]
            if c["title"] and c["title"] == last["title"] and same_mode:
                last["end"] = c["end"]
                continue
            if not c["title"] and not last["title"] and same_mode \
                    and c["end"] - last["start"] <= SEG_MAX:
                last["end"] = c["end"]
                continue
        merged.append(c)

    # Nothing trustworthy survived: say what was measured instead of inventing a
    # topic. This has to run BEFORE the dedupe pass, which would otherwise see a
    # dozen empty titles as "the same title repeated" and bolt an arbitrary OCR
    # line onto them (" · 石化-范家彬").
    neutral = 0
    for c in merged:
        if c.get("title_kind") == "neutral":
            c["title"] = neutral_title(c.get("mode"), c["end"] - c["start"])
            neutral += 1
            continue
        if verdict(c["title"], chrome_for(c.get("mode"), chrome, mode_chrome), acronyms) is not None:
            c["title_kind"] = "neutral"
            c["dropped_title"] = c["title"]
            c["title"] = neutral_title(c.get("mode"), c["end"] - c["start"])
            neutral += 1
        else:
            c.pop("title_kind", None)

    # disambiguate repeated titles: same deck header on several slides is
    # useless in a table of contents, so append the next distinct line
    from collections import Counter
    dup = Counter(c["title"] for c in merged if c.get("title_kind") != "neutral")
    # A "title" whose OCR reading is different every time is a header, not a
    # name: real slide titles read consistently. Measured on a screen-recording
    # course, the app's product name appeared as four readings over 24 chapters
    # («ACME企业开放服务云平台» / «XACME…» / «ACME3…»), while three genuinely different
    # chapters on one topic («…服务代码书写» / «…服务注册» / «…表单配置服务») stay
    # distinct at 0.90 similarity and must not be touched.
    clusters = []
    for c in merged:
        if c.get("title_kind") == "neutral" or not c.get("title"):
            continue
        nb = norm(c["title"].split(SEP)[0])
        for cl in clusters:
            if similar(nb, cl["norm"], 0.90):
                cl["items"].append(c)
                cl["variants"].add(nb)
                break
        else:
            clusters.append({"norm": nb, "items": [c], "variants": {nb}})
    noisy = 0
    for cl in clusters:
        if len(cl["variants"]) >= 3 and len(cl["items"]) >= 6:
            for c in cl["items"]:
                c["title_kind"] = "neutral"
                c["dropped_title"] = c["title"]
                c["title"] = neutral_title(c.get("mode"), c["end"] - c["start"])
                noisy += 1

    for n, c in enumerate(merged):
        if c.get("title_kind") == "neutral":
            continue
        if dup[c["title"]] > 1:
            RIBBON = ("操作说明搜索", "绘图工具", "幻灯片放映", "开始", "插入",
                      "设计", "切换", "动画", "审阅", "视图", "绘图")
            extra = next((t for t in c.get("ocr_lines", [])
                          if t != c["title"] and len(t) >= 4 and title_ok(t)
                          and verdict(t, chrome_for(c.get("mode"), chrome, mode_chrome), acronyms) is None
                          and not any(w in t for w in RIBBON)), "")
            if extra:
                c["title"] = f"{c['title']} · {extra[:18]}"

    # A "title" whose OCR reading is different every time is a header, not a
    # name: real slide titles read consistently. Measured on a screen-recording
    # course, the app's product name appeared as four readings over 24 chapters
    # («ACME企业开放服务云平台» / «XACME…» / «ACME3…»), while three genuinely different
    # chapters on one topic («…服务代码书写» / «…服务注册» / «…表单配置服务») stay
    # distinct at 0.90 similarity and must not be touched.
    clusters = []
    for c in merged:
        if c.get("title_kind") == "neutral" or not c.get("title"):
            continue
        nb = norm(c["title"].split(SEP)[0])
        for cl in clusters:
            if similar(nb, cl["norm"], 0.90):
                cl["items"].append(c)
                cl["variants"].add(nb)
                break
        else:
            clusters.append({"norm": nb, "items": [c], "variants": {nb}})
    noisy = 0
    for cl in clusters:
        if len(cl["variants"]) >= 3 and len(cl["items"]) >= 6:
            for c in cl["items"]:
                c["title_kind"] = "neutral"
                c["dropped_title"] = c["title"]
                c["title"] = neutral_title(c.get("mode"), c["end"] - c["start"])
                noisy += 1

    for n, c in enumerate(merged):
        c["id"] = n
        c["dur"] = round(c["end"] - c["start"], 2)
        c["mode_switch"] = bool(n and c["mode"] != merged[n - 1]["mode"])
        for k in ("jump", "jump_lag", "jump_title_match", "title_score"):
            c.pop(k, None)

    json.dump({"video": d["video"], "duration": d["duration"],
               "source": f"{outdir}/states.json", "min_chapter": MIN_CH,
               "chapters": merged}, open(f"{outdir}/chapters.json", "w"),
              ensure_ascii=False, indent=1)

    def mmss(x):
        return f"{int(x//60):02d}:{int(x%60):02d}"

    print(f"{len(d['states'])} units -> {len(merged)} chapters (min {MIN_CH:.0f}s), "
          f"{neutral + noisy} 章用中性标题（标题不可信/页头）")
    for c in merged:
        print(f"{c['id']:>3} {mmss(c['start'])}-{mmss(c['end'])} {c['dur']:>6.0f}s "
              f"{c['mode']:<9} {c['title'][:46]}")


if __name__ == "__main__":
    main()
