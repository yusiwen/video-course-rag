#!/usr/bin/env python3
"""Cut retrievable chunks on picture-unit boundaries.

One unit is usually one chunk; a unit longer than MAX_S is split at sentence
boundaries so a question about the middle of a five-minute slide still lands
near the sentence that answered it. Chunks never cross a unit, and every chunk
carries the chapter it belongs to, so retrieval can be aggregated back up.

Text comes from two independent channels and both are kept, tagged:
  [ASR]   what was said      (speech)
  [屏幕]  what was on screen (OCR, UI noise filtered out)

Usage: scripts/chunks.py <run-id>
"""
from ..paths import run as _run
import json
import re
import sys
from pathlib import Path

MAX_S = 40.0
MIN_S = 6.0

# UI furniture that survives OCR on this recording setup
UI_WORDS = ("开始", "插入", "设计", "切换", "动画", "幻灯片放映", "审阅", "视图",
            "帮助", "操作说明搜索", "单击此处", "备注", "批注", "PowerPoint",
            "文件", "主页", "共享", "开发工具", "加载项", "绘图")
CLOCK = re.compile(r"^\d{1,2}:\d{2}$")
CJK = re.compile(r"[\u4e00-\u9fff]")


def clean_screen(lines):
    out = []
    for ln in lines or []:
        t = ln.strip()
        if len(t) < 3 or CLOCK.match(t):
            continue
        if any(w in t for w in UI_WORDS) and len(t) < 14:
            continue
        if len(CJK.findall(t)) < 2:
            continue
        # short all-CJK lines are labels, names and app names ("参会人", "此电脑")
        if len(CJK.findall(t)) <= 4 and not re.search(r"[0-9A-Za-z]", t):
            continue
        # participant names and watermarks: 白代-泊案花, 石油-王少文
        if len(CJK.findall(t)) <= 6 and re.search(r"[-·•]", t) and not re.search(r"[0-9]", t):
            continue
        if t not in out:
            out.append(t)
    return out


def load_glossary(run_id):
    p = _run(run_id).glossary
    if not p.exists():
        return []
    terms = json.loads(p.read_text(encoding="utf-8"))
    return sorted(terms.items(), key=lambda kv: -len(kv[0]))


def fix_terms(text, glossary):
    for bad, good in glossary:
        text = text.replace(bad, good)
    return text


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "sample"
    outdir = str(_run(run_id).root)
    units = json.load(open(f"{outdir}/states.json"))["states"]
    chapters = json.load(open(f"{outdir}/chapters_fused.json"))["chapters"]
    sents = json.load(open(str(_run(run_id).asr_sent)))["items"]
    glossary = load_glossary(run_id)

    # sentence -> unit by midpoint
    by_unit = {u["id"]: [] for u in units}
    bounds = [(u.get("onset", u["start"]), u["end"], u["id"]) for u in units]
    for s in sents:
        mid = (s["start"] + s["end"]) / 2000.0
        for a, b, uid in bounds:
            if a <= mid < b:
                by_unit[uid].append(s)
                break

    def chap_of(t):
        for c in chapters:
            if c["start"] <= t < c["end"]:
                return c
        return chapters[-1] if chapters else None

    chunks = []
    for u in units:
        a = u.get("onset", u["start"])
        b = u["end"]
        my_sents = by_unit.get(u["id"], [])
        screen = clean_screen(u.get("ocr_lines"))
        groups = []
        cur = []
        for s in my_sents:
            if cur and (s["end"] / 1000.0 - cur[0]["start"] / 1000.0) > MAX_S:
                groups.append(cur)
                cur = []
            cur.append(s)
        if cur:
            groups.append(cur)
        if not groups:
            groups = [[]]

        for gi, g in enumerate(groups):
            start = round(g[0]["start"] / 1000.0, 2) if g else round(a, 2)
            end = round(g[-1]["end"] / 1000.0, 2) if g else round(b, 2)
            if end - start < MIN_S:
                end = round(min(b, start + MIN_S), 2)
            asr = fix_terms("".join(s["text"] for s in g).strip(), glossary)
            if not asr and not screen:
                continue
            # +1s slack: a sentence can start a fraction before the picture
            # flips, so a chunk at 344.14 belongs to the chapter starting 344.3
            c = chap_of(start + 1.0)
            text = ""
            if asr:
                text += "[ASR] " + asr + "\n"
            if screen:
                text += "[屏幕] " + " / ".join(screen[:12])
            chunks.append({
                "id": f"{run_id}-{len(chunks):04d}",
                "run_id": run_id,
                "start": start,
                "end": end,
                # the unit's jump points at the frame where the slide became
                # readable; if that moment is outside this window (a silent
                # intro inside one unit), seek to the window instead
                "jump": round(u["jump"], 2) if u["jump"] >= start - 0.1 else round(start + 0.3, 2),
                "mode": u["mode"],
                "unit": u["id"],
                "chapter_id": c["id"] if c else None,
                "chapter_title": c["title"] if c else "",
                "n_sent": len(g),
                "screen": screen,
                "text": text.strip(),
            })

    # tile: every second of the video belongs to exactly one chunk
    for i, c in enumerate(chunks):
        nxt = chunks[i + 1]["start"] if i + 1 < len(chunks) else None
        if nxt is not None and nxt > c["start"]:
            c["end"] = round(nxt, 2)

    # repeated screen text is boilerplate: the same slide title bar, the same
    # watermark, the same participant list shows up in dozens of chunks and
    # turns BM25 into a "which chunk also shows this slide" machine.
    freq = {}
    for c in chunks:
        for ln in set(c.get("screen", [])):
            freq[ln] = freq.get(ln, 0) + 1
    thresh = max(3, int(0.08 * len(chunks)))
    boiler = {ln for ln, n in freq.items() if n >= thresh}
    for c in chunks:
        keep = [ln for ln in c.get("screen", []) if ln not in boiler]
        if keep != c.get("screen"):
            c["screen"] = keep
            asr = c["text"].split("[屏幕]")[0].strip()
            c["text"] = (asr + "\n" if asr else "") + (
                "[屏幕] " + " / ".join(keep[:12]) if keep else "")
            c["text"] = c["text"].strip()
    chunks = [c for c in chunks if c["text"]]

    with open(f"{outdir}/chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    silent = sum(1 for c in chunks if c["n_sent"] == 0)
    dur = sum(c["end"] - c["start"] for c in chunks)
    print(f"{len(chunks)} chunks -> {outdir}/chunks.jsonl")
    print(f"  text-only-screen: {silent}   covered {dur/60:.1f} min   "
          f"chars: {sum(len(c['text']) for c in chunks)}   "
          f"boilerplate lines dropped: {len(boiler)}")
    for c in chunks[:3]:
        print(f"  {c['id']} [{c['start']:.0f}-{c['end']:.0f}s] {c['mode']:7} "
              f"{c['chapter_title'][:12]} | {c['text'][:70]!r}")


if __name__ == "__main__":
    main()
