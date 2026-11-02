#!/usr/bin/env python3
"""Fuse transcript-level chapter boundaries with picture-level unit boundaries.

Text decides WHERE a chapter starts (semantics); the picture decides WHICH FRAME
to show, because a lecturer usually announces the next topic a few seconds
before clicking. Two rules:

  snap     text boundary t_text -> nearest visual boundary onset within SNAP_S
  rescue   a visual boundary that falls inside a speech gap longer than GAP_S
           becomes its own chapter (the lecturer was silent and operating, so
           the transcript has nothing to say about that transition)

Inputs : data/asr/<id>_sent.json      sentences with absolute ms
         data/llm/<id>_chapters.json  [{start_sentence, end_sentence, title, summary}]
         out/runs/<id>/states.json    visual units with onset/jump/mode/title
Output : out/runs/<id>/chapters_fused.json
"""
from ..paths import run as _run
import bisect
import difflib
import re
import json
import os
import sys

from video_rag.title.gate import (chrome_and_acronyms, chrome_for, neutral_title,
                       verdict)  # noqa: E402

# asymmetric: the picture usually changes BEFORE the lecturer names the topic
# (click, then explain), but sometimes he announces first and clicks 5-8s later
SNAP_BACK = 25.0
SNAP_FWD = 8.0
GAP_S = 15.0
MAX_CHAPTER_S = 300.0   # a chapter longer than this is too coarse, split it

# A silence is MEASURED, not guessed. The transcript says nobody was talking; the
# picture says whether anything was happening. Both signals dead = no content.
SILENCE_S = 120.0       # no speech for this long makes the span a candidate
BIG_FRAC = 0.02         # a frame differing in >=2% of its pixels is a content
                        # change (page turn, scroll, window switch); a cursor
                        # drifting measures 0.05-0.5% and must not count
MERGE_S = 15.0          # clicks of one operating session are closer than this;
                        # a longer pause starts a new session
OP_MIN_S = 30.0         # a session shorter than this is a stray click on an
                        # otherwise frozen screen, not content
ROW_MIN_S = 60.0        # dead time shorter than this is not worth a TOC row
ACT_FPS, ACT_W, ACT_H, ACT_PIX = 2.0, 320, 180, 20
ACT_CACHE = "activity.json"
SUFFIX = "（续）"
CONT_TAIL = re.compile(r"（续\d*）$")     # 已带续号（含「（续5）」）就算标过了


def hhmm(t):
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def dur_txt(sec):
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} 秒"
    m = sec / 60.0
    return f"{m:.0f} 分钟" if m >= 2 else f"{m:.1f} 分钟"


def big_change_times(video, t0, t1, cache):
    """Times in [t0,t1] where the picture changed by at least BIG_FRAC.

    Same sampling as states_v4 (2 fps, 320x180 grey, bottom 8% masked so the
    taskbar clock cannot fake motion) but a much larger area threshold, because
    the question here is "did anything happen", not "did the layout change".
    """
    key = f"{t0:.1f}-{t1:.1f}"
    if key in cache:
        return cache[key]
    import subprocess

    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.2f}", "-i", video,
                        "-t", f"{max(0.0, t1 - t0):.2f}",
                        "-vf", f"fps={ACT_FPS},scale={ACT_W}:{ACT_H}",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(a) // (ACT_W * ACT_H)
    times = []
    if n >= 2:
        fr = a[: n * ACT_W * ACT_H].reshape(n, ACT_H, ACT_W).astype(np.int16)
        fr[:, int(ACT_H * 0.92):, :] = 0
        frac = (np.abs(fr[1:] - fr[:-1]) > ACT_PIX).mean(axis=(1, 2))
        times = [round(t0 + (i + 2) / ACT_FPS, 2) for i in np.where(frac >= BIG_FRAC)[0]]
    cache[key] = times
    return times


def op_sessions(times, lo, hi):
    """Group content changes into operating sessions (first, last)."""
    sess = []
    for t in times:
        if not (lo <= t <= hi):
            continue
        if sess and t - sess[-1][1] <= MERGE_S:
            sess[-1][1] = t
        else:
            sess.append([t, t])
    return [(a, b) for a, b in sess if b - a >= OP_MIN_S]


def quiet_rows(g0, g1, video, cache):
    """Lay one silent span out as 休息 / 操作演示 rows backed by measurement."""
    try:
        times = big_change_times(video, g0, g1, cache)
    except Exception:
        return [("静音段（%s）" % dur_txt(g1 - g0),
                 f"{hhmm(g0)}–{hhmm(g1)} 无法读取视频，未能判断画面是否变化", g0, g1)]
    pieces, head = [], g0
    for s0, s1 in op_sessions(times, g0, g1):
        pieces += [["dead", head, s0], ["op", s0, s1]]
        head = s1
    pieces.append(["dead", head, g1])
    merged = []
    for kind, s0, s1 in pieces:
        if s1 - s0 <= 0:
            continue
        if kind == "dead" and s1 - s0 < ROW_MIN_S and merged:
            merged[-1][2] = s1          # too short to be a row: fold it away
            continue
        merged.append([kind, s0, s1])
    if len(merged) > 1 and merged[0][0] == "dead" and merged[0][2] - merged[0][1] < ROW_MIN_S:
        merged[1][1] = merged[0][1]
        merged.pop(0)
    rows = []
    for kind, s0, s1 in merged:
        d = s1 - s0
        if kind == "op":
            rows.append((f"操作演示（无讲解 {dur_txt(d)}）",
                         f"{hhmm(s0)}–{hhmm(s1)} 画面持续变化，全程无讲解", s0, s1))
        else:
            rows.append((f"休息（无内容 {dur_txt(d)}）",
                         f"{hhmm(s0)}–{hhmm(s1)} 全程无讲解，画面无变化", s0, s1))
    return rows


def split_silences(chapters, sents, dur, video, cache):
    """Cut long silences out as their own rows, split by what the screen did.

    A break shows up only in the transcript: one 788 s "chapter" held 72 s of
    speech. Inside such a span the picture says whether anything happened, and
    that separates empty time from an un-narrated demo:
      no content change  -> 「休息（无内容 …）」
      a session of them  -> 「操作演示（无讲解 …）」   (this IS content)
    """
    spans = []
    for i, c in enumerate(chapters):
        nxt = chapters[i + 1]["start"] if i + 1 < len(chapters) else dur
        spans.append((c, c["start"], nxt))

    out = []
    for c, a, b in spans:
        gaps = []
        for s0, s1 in zip(sents, sents[1:]):
            g0, g1 = s0["end"] / 1000.0, s1["start"] / 1000.0
            if g1 - g0 < SILENCE_S:
                continue
            lo, hi = max(g0, a), min(g1, b)
            if hi - lo < SILENCE_S:
                continue
            gaps.append((lo, hi))
        if not gaps:
            out.append({**c, "start": round(a, 2), "end": round(b, 2)})
            continue
        head = a
        for g0, g1 in gaps:
            # A break usually ends where the next chapter begins, so the tail is
            # empty: let the silence rows absorb it instead of skipping the split.
            tail_end = b
            if b - g1 < 20:
                tail_end = g1
            if g0 - head >= 20:
                out.append({**c, "start": round(head, 2), "end": round(g0, 2)})
            for title, summary, s0, s1 in quiet_rows(g0, tail_end, video, cache):
                out.append({"start": round(s0, 2), "end": round(s1, 2), "jump": round(s0, 2),
                            "source": "silence", "title": title, "summary": summary,
                            "offset": 0.0, "unit": c.get("unit"),
                            "unit_mode": c.get("unit_mode"), "dur_estimate": 0.0})
            head = tail_end
            if tail_end >= b:
                break
        if b - head >= 20:
            out.append({**c, "start": round(head, 2), "end": round(b, 2),
                        "jump": round(max(c["jump"], head), 2) if head > a else c["jump"]})
    out.sort(key=lambda c: c["start"])
    return out


def cont(title):
    """Continuation marker, never stacked twice ("（续）（续）").

    A numbered marker counts as already marked: appending to "…（续5）" produced
    "…（续5）（续）" in a delivered table of contents.
    """

    def tidy(t):
        t = str(t).strip()
        while t.endswith(SUFFIX * 2):
            t = t[: -len(SUFFIX)]
        return t

    t = tidy(title)
    return t if CONT_TAIL.search(t) else t + SUFFIX


def _flat(s):
    return "".join(ch for ch in s if not ch.isspace())


ANCHOR_MIN = 0.62


def resolve_start(c, sents, floor=0):
    """Which sentence does this chapter actually start on?

    The table stores `start_sentence`, but that number is only valid in the run
    that produced it. When the same table is applied to a run whose ASR split
    sentences differently, every number after the first moved split points at
    the wrong place — measured across a 92-minute lecture, chapter starts slid
    by up to 47 seconds toward the end while the titles stayed right.

    `start_text` (written since this change, or backfilled) is the sentence the
    chapter opened on; matching that text finds the boundary again. The number
    remains the fallback so old tables keep working.
    """
    a = _flat(c.get("start_text") or "")
    if a:
        best, idx = 0.0, None
        for i in range(floor, len(sents)):
            r = difflib.SequenceMatcher(None, a, _flat(sents[i]["text"])).ratio()
            if i + 1 < len(sents):
                # a thin opening sentence is stored together with the next one
                pair = _flat(sents[i]["text"]) + _flat(sents[i + 1]["text"])
                r = max(r, difflib.SequenceMatcher(None, a, pair).ratio())
            if r > best:
                best, idx = r, i
        if idx is not None and best >= ANCHOR_MIN:
            return idx
    return max(1, int(c["start_sentence"])) - 1


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "sample"
    outdir = str(_run(run_id).root)
    sents = json.load(open(str(_run(run_id).asr_sent)))["items"]
    llm = json.load(open(str(_run(run_id).chapters_llm)))["chapters"]
    st = json.load(open(f"{outdir}/states.json"))
    units = st["states"]
    dur = st["duration"]
    chrome, mode_chrome, acronyms = chrome_and_acronyms(units)

    t_of = [s["start"] / 1000.0 for s in sents]
    ends = [s["end"] / 1000.0 for s in sents]

    # speech gaps longer than GAP_S
    gaps = []
    for i in range(1, len(sents)):
        g = t_of[i] - ends[i - 1]
        if g > GAP_S:
            gaps.append((ends[i - 1], t_of[i]))

    onsets = [u.get("onset", u["start"]) for u in units]
    jumps = [u.get("jump", u["start"]) for u in units]

    def snap(t):
        """Nearest visual onset inside an asymmetric window, else None."""
        if not onsets:
            return None
        i = bisect.bisect_left(onsets, t)
        cand = [k for k in (i - 1, i, i + 1) if 0 <= k < len(onsets)]
        if not cand:
            return None
        best = min(cand, key=lambda k: abs(onsets[k] - t))
        d = onsets[best] - t
        if -SNAP_BACK <= d <= SNAP_FWD:
            return best
        return None

    chapters = []
    prev = 0
    for c in llm:
        s_idx = resolve_start(c, sents, prev)
        prev = s_idx
        t_text = t_of[s_idx]
        k = snap(t_text)
        if k is None:
            chapters.append({"start": round(t_text, 2), "jump": round(t_text + 0.3, 2),
                             "source": "text", "title": c["title"],
                             "summary": c["summary"], "offset": 0.0})
        else:
            chapters.append({"start": round(onsets[k], 2), "jump": round(jumps[k], 2),
                             "source": "snapped", "title": c["title"],
                             "summary": c["summary"],
                             "offset": round(onsets[k] - t_text, 2),
                             "unit": k, "unit_mode": units[k]["mode"]})

    def set_ends(chs):
        chs.sort(key=lambda c: c["start"])
        for i, c in enumerate(chs):
            c["end"] = round(chs[i + 1]["start"] if i + 1 < len(chs) else dur, 2)
            c["dur_estimate"] = round(c["end"] - c["start"], 2)
        return chs

    chapters = set_ends(chapters)

    # first chapter always starts at 0
    if chapters and chapters[0]["start"] > 0:
        chapters[0]["start"] = 0.0
        chapters[0]["jump"] = min(chapters[0]["jump"], 1.0)

    # a titled unit within a few screens of a flip names it better than the LLM can
    def unit_title(i, lookahead=4):
        for k in range(i, min(i + lookahead, len(units))):
            if units[k].get("title_ok") and units[k].get("title"):
                cand = units[k]["title"]
                # A unit titled after the product header or a menu label names
                # nothing; treat it as untitled so the caller falls back to the
                # host chapter's title (or a neutral label) instead.
                if verdict(cand, chrome_for(units[k].get("mode"), chrome, mode_chrome), acronyms) is not None:
                    continue
                return cand, jumps[k]
        return None, None

    used = {c.get("unit") for c in chapters}
    rescued = []
    for i, u in enumerate(units):
        if i in used:
            continue
        t = onsets[i]
        in_gap = any(a <= t <= b for a, b in gaps)
        if not in_gap:
            continue
        host = next((c for c in chapters
                     if c["start"] < t < c.get("end", c["start"] + c.get("dur_estimate", 0))), None)
        if host is not None:
            # inside a chapter: this is the screen showing what the chapter talks
            # about. Keep the TOC honest — record it as a sub-boundary, and only
            # split when the chapter is over the coarse limit anyway.
            title, jmp = unit_title(i)
            if host["dur_estimate"] > MAX_CHAPTER_S:
                chapters.remove(host)
                chapters.append({**host, "end": round(t, 2), "dur_estimate": round(t - host["start"], 2)})
                chapters.append({"start": round(t, 2), "end": host["end"],
                                 "jump": round(jmp or jumps[i], 2),
                                 "source": "visual", "title": title or cont(host["title"]),
                                 "summary": "画面在此切换，原章过长被拆开", "offset": 0.0,
                                 "unit": i, "unit_mode": u["mode"], "dur_estimate": 0.0})
            else:
                subs = host.setdefault("sub", [])
                if not any(abs(t - x["start"]) < 6 for x in subs):
                    subs.append({"start": round(t, 2),
                                 "jump": round(jmp or jumps[i], 2),
                                 "title": title or ""})
            continue
        if t > (chapters[0]["start"] + 30 if chapters else 30):
            title, jmp = unit_title(i)
            rescued.append({"start": round(t, 2), "jump": round(jmp or jumps[i], 2),
                            "source": "visual",
                            "title": title or neutral_title(u["mode"], u["dur"]),
                            "title_kind": None if title else "neutral",
                            "summary": "讲师在此处静音操作，由画面变化切分",
                            "offset": 0.0, "unit": i, "unit_mode": u["mode"],
                            "dur_estimate": 0.0})
    chapters = set_ends(chapters + rescued)

    # a visual rescue that sits just before a titled chapter is the same topic:
    # the screen changed first, the lecture caught up seconds later. Keep the
    # screen time, take the title from the transcript.
    merged = []
    i = 0
    while i < len(chapters):
        c = chapters[i]
        nxt = chapters[i + 1] if i + 1 < len(chapters) else None
        if (c["source"] == "visual" and nxt and nxt["source"] != "visual"
                and 0 <= nxt["start"] - c["start"] <= 25):
            merged.append({"start": c["start"], "jump": c["jump"],
                           "source": "visual+text", "title": nxt["title"],
                           "summary": nxt["summary"], "offset": 0.0,
                           "unit": c.get("unit"), "unit_mode": c.get("unit_mode")})
            i += 2
            continue
        merged.append(c)
        i += 1
    chapters = merged

    # dedupe anything still closer than 20s, preferring the titled one
    dedup = []
    for c in chapters:
        if dedup and c["start"] - dedup[-1]["start"] < 20:
            if dedup[-1]["source"] == "visual" and c["source"] != "visual":
                dedup[-1] = c
            continue
        dedup.append(c)
    chapters = dedup

    cache_path = f"{outdir}/{ACT_CACHE}"
    try:
        cache = json.load(open(cache_path))["windows"]
    except Exception:
        cache = {}
    chapters = split_silences(chapters, sents, dur, st["video"], cache)
    json.dump({"video": st["video"], "windows": cache}, open(cache_path, "w"))

    for i, c in enumerate(chapters):
        c["id"] = i
        c["end"] = round(chapters[i + 1]["start"] if i + 1 < len(chapters) else dur, 2)
        c["dur"] = round(c["end"] - c["start"], 2)

    json.dump({"video": st["video"], "duration": dur, "source": "asr+visual fusion",
               "params": {"snap_back_s": SNAP_BACK, "snap_fwd_s": SNAP_FWD, "gap_s": GAP_S,
                          "silence_s": SILENCE_S, "big_frac": BIG_FRAC, "merge_s": MERGE_S,
                          "op_min_s": OP_MIN_S, "row_min_s": ROW_MIN_S,
                          "act_fps": ACT_FPS, "act_size": [ACT_W, ACT_H]},
               "chapters": chapters}, open(f"{outdir}/chapters_fused.json", "w"),
              ensure_ascii=False, indent=1)

    # --- A/B against the existing (human-checked) chapter list ---
    base = json.load(open(f"{outdir}/chapters.json"))["chapters"]
    bb = [b["start"] for b in base]
    dists = []
    for c in chapters:
        j = bisect.bisect_left(bb, c["start"])
        cand = [bb[k] for k in (j - 1, j) if 0 <= k < len(bb)]
        dists.append(min(abs(c["start"] - x) for x in cand) if cand else 0)
    covered = sum(1 for b in bb
                  if any(abs(c["start"] - b) <= 10 for c in chapters))

    def mmss(t):
        return f"{int(t // 60):02d}:{int(t % 60):02d}"

    print(f"fused chapters: {len(chapters)}   baseline: {len(base)}")
    src = {}
    for c in chapters:
        src[c["source"]] = src.get(c["source"], 0) + 1
    print("source: " + ", ".join(f"{k}={v}" for k, v in sorted(src.items())))
    dists_sorted = sorted(dists)
    print(f"boundary distance to nearest baseline: median {dists_sorted[len(dists)//2]:.1f}s, "
          f"p90 {dists_sorted[int(len(dists)*0.9)]:.1f}s, max {dists_sorted[-1]:.1f}s")
    print(f"baseline boundaries covered within 10s: {covered}/{len(bb)}")
    span = []
    for c in chapters:
        n = sum(1 for b in base if c["start"] <= b["start"] < c["end"])
        span.append(n)
    print("baseline chapters per fused chapter: " +
          ", ".join(str(x) for x in span) + f"  (total {sum(span)})")
    print()
    for c in chapters:
        note = {"snapped": f"snap {c.get('offset', 0):+.1f}s", "text": "text-only",
                "visual": "visual rescue",
                "visual+text": "screen first, title from text"}.get(c["source"], c["source"])
        print(f"{c['id']:>2} {mmss(c['start'])}-{mmss(c['end'])} {c['dur']:>5.0f}s "
              f"[{note:>14}] {c['title'][:32]}")


if __name__ == "__main__":
    main()
