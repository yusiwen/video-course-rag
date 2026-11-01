#!/usr/bin/env python3
"""Rebuild sentence-level segments from FunASR token timestamps + punctuation.

FunASR's paraformer-zh here returns `text` (punctuated by ct-punc) and
`timestamp` (per-token [start_ms, end_ms]) but NO sentence_info. The token list
aligns 1:1 with the un-punctuated text, so sentences are rebuilt by walking both.

Two cut rules, because punctuation alone is not enough:
  * punctuation  。？！； ends a sentence
  * a silence gap between consecutive tokens longer than GAP_MS also ends one
    — otherwise a 9-character "嗯大家先签一下到吧" inherits the whole 94-second
    stretch it was spoken in, which ruins both the subtitle and any timestamp
    derived from it

A sentence still longer than MAX_S gets split at its largest internal gap, or
by interpolation if it has no gap at all.
"""
from ..paths import run as _run, runs_root, mkdirs
import json
import sys

END_PUNCT = "。？！；"
ALL_PUNCT = "，。？！；：、…—\"'（）()《》 \n"
GAP_MS = 1200
MAX_S = 25.0


def fmt_ts(ms):
    h, rem = divmod(int(round(ms)), 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def tokenize(text, ts):
    """(char, start, end) for every character that has a timestamp."""
    toks, ti = [], 0
    for ch in text:
        if ch in ALL_PUNCT:
            continue
        if ti >= len(ts):
            break
        s, e = ts[ti]
        ti += 1
        toks.append((ch, float(s), float(e)))
    return toks


def split_sentences(text, ts):
    toks = tokenize(text, ts)
    # mark sentence ends on punctuation *in the original text order*
    ends = set()
    ti = 0
    for i, ch in enumerate(text):
        if ch in ALL_PUNCT:
            if ch in END_PUNCT and ti > 0:
                ends.add(ti - 1)
            continue
        ti += 1

    out, cur = [], []
    for i, (ch, s, e) in enumerate(toks):
        if cur:
            gap = s - cur[-1][2]
            if gap > GAP_MS:
                out.append(cur)
                cur = []
        cur.append((ch, s, e))
        if i in ends:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)

    # a sentence can still be long if the speaker never paused: cut at the
    # biggest internal gap, or interpolate when there is none
    def emit(group):
        dur = (group[-1][2] - group[0][1]) / 1000.0
        if dur <= MAX_S or len(group) < 8:
            return [group]
        gaps = [(group[i + 1][1] - group[i][2], i + 1) for i in range(len(group) - 1)]
        gap, at = max(gaps)
        if gap > 400:
            return emit(group[:at]) + emit(group[at:])
        half = len(group) // 2
        return [group[:half], group[half:]]

    flat = [g for grp in out for g in emit(grp)]
    return [{"start": round(g[0][1], 0), "end": round(g[-1][2], 0),
             "text": "".join(c for c, _, _ in g)} for g in flat if g]


def main(path=None, out_prefix=None):
    if path is None:
        rid = sys.argv[1] if len(sys.argv) > 1 else "sample"
        run = _run(rid)
        path, out_prefix = str(run.asr_raw), str(run.root / "asr_sent")
    d = json.load(open(path))[0]
    out = split_sentences(d["text"], d["timestamp"])

    json.dump({"engine": "funasr/paraformer-zh (sentence rebuilt, gap-aware)",
               "items": out},
              open(f"{out_prefix}.json", "w"), ensure_ascii=False, indent=1)

    with open(f"{out_prefix}.srt", "w", encoding="utf-8") as f:
        for i, it in enumerate(out, 1):
            f.write(f"{i}\n{fmt_ts(it['start'])} --> {fmt_ts(it['end'])}\n{it['text']}\n\n")

    durs = sorted((i["end"] - i["start"]) / 1000 for i in out)
    cov = sum(durs)
    print(f"{len(out)} sentences, {sum(len(i['text']) for i in out)} chars, "
          f"speech {out[0]['start']/1000:.1f}s -> {out[-1]['end']/1000:.1f}s, "
          f"coverage {cov:.0f}s")
    print(f"duration: median {durs[len(durs)//2]:.1f}s  p90 {durs[int(len(durs)*0.9)]:.1f}s  "
          f"max {durs[-1]:.1f}s  (>{MAX_S:.0f}s: {sum(1 for x in durs if x > MAX_S)})")


if __name__ == "__main__":
    main(*sys.argv[1:])
