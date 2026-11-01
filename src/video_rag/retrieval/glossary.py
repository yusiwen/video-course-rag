#!/usr/bin/env python3
"""Build the per-course term map: ASR mishearing -> the spelling on the slide.

The screen is a dictionary: it holds the real spelling (GitLab, JIRA, oauth2)
while the transcript holds what the recogniser heard ("gattlab", "Java"). For
latin terms an edit-distance match is reliable; Chinese homophones (VK vs Wiki)
need the manual entries already in the file, which are never overwritten.

Usage: scripts/glossary.py <run-id>
Writes: data/glossary/<run>.json  (merged; manual entries win)
"""
from ..paths import run as _run
import json
import re
import sys
from pathlib import Path

TERM = re.compile(r"[A-Za-z][A-Za-z0-9_.+#-]{2,}")
UI = {"powerpoint", "pptx", "excel", "docx", "http", "https", "com", "cn", "www"}


def lev(a, b, cap=4):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "sample"
    units = json.load(open(str(_run(run_id).states)))["states"]
    sents = json.load(open(str(_run(run_id).asr_sent)))["items"]

    screen_terms, screen_count = {}, {}
    for u in units:
        for ln in u.get("ocr_lines") or []:
            for t in TERM.findall(ln):
                low = t.lower().strip(".-_")
                if len(low) < 3 or low in UI:
                    continue
                screen_terms.setdefault(low, t)
                screen_count[low] = screen_count.get(low, 0) + 1

    asr_text = " ".join(s["text"] for s in sents)
    asr_terms = {t.lower() for t in TERM.findall(asr_text)}

    # Auto matches go to a .suggested file, never into the applied map: the
    # screen term itself can be OCR junk ("O5P" for ACME, "pEP" for PPT), and a
    # wrong mapping rewrites the transcript silently. Review, then copy.
    p = _run(run_id).root / "glossary.suggested.json"

    auto = {}
    for low, shown in sorted(screen_terms.items(), key=lambda kv: -screen_count[kv[0]]):
        if low in asr_terms:
            continue
        cands = [t for t in asr_terms
                 if t and t[0] == low[0] and abs(len(t) - len(low)) <= 2]
        if not cands:
            continue
        best = min(cands, key=lambda t: lev(t, low))
        d = lev(best, low)
        if 1 <= d <= max(2, len(low) // 3):
            auto[best] = shown          # asr spelling -> spelling on screen

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(auto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"{len(auto)} suggestion(s) -> {p}  (review, then copy into "
          f"data/glossary/{run_id}.json)")
    for k in sorted(auto):
        print(f"  {k!r} -> {auto[k]!r}")


if __name__ == "__main__":
    main()
