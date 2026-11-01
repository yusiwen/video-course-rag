#!/usr/bin/env python3
"""Measure the ask pipeline against hand-written questions.

A question counts as a hit when at least one cited jump time falls inside an
accepted window. Windows rather than chapter ids, because re-chaptering moves
ids around while the content stays where it was — an id-based expectation goes
stale the moment the chapter table is regenerated.

Usage: scripts/eval_ask.py [--run sample]
"""
import argparse
import json
import sys
from pathlib import Path

import ask as A  # noqa: E402
from video_rag.chapter.llm_chapters import load_dotenv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sample")
    ap.add_argument("--set", default=None)
    ap.add_argument("--show", action="store_true", help="print every answer")
    a = ap.parse_args()
    load_dotenv()

    path = a.set or f"data/eval/{a.run}.jsonl"
    cases = json.load(open(path, encoding="utf-8"))

    hits, rows, tokens = 0, [], 0
    for n, case in enumerate(cases, 1):
        try:
            res = A.ask(case["q"], a.run)
        except Exception as e:                       # keep going, record the failure
            rows.append((case["q"], "ERROR", str(e)[:60]))
            continue
        jumps = [res["by_label"][l]["jump"] for l in res["cites"]]
        ok = any(a <= j <= b for j in jumps for a, b in case["t"])
        got = [round(j) for j in jumps]
        hits += ok
        u = res["usage"] or {}
        tokens += (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        first = res["by_label"][res["cites"][0]] if res["cites"] else None
        want = ",".join(f"{a}-{b}" for a, b in case["t"])
        rows.append((case["q"],
                     "✓" if ok else "✗",
                     f"期望 {want} 引用 {got}"
                     + (f" | 首条 {first['chapter_title']}" if first else " | 无引用")))
        if a.show:
            print(f"\n[{n}] {case['q']}\n    {res['answer'][:160]}")

    print()
    for q, mark, detail in rows:
        print(f"{mark} {q:28} {detail}")
    print(f"\n命中 {hits}/{len(cases)}  tokens={tokens}")


if __name__ == "__main__":
    main()
