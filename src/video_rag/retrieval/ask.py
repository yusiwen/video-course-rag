#!/usr/bin/env python3
"""Ask a question against an indexed course; get an answer plus a timecode.

Retrieval is hybrid (BM25 + bge-m3, RRF) from search.py, then chunks are
expanded to their neighbours inside the same chapter so the model sees a
continuous stretch instead of three orphan sentences.

The model is given candidates labelled C1..Cn and may only cite those labels;
timestamps are attached afterwards from the index. That is the same rule as
the chaptering step, for the same reason: an LLM that writes "03:12" is
inventing it, and a timestamp nobody can check is worse than no answer.

Usage:
  scripts/ask.py "第三章讲了什么" [--run lecture-1] [--open] [--json]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import video_rag.retrieval.search as S  # noqa: E402  (BM25, embed_one, load, mmss)

from video_rag.chapter.llm_chapters import load_dotenv  # noqa: E402

PROMPT = """你是这门课的助教。下面是从课程录像里检索到的片段，编号 C1…Cn，每段带章节名和片段内容（[ASR] 是讲师原话，[屏幕] 是画面上的文字）。

请只用这些片段回答学生的问题。

规则：
1. 只输出 JSON，不要输出任何时间数字（分钟、秒、时间戳都不行）。
2. `cites` 里只能出现上面给过的片段编号（如 "C3"），不允许编造编号。
3. 如果片段不足以回答，`answer` 留空、`cites` 留空、`insufficient` 设 true——宁可说不知道。
4. `answer` 用中文，一到三句话，直接回答问题本身，不要复述问题。
5. 回答要完整：片段里凡是与问题相关的办理途径、前提条件（VPN、权限、迁移）、注意点，
   都要写进回答并各自引用，不要只挑看起来最像的一条。

输出格式：
{{"answer": "...", "cites": ["C1"], "insufficient": false}}

问题：{question}

片段：
{context}
"""


def build_context(chunks, meta, chapters, top):
    """Top hits + the neighbours of each hit inside the same chapter."""
    by_chap = {}
    for i, c in enumerate(chunks):
        by_chap.setdefault(c["chapter_id"], []).append(i)
    keep = set()
    for i in top:
        cid = chunks[i]["chapter_id"]
        order = by_chap.get(cid, [])
        if i in order:
            p = order.index(i)
            keep.update(order[max(0, p - 1):p + 2])
    keep.update(top)
    # one continuous block per chapter, in timeline order
    keep = sorted(keep, key=lambda i: chunks[i]["start"])
    return [(k, chunks[k]) for k in keep]


def ask(question, run, k=6):
    chunks, meta, chapters, top, _ = S.retrieve(run, question, k)
    ctx = build_context(chunks, meta, chapters, top)

    lines = []
    for n, (i, c) in enumerate(ctx, 1):
        body = c["text"].replace("\n", " ")
        lines.append(f"C{n}｜{c['chapter_title']}｜{body[:400]}")
    context = "\n".join(lines)
    by_label = {f"C{n}": chunks[i] for n, (i, _) in enumerate(ctx, 1)}

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set (see PLAN.md §0)")

    body = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [{"role": "user",
                      "content": PROMPT.format(question=question, context=context)}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    import urllib.request
    req = urllib.request.Request("https://api.deepseek.com/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    raw = d["choices"][0]["message"]["content"]
    usage = d.get("usage") or {}

    try:
        ans = json.loads(raw)
    except json.JSONDecodeError:
        ans = {"answer": raw.strip(), "cites": [], "insufficient": False}

    cites = [c for c in (ans.get("cites") or []) if c in by_label]
    dropped = [c for c in (ans.get("cites") or []) if c not in by_label]
    return {"question": question, "run": run,
            "answer": (ans.get("answer") or "").strip(),
            "cites": cites, "dropped_cites": dropped,
            "insufficient": bool(ans.get("insufficient")),
            "context": ctx, "by_label": by_label, "usage": usage,
            "chapters": chapters, "meta": meta}


def render(res):
    print(f"Q: {res['question']}\n")
    if res["insufficient"] or not res["answer"]:
        print("（检索到的片段不足以回答这个问题）")
    else:
        print(res["answer"])
    print()
    seen = set()
    for lbl in res["cites"]:
        c = res["by_label"][lbl]
        if c["chapter_id"] in seen:
            continue
        seen.add(c["chapter_id"])
        chap = next((x for x in res["chapters"] if x["id"] == c["chapter_id"]), None)
        rng = f"{S.mmss(chap['start'])}–{S.mmss(chap['end'])}" if chap else ""
        print(f"  ▸ {S.mmss(c['jump'])}  {c['chapter_title']}  [{c['mode']}]  （本章 {rng}）")
    if res["dropped_cites"]:
        print(f"  ⚠ 丢弃了索引里不存在的引用: {res['dropped_cites']}")
    u = res["usage"]
    if u:
        print(f"\n  tokens: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--run", default="sample")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--open", action="store_true", help="open mpv at the first citation")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    load_dotenv()
    res = ask(a.question, a.run, a.k)

    if a.json:
        out = {k: v for k, v in res.items() if k not in ("context", "by_label", "chapters")}
        out["cites"] = [{"label": l, "jump": res["by_label"][l]["jump"],
                         "chapter": res["by_label"][l]["chapter_title"]}
                        for l in res["cites"]]
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        render(res)

    if a.open and res["cites"]:
        c = res["by_label"][res["cites"][0]]
        video = res["meta"]["video"]
        print(f"\nlaunching mpv at {S.mmss(c['jump'])}")
        subprocess.run(["mpv", f"--start={c['jump']}", "--pause", video])


if __name__ == "__main__":
    main()
