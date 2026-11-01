#!/usr/bin/env python3
"""Objective audit of the title layer — the尺子 for retitle.py.

The title picker is a heuristic (largest font, highest, inside a per-mode
region, minus a blocklist). When it locks onto shell text nothing throws: the
only symptom is a table of contents made of "未登录", "分享", a deck file name,
or the product's own header. Every failure mode seen so far is measurable from
the run's own artifacts, so measure it:

  persistence   a line present in a large share of ALL units is shell or
                watermark; a slide title turns over every slide.
  exposure      a bad title only hurts if it ships. Chapter titles usually come
                from the transcript, so unit-level garbage stays invisible —
                until a material has no usable speech (a frozen recording, a
                silent demo), and then it IS the table of contents.
  repetition    the same title on >=3 consecutive chapters means the picker
                froze on something persistent (genuine repeats get 「（续）」).
  deck names    a title carrying a file name is a window title bar, not a slide.

Two numbers matter, and they are reported separately:
  EXPO   bad titles present in the shipped table of contents  -> blocks delivery
  RISK   bad unit titles that would ship if the transcript
         chapter step were unavailable                        -> degrade risk

Usage:
    scripts/title_audit.py                 # every run under out/runs/
    scripts/title_audit.py lecture-1 lecture-2   # named runs
    scripts/title_audit.py --strict        # exit 1 when any EXPO error exists

Writes out/runs/<id>/title_audit.md, prints one line per run.
"""
from ..paths import run as _run, runs_root
import json
import os
import sys
from collections import Counter
from pathlib import Path

from video_rag.title.gate import (NEUTRAL_RE, chrome_and_acronyms, chrome_for, suspect,  # noqa: E402
                       verdict)

MISSING_SHARE_ERROR = 0.30   # per mode: this much title_ok=false means a bad region
NEUTRAL_SHARE_MAX = 0.20     # shipped TOC: this many «幻灯片（N 分钟）» rows means
                             # the transcript chaptering never ran


def hhmm(t):
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def load(run_id):
    st = json.load(open(str(_run(run_id).states)))
    fused = json.load(open(str(_run(run_id).chapters_fused)))
    shipped = None
    p = _run(run_id).index_dir / "chapters.json"
    if p.exists():
        shipped = json.load(open(p))["chapters"]
    return st, fused, shipped


def audit(run_id):
    st, fused, shipped = load(run_id)
    units = st["states"]
    chapters = fused["chapters"]
    n = max(1, len(units))

    chrome, mode_chrome, acronyms = chrome_and_acronyms(units)
    line_count = Counter()
    for u in units:
        for line in set(u.get("ocr_lines") or []):
            line_count[line.strip()] += 1

    expo, risk, infos, blind = [], [], [], []

    def rec(kind, msg, where, level, bucket):
        bucket.append({"level": level, "kind": kind, "msg": msg, "where": where})

    # ---- unit titles: what the picker produced
    per_mode = {}
    for u in units:
        m = u.get("mode", "?")
        d = per_mode.setdefault(m, {"n": 0, "bad": 0})
        d["n"] += 1
        d["bad"] += 0 if u.get("title_ok") else 1
    # A mode without a title band is not a bug: the calibration measured that the
    # strip which would hold a title does not behave like one on this material
    # (an application window, an Explorer window, a desktop with the menu bar
    # where a deck would put its title). Saying "the region was derived wrong"
    # sends the reader to fix a decision that was already measured — and it fired
    # on every application course.
    prof = {}
    try:
        prof = json.load(open(_run(run_id).profile))
    except Exception:
        pass
    for m, d in sorted(per_mode.items()):
        if d["n"] >= 5 and d["bad"] / d["n"] > MISSING_SHARE_ERROR:
            info = (prof.get("modes") or {}).get(m) or {}
            band = info.get("band")
            share = f"{d['bad']}/{d['n']}（{d['bad'] / d['n'] * 100:.0f}%）"
            if not band:
                rec("bandless_mode",
                    f"模式 {m} 有 {share} 个单元没有可用标题——标定判定该模式没有标题行"
                    f"（{info.get('frames', '?')} 帧采样），中性标签就是预期结果，"
                    f"不是区域推导错",
                    f"mode={m}", "INFO", infos)
            else:
                y = band.get("yield")
                rec("mode_region",
                    f"模式 {m} 有 {share} 个单元没有可用标题；标定出的带 y={band['y']:.2f}"
                    f" 只在 {y * 100:.0f}% 的采样帧里产出标题"
                    f"——抽查画面里这条带常常没有文字（资源管理器/应用窗口），"
                    f"未必是带选错了",
                    f"mode={m}", "INFO", infos)
    for u in units:
        t = (u.get("title") or "").strip()
        if not t:
            continue          # "no title at all" is the title_ok path, not a bad title
        c = chrome_for(u.get("mode"), chrome, mode_chrome)
        v = verdict(t, c, acronyms)
        if v:
            rec(v[0], f"{v[1]}：{t!r}", f"{hhmm(u['start'])} mode={u.get('mode')}", "RISK", risk)
        else:
            sp = suspect(t)
            if sp:
                rec(sp[0], f"{sp[1]}：{t!r}", f"{hhmm(u['start'])}", "BLIND", blind)
    if not any(u.get("title_ok") for u in units):
        rec("no_titles", "整个运行没有一个单元拿到可用标题", "", "ERROR", infos)

    # ---- what actually ships
    if shipped is None:
        rec("no_index", "index/<run>/chapters.json 不存在，无法判断暴露面", "", "INFO", infos)
        shipped = []
    for c in shipped:
        t = (c.get("title") or "").strip()
        v = verdict(t, chrome_for(c.get("mode"), chrome, mode_chrome), acronyms)
        if v:
            rec(v[0], f"交付目录里的坏标题，{v[1]}：{t!r}",
                f"{hhmm(c['start'])}", "ERROR", expo)
        if "（续" in t and "（续2" not in t and t.count("（续") > 1:
            rec("stacked_suffix", f"续号叠加：{t!r}", hhmm(c["start"]), "ERROR", expo)

    # A neutral label is the honest answer for a chapter nobody could name, but a
    # table of contents made mostly of «幻灯片（3 分钟）» means the transcript
    # chaptering never ran — the student would open the package and find no topics.
    # Silence rows (休息/操作演示) are measured facts, not neutral rows, and are
    # counted elsewhere.
    neutral_rows = [c for c in shipped if NEUTRAL_RE.match((c.get("title") or "").strip())]
    if shipped and len(neutral_rows) / len(shipped) > NEUTRAL_SHARE_MAX:
        rec("neutral_toc",
            f"交付目录里 {len(neutral_rows)}/{len(shipped)} 章是中性标签"
            f"（{len(neutral_rows) / len(shipped):.0%}，上限 {NEUTRAL_SHARE_MAX:.0%}），"
            f"例如「{neutral_rows[0]['title']}」——字幕切章没有生效，"
            f"学生看到的是时长而不是主题",
            "index/<run>/chapters.json", "ERROR", expo)

    # ---- repetition inside the delivered list
    # Neutral labels are skipped as well as the measured silence rows: a degraded
    # table of contents repeats them, and sixteen "the selector froze" errors bury
    # the one real cause (the neutral_toc error raised further down).
    content = [c for c in shipped
               if not (c.get("title") or "").startswith(("休息（", "操作演示（"))
               and not NEUTRAL_RE.match((c.get("title") or "").strip())]
    run_len = 1
    for prev, cur in zip(content, content[1:]):
        same = cur["title"] == prev["title"] and "（续" not in cur["title"]
        run_len = run_len + 1 if same else 1
        if run_len >= 3:
            rec("title_frozen",
                f"「{cur['title']}」连续出现在 {run_len}+ 章，选择器锁在了不会变的东西上",
                hhmm(cur["start"]), "ERROR", expo)
    counts = Counter(c["title"] for c in content)
    for t, k in counts.most_common(2):
        if k >= 5 and k / max(1, len(content)) > 0.15:
            rec("title_dominant", f"「{t}」占了 {k}/{len(content)} 章", "", "WARN", infos)

    # ---- how much of the TOC owes its titles to the transcript
    from_text = sum(1 for c in fused["chapters"] if c.get("source") in ("text", "snapped", "visual+text"))
    rec("transcript_titled",
        f"{from_text}/{len(fused['chapters'])} 章的标题来自字幕（LLM）；"
        f"这些章节在字幕不可用时会退回单元标题",
        "", "INFO", infos)

    # ---- the degraded path: what the TOC looks like when no transcript titles
    # exist. chapters.py builds chapters from visual units, so a course whose
    # speech is missing (or a frozen recording) ships exactly these titles.
    degraded = {"total": 0, "bad": 0, "neutral": 0, "suspect": 0, "worst": [], "share": 0.0}
    dp = _run(run_id).chapters
    if dp.exists():
        units_ch = json.load(open(dp))["chapters"]
        for c in units_ch:
            t = (c.get("title") or "").strip()
            degraded["total"] += 1
            if c.get("title_kind") == "neutral":
                degraded["neutral"] += 1      # deliberate: describe, don't invent
                continue
            v = verdict(t, chrome_for(c.get("mode"), chrome, mode_chrome), acronyms)
            if v:
                degraded["bad"] += 1
            else:
                sp = suspect(t)
                if sp:
                    degraded["suspect"] = degraded.get("suspect", 0) + 1
        degraded["share"] = round(degraded["bad"] / max(1, degraded["total"]), 3)
        cnt = Counter((c.get("title") or "") for c in units_ch)
        degraded["worst"] = cnt.most_common(3)
        if degraded["share"] > 0.10:
            rec("degraded_titles",
                f"若字幕章节不可用，单元派生目录里 {degraded['bad']}/{degraded['total']} 章"
                f"（{degraded['share']:.0%}）拿不到可用标题，最高频是 "
                f"「{degraded['worst'][0][0][:24]}」×{degraded['worst'][0][1]}",
                "", "RISK", risk)

    return {"run_id": run_id, "units": n, "chapters": len(fused["chapters"]),
            "shipped": len(shipped), "chrome_lines": len(chrome),
            "expo": expo, "risk": risk, "infos": infos, "blind": blind,
            "degraded": degraded,
            "mode_share": {m: round(d["n"] / n, 3) for m, d in per_mode.items()}}


def write_report(r):
    p = _run(r["run_id"]).report_dir / "title_audit.md"
    L = [f"# 标题体检 · {r['run_id']}", "",
         f"- 单元 {r['units']}，融合章节 {r['chapters']}，交付章节 {r['shipped']}",
         f"- 高持续率文本（判为外壳）{r['chrome_lines']} 条",
         "- 模式占比 " + ", ".join(f"{k}={v:.0%}" for k, v in sorted(r["mode_share"].items())),
         f"- 退化路径（单元派生目录）{r['degraded']['bad']}/{r['degraded']['total']} 章标题不可用"
         f"（{r['degraded']['share']:.0%}），其中 {r['degraded'].get('neutral', 0)} 章改用中性标题",
         f"- 退化目录里疑似垃圾（规则判不了，标定目标）{r['degraded'].get('suspect', 0)} 章",
         f"- **EXPO {len(r['expo'])}（交付目录污染）/ RISK {len(r['risk'])}（退化风险）"
         f" / BLIND {len(r['blind'])}（判不了的疑似垃圾）**", ""]
    blocks = [("EXPO — 已进入交付目录，必须修", r["expo"]),
              ("RISK — 单元标题坏了，字幕不可用时会暴露", r["risk"]),
              ("BLIND — 现有规则判不了，交给标定阶段", r.get("blind", [])),
              ("INFO", r["infos"])]
    for head, items in blocks:
        L.append(f"## {head}（{len(items)}）")
        L.append("")
        if not items:
            L += ["- 无", ""]
            continue
        kinds = Counter(f["kind"] for f in items)
        L.append("- 分类：" + ", ".join(f"{k}×{v}" for k, v in kinds.most_common()))
        L.append("")
        seen = set()
        for f in items:
            key = (f["kind"], f["msg"])
            if key in seen:
                continue
            seen.add(key)
            where = f" `{f['where']}`" if f["where"] else ""
            L.append(f"- **{f['kind']}**{where} {f['msg']}")
            if len(seen) >= 25:
                L.append(f"- …同类共 {len(items)} 条")
                break
        L.append("")
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    if not args:
        args = sorted(p.name for p in runs_root().iterdir()
                      if (p / "chapters_fused.json").exists())
    expo_total = 0
    for rid in args:
        r = audit(rid)
        p = write_report(r)
        expo_total += len(r["expo"])
        print(f"{rid:<9} EXPO {len(r['expo']):>3}  RISK {len(r['risk']):>3}  "
              f"BLIND {len(r['blind']):>3}   {r['shipped']:>3} 章  -> {p}")
    if strict and expo_total:
        print(f"strict: 交付目录里有 {expo_total} 处坏标题")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
