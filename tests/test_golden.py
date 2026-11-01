"""Invariants over the real indexed runs.

These are not golden numbers — chapter counts move whenever the title layer is
tuned, and pinning them would make every experiment a test failure. What must
never move is what a student sees: a delivered table of contents with real
titles, neutral rows that are labelled as measurements, silence rows that are
long enough to be worth a row, and an audit that passes.

Skips itself (returns "skip") when the artifacts are absent, so a fresh clone
can run the suite before indexing anything.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from video_rag.paths import all_runs, run as _run  # noqa: E402

BAD_TITLE = re.compile(r"^(未登录|分享|登录|PLEASE READ|扫码登录)$", re.I)
NEUTRAL = re.compile(r"^(幻灯片|浏览器操作|桌面操作|窗口操作|片段)（.+）$")
SILENCE = re.compile(r"^(休息（无内容|操作演示（无讲解)")
# The two kinds have different thresholds on purpose: a row of dead time needs
# 60 s to be worth a line (ROW_MIN_S), while an operation needs 30 s to be
# recognised as one at all (OP_MIN_S) — a measured 30-second silent
# demonstration is a real row, not a rounding error.


def runs():
    if not all_runs():
        return []
    out = []
    for d in [r.root for r in all_runs()]:
        if (d / "states.json").exists() and (d / "index" / "chapters.json").exists():
            out.append(d.name)
    return out


def test_delivered_toc_is_ordered_and_titled():
    names = runs()
    if not names:
        return "skip"
    for rid in names:
        st = json.load(open(_run(rid).states))
        ch = json.load(open(_run(rid).index_dir / "chapters.json"))["chapters"]
        assert ch, rid
        prev_end = -1.0
        last_title = None
        for c in ch:
            t = (c.get("title") or "").strip()
            assert t, f"{rid}: chapter without a title at {c['start']}"
            assert not BAD_TITLE.match(t), f"{rid}: shell text as a title: {t!r}"
            assert t != last_title, f"{rid}: two consecutive chapters named {t!r}"
            assert 0 <= c["start"] < c["end"] <= st["duration"] + 1, f"{rid}: {c['start']}-{c['end']}"
            assert c["start"] >= prev_end - 1, f"{rid}: overlapping chapters at {c['start']}"
            prev_end, last_title = c["end"], t


def test_no_chrome_reached_the_delivered_toc():
    names = runs()
    if not names:
        return "skip"
    sys.path.insert(0, str(ROOT / "scripts"))
    from video_rag.title.gate import chrome_and_acronyms, chrome_for, norm
    for rid in names:
        prof_path = ROOT / "data" / "profiles" / f"{rid}.json"
        if not prof_path.exists():
            continue
        st = json.load(open(_run(rid).states))
        chrome, mode_chrome, _ = chrome_and_acronyms(st["states"])
        for c in json.load(open(_run(rid).index_dir / "chapters.json"))["chapters"]:
            base = c["title"].split(" · ")[0]
            assert norm(base) not in chrome_for(c.get("mode"), chrome, mode_chrome), \
                f"{rid}: chrome as a title: {c['title']!r}"


def test_neutral_rows_are_labelled_as_measurements():
    names = runs()
    if not names:
        return "skip"
    for rid in names:
        p = _run(rid).chapters
        if not p.exists():
            continue
        for c in json.load(open(p))["chapters"]:
            if c.get("title_kind") == "neutral":
                assert NEUTRAL.match(c["title"]), f"{rid}: {c['title']!r}"
                assert c["title"].strip() != " · ", rid


def test_silence_rows_are_not_momentary():
    names = runs()
    if not names:
        return "skip"
    for rid in names:
        p = _run(rid).index_dir / "chapters.json"
        for c in json.load(open(p))["chapters"]:
            if SILENCE.match(c["title"]):
                span = c["end"] - c["start"]
                floor = 60 if c["title"].startswith("休息") else 30
                assert span >= floor, f"{rid}: {c['title']} lasts only {span:.0f}s"


def test_audit_is_strict_clean():
    names = runs()
    if not names:
        return "skip"
    for rid in names:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        r = subprocess.run([sys.executable, "-m", "video_rag.title.audit", "--strict", rid],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"{rid}: EXPO in the delivered TOC\n{r.stdout[-800:]}"


def test_silence_rows_carry_their_measurement():
    """The caption inside a dead-time chapter shows this text.

    Without it the page can only repeat the previous slide's caption, which is
    what it did before: the row said 「休息」 while the caption described a slide
    from ten minutes earlier.
    """
    names = runs()
    if not names:
        return "skip"
    for rid in names:
        for c in json.load(open(_run(rid).index_dir / "chapters.json"))["chapters"]:
            if SILENCE.match(c["title"]):
                assert (c.get("summary") or "").strip(), f"{rid}: {c['title']} has no summary"
