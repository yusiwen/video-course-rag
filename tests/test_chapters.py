"""The chapter collapse rules, exercised on synthetic runs.

chapters.py is where the title layer either stays honest or silently ships a
table of contents made of headers, so its three structural rules are pinned
here: an unnameable MODE becomes neutral segments, a FOLD may not build a
monster chapter, and a name whose OCR reading never repeats is a header.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(states, profile, run_id="case"):
    """Materialise a run in a temp dir and collapse it; return the chapters."""
    with tempfile.TemporaryDirectory() as tmp:
        outdir = os.path.join(tmp, run_id)
        os.makedirs(outdir)
        json.dump({"video": "v.mp4", "duration": max(u["end"] for u in states),
                   "states": states}, open(os.path.join(outdir, "states.json"), "w"))
        json.dump(profile, open(os.path.join(outdir, "profile.json"), "w"))
        env = dict(os.environ, VIDEO_COURSE_RAG_RUNS=tmp, PYTHONPATH=str(ROOT / "src"))
        r = subprocess.run([sys.executable, "-m", "video_rag.chapter.table", "6.0", run_id],
                           cwd=tmp, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        return json.load(open(os.path.join(outdir, "chapters.json")))["chapters"]


def unit(i, start, end, mode, title="", ok=True):
    return {"id": i, "start": start, "end": end, "dur": end - start, "mode": mode,
            "title": title, "title_ok": ok, "title_score": 0.03,
            "ocr_lines": [title] if title else [], "thumb": f"thumbs/{i:03d}.png"}


def test_bandless_mode_becomes_neutral_segments():
    """A mode with no title band keeps its units; folding them glues hours together."""
    states = [unit(i, i * 60.0, (i + 1) * 60.0, "desktop") for i in range(20)]
    ch = run(states, {"modes": {"desktop": {"band": None}}})
    # 20 x 60 s = 20 min of unnameable screen: capped into ~6-minute segments,
    # never folded into one row and never dropped
    assert 3 <= len(ch) <= 5, f"{len(ch)} chapters"
    assert abs(sum(c["end"] - c["start"] for c in ch) - 1200) < 1
    for c in ch:
        assert c["title_kind"] == "neutral", c["title"]
        assert c["title"].startswith("桌面操作（"), c["title"]
        assert c["end"] - c["start"] <= 361, c["end"] - c["start"]


def test_fold_is_capped():
    """An untitled 400 s unit must not extend a 600 s chapter to 1000 s."""
    states = [
        unit(0, 0.0, 600.0, "ppt", "分级字典案例"),
        unit(1, 600.0, 1000.0, "ppt"),
        unit(2, 1000.0, 1600.0, "ppt", "单据模型定义"),
    ]
    ch = run(states, {"modes": {"ppt": {"band": {"y": 0.7, "y_range": [0.66, 0.74],
                                                "x_range": [0.1, 0.9]}}}})
    assert len(ch) == 3, [c["title"] for c in ch]
    assert ch[1]["title_kind"] == "neutral"
    assert ch[0]["end"] - ch[0]["start"] <= 600.1


def test_short_untitled_unit_still_folds():
    """The cap must not stop normal transition units from folding."""
    states = [
        unit(0, 0.0, 120.0, "ppt", "分级字典案例"),
        unit(1, 120.0, 125.0, "ppt"),
        unit(2, 125.0, 300.0, "ppt", "单据模型定义"),
    ]
    ch = run(states, {"modes": {"ppt": {"band": {"y": 0.7, "y_range": [0.66, 0.74],
                                                "x_range": [0.1, 0.9]}}}})
    assert len(ch) == 2, [c["title"] for c in ch]


def test_empty_titles_are_not_duplicates():
    """Empty titles once read as "the same title repeated" and got junk appended."""
    states = [unit(i, i * 120.0, (i + 1) * 120.0, "browser") for i in range(8)]
    states[0]["ocr_lines"] = ["石化-范家彬", "2023/3/8 10:22"]
    ch = run(states, {"modes": {"browser": {"band": None}}})
    for c in ch:
        assert " · " not in c["title"], c["title"]
        assert "石化" not in c["title"], c["title"]


def test_unstable_reading_is_a_header():
    """Four OCR readings of one name over 12 chapters is a header, not a topic."""
    names = ["ACME 平台名称", "SACME 平台名称", "3ACME 平台名称"]
    states = [unit(i, i * 120.0, (i + 1) * 120.0, "browser", names[i % 3])
              for i in range(12)]
    ch = run(states, {"modes": {"browser": {"band": {"y": 0.8, "y_range": [0.76, 0.84],
                                                    "x_range": [0.1, 0.9]}}}})
    neutral = [c for c in ch if c.get("title_kind") == "neutral"]
    assert len(neutral) >= 6, [c["title"] for c in ch]


def test_distinct_titles_on_one_theme_are_kept():
    """Three real chapters sharing a prefix must survive the variant rule."""
    names = ["数据管理-自定义数据模型-数据模型编辑",
             "数据管理-自定义数据模型-数据模型注册",
             "数据管理-自定义数据模型-数据校验规则"]
    states = [unit(i, i * 120.0, (i + 1) * 120.0, "browser", names[i % 3])
              for i in range(9)]
    ch = run(states, {"modes": {"browser": {"band": {"y": 0.8, "y_range": [0.76, 0.84],
                                                    "x_range": [0.1, 0.9]}}}})
    assert not any(c.get("title_kind") == "neutral" for c in ch), \
        [c["title"] for c in ch]
