#!/usr/bin/env python3
"""Build a hand-out package from one or more lecture videos.

The package is deliberately small: an index page and a README. The videos stay
where they are — the reader extracts the package into the folder that holds
them, opens the page, and picks that folder. The page binds files to courses by
content (size + sha1 of head||tail), so file names, extra files and several
courses in one folder all work.

    scripts/build_pkg.py "/path/to/recordings" --out dist/course --title "Course recordings"
    scripts/build_pkg.py 1.mp4 --out dist/course --reuse       # skip a finished run

Stages: run the pipeline per video (unless --reuse and the index is already
there) -> generate the page with --dist -> write README -> gate the result.
The gate is the point: a package that carries a key, an absolute path or a
missing video reference must never leave this script.
"""
from ..paths import run as _run, runs_root
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".mkv", ".webm"}

README = """\
{titles}
================

这个目录里有什么
----------------
  index.html   唯一入口，双击用浏览器打开（不要拖进 Word/预览）
  README.txt   本文件

没有视频文件。视频请放在本页所在的同一个目录（或任意位置），页面会按
文件内容自己找，不看文件名。

怎么用
------
1. 把 index.html 放到视频所在的目录里（和 .mp4 同级就行，子目录也可以）；
2. 双击 index.html，用 Chrome 或 Edge 打开（Safari 未验证）；
3. 点「选择视频目录…」，选中放视频的文件夹；
4. 看到「已匹配」就可以用了：
     - 左栏是章节 / 检索结果 / 转写，点一条就跳到那一分钟；
     - 搜索框是关键词检索（讲师原话 + 屏幕文字），完全本地，不需要网络；
     - j / k 上下章，/ 搜索，空格播放，c 开关字幕。

关于「问 AI」
-------------
默认关闭。要用提问，点左栏的「设置」，填自己的 API key（DeepSeek / OpenAI /
OpenRouter / 本地 Ollama 都可以），页面直接把问题和命中的字幕片段发给该服务商，
不经过任何中转。

  · key 只存在你自己浏览器的本地存储里，所有 file:// 页面共用同一份存储，
    同机的其它本地网页也读得到 —— 公用电脑请用完点「清除」。
  · 提问会把「你的问题 + 命中的几段字幕原文」发出去；不发送视频，也不发送
    整份字幕。
  · 回答里凡是带时间点的结论，都能点回原文；模型引用的片段如果不在本次
    检索结果里，那句会被标红，整段引用查不到时整条答案会被弃用 —— 这是
    故意的，宁可不答，也不给一个看着像真的的时间点。

这份索引覆盖的课程
------------------
{courses}

数据说明
--------
  · 章节与时间点来自 ASR 字幕 + 画面变化，不是人工标注，个别边界可能偏 1–2 秒；
  · 字幕由 FunASR 转写，专业术语会有错字，屏幕上的人名水印也可能被 OCR 带进来；
  · 检索只用于定位「哪一分钟讲过」，不能当作准确讲义。
"""


def slug(s):
    out = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return out or ""


def human(n):
    return f"{n/1048576:.0f} MB" if n >= 1048576 else f"{n/1024:.0f} KB"


def collect(src):
    p = Path(src)
    if p.is_file():
        return [p]
    if p.is_dir():
        # macOS leaves AppleDouble stubs ("._2.mp4", 4 KB, no moov atom) next to
        # real files on a non-HFS volume; they carry the same extension and would
        # be fed to ffmpeg as a video.
        return sorted([f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in VIDEO_EXT
                       and not f.name.startswith((".", "_"))],
                      key=lambda x: x.name)
    raise SystemExit(f"no such file or directory: {src}")


def run(cmd, what):
    print(f"  · {what}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"failed: {what} ({cmd[0]} exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="a video file, or a directory of videos "
                                          "(omit when packaging existing runs with --runs)")
    ap.add_argument("--runs", default="",
                    help="comma separated run ids to package from the runs root")
    ap.add_argument("--out", required=True, help="output directory, e.g. dist/course")
    ap.add_argument("--title", default="", help="package title used in the README")
    ap.add_argument("--max-mb", type=float, default=20.0,
                    help="warn above this package size (default 20 MB ≈ 85 courses)")
    ap.add_argument("--reuse", action="store_true",
                    help="keep index/<run-id>/ if it already exists (skip the pipeline)")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="never run the pipeline; only rebuild the page from existing indexes")
    a = ap.parse_args()

    if a.runs:
        ids = [x.strip() for x in a.runs.split(",") if x.strip()]
        for rid in ids:
            if not _run(rid).chunks.exists():
                raise SystemExit(f"run {rid} has no chunks.jsonl — index it first")
        print(f"[1/4] pipeline: reusing {len(ids)} run(s)")
    else:
        if not a.src:
            raise SystemExit("give a video/directory, or --runs to package existing runs")
        videos = collect(a.src)
        if not videos:
            raise SystemExit(f"no video files in {a.src}")

        print(f"[1/4] pipeline: {len(videos)} video(s)")
        catalog = json.load(open(runs_root() / "catalog.json", encoding="utf-8"))
        by_name = {Path(c["video"]).name: c["run_id"] for c in catalog["courses"]}
        ids = []
        for i, v in enumerate(videos):
            # reuse by name too: the run may already exist under a different id
            rid = by_name.get(v.name) or slug(v.stem) or f"v{i+1}"
            ids.append(rid)
            have = _run(rid).chunks.exists()
            if (a.reuse and have) or a.skip_pipeline:
                if not have:
                    raise SystemExit(f"run {rid} has no chunks.jsonl (index step skipped)")
                print(f"  · reuse {rid}")
                continue
            # The pipeline is the CLI's own stages; a package build is not the
            # place to shell out to a script that only existed in the old layout.
            for step in ("ingest", "title", "chapter", "index"):
                r = subprocess.run([sys.executable, "-m", "video_rag", step, str(v), rid], cwd=ROOT)
                if r.returncode != 0:
                    raise SystemExit(f"failed: {step} {v.name} -> {rid}")

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("[2/4] page (dist mode: no local video paths)")
    r = subprocess.run([sys.executable, "-m", "video_rag", "page", "--dist",
                        "--runs", ",".join(ids), "--out", str(out / "index.html")], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("failed: index.html")

    print("[3/4] README")
    cat = json.load(open(runs_root() / "catalog.json", encoding="utf-8"))
    titles = {c["run_id"]: c for c in cat["courses"]}
    rows, missing = [], []
    for rid in ids:
        e = titles.get(rid)
        if not e:
            missing.append(rid)
            continue
        mins = int(round(e["duration"] / 60))
        size = os.path.getsize(e["video"]) if Path(e["video"]).exists() else 0
        rows.append(f"  · {e['title']}\n    {Path(e['video']).name} · {mins} 分钟"
                    + (f" · {human(size)}" if size else ""))
    if missing:
        raise SystemExit(f"catalog has no entry for: {', '.join(missing)}")
    (out / "README.txt").write_text(
        README.format(titles=a.title or "课程录像索引", courses="\n".join(rows)),
        encoding="utf-8")

    print("[4/4] gate")
    html = (out / "index.html").read_text(encoding="utf-8")
    problems = []
    # A leaked key is a pattern, not a substring: the transcript can contain
    # anything (one sample chunk holds "C:/Users/86151/Desktop/...", which is
    # course content, not a leaked path).
    for m in re.finditer(r"sk-[A-Za-z0-9_\-]{16,}", html):
        problems.append(f"疑似 API key 进了页面：{m.group(0)[:8]}…")
    if "file:///" in html:
        problems.append("页面里有 file:/// 绝对地址")
    if "SVC_PROBE = false" not in html:
        problems.append("页面仍会探测本地开发服务（dist 模式应为 SVC_PROBE = false）")
    for m in re.finditer(r"""(?:src|href)\s*=\s*["'](/(?!/))""", html):
        problems.append(f"绝对路径属性：{m.group(0)[:40]}")
    mp4 = list(out.glob("*.mp4"))
    if mp4:
        problems.append(f"包里有视频：{[p.name for p in mp4]}")
    m = re.search(r"const DATA = (\[.*?\]);\n", html, re.S)
    blob = json.loads(m.group(1)) if m else []
    if len(blob) != len(ids):
        problems.append(f"页面里 {len(blob)} 门课，预期 {len(ids)}")
    for c in blob:
        if not c.get("sha1") or not c.get("bytes"):
            problems.append(f"{c.get('run_id')}: 缺指纹（构建时视频不在原路径？）")
        if c.get("src"):
            problems.append(f"{c.get('run_id')}: src 未清空（dist 模式必须为空）")

    # Title layer: a table of contents made of shell text is useless to a reader
    # and nothing else in this gate would notice. Only titles that are actually
    # shipped count (EXPO); unit-level garbage is reported but not fatal.
    audit = subprocess.run([sys.executable, "-m", "video_rag", "audit", "--strict", *ids],
                           cwd=ROOT, capture_output=True, text=True)
    if audit.returncode != 0:
        for line in (audit.stdout + audit.stderr).strip().splitlines():
            if "EXPO" in line or "strict" in line:
                problems.append(f"标题体检：{line.strip()}")
    if problems:
        print("\n打包失败：")
        for p in problems:
            print("  ✗ " + p)
        raise SystemExit(1)

    total = (out / "index.html").stat().st_size + (out / "README.txt").stat().st_size
    if total > a.max_mb * 1048576:
        # Not a failure, and no sidecar split. Measured on this page (headless
        # Chromium, file://): 6 courses 1.5 MB = 67 ms to first render, 30
        # courses 7.2 MB = 76 ms, 60 = 121 ms, 120 (28.6 MB) = 235 ms. Search is
        # BM25 rebuilt per course, so query cost does not grow with the corpus;
        # what grows is the file. Splitting it into <script src> shards would cost
        # the one thing the package is for — a single file the student opens.
        print(f"\n提醒：包 {total/1048576:.1f} MB，超过 --max-mb={a.max_mb:g}。"
              f"实测首屏 6 门 67ms / 30 门 76ms / 60 门 121ms / 120 门 235ms，"
              f"检索按课程重建、不随课程数变慢；真到打开要若干秒再考虑拆 sidecar。")
    print(f"\nok  {out}/  {total/1024:.0f} KB — {len(blob)} 门课，"
          f"{sum(len(c['chunks']) for c in blob)} chunks")
    for c in blob:
        print(f"  · {c['run_id']:<12} {Path(c['file'] or '?').name:<20} "
              f"{int(round(c['duration']/60)):>3} min  {(c['sha1'] or '')[:12]}")
    print("\n把 index.html 和 README.txt 放进视频所在目录，双击 index.html。")


if __name__ == "__main__":
    main()
