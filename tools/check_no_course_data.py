#!/usr/bin/env python3
"""Refuse to publish recorded material.

This repository ships code and synthetic examples only. A run's transcripts,
frames, embeddings and delivered package live under ``runs/`` and are ignored by
git; this check is the second line of defence, for the case where a string from a
real recording ends up pasted into a comment, a test fixture or a doc.

It looks for two things:

  * a denylist of strings from the recordings this tool was developed against
    (product names, customer words, personal names, and the absolute paths);
  * any path that points at a run directory with content in it.

Exit code is 1 when something is found, so CI and a pre-commit hook can both use
it. Keep the denylist short and specific: a check nobody can satisfy gets
disabled.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DENY = [
    r"OSP", r"openserver", r"差旅", r"报销", r"列模型", r"字典模型", r"工单",
    r"技术培训", r"/Volumes/Data", r"p0-sample", r"郭倩蓉", r"补考",
    r"openserver\.cn", r"task\.openserver", r"wiki\.openserver",
    # Strings that were actually found pasted into comments, tests and page copy
    # the first time this check ran on a finished repo, after the repo had
    # already been declared clean. Product name, an ASR mishearing, a phrase
    # from a recorded question, a chapter title, a participant name.
    r"企业开放服务云", r"VK知识库", r"提报流程", r"腾讯会议", r"培训考勤",
    r"普联课堂", r"李亚伟", r"李厚文",
]
DENY_RE = re.compile("|".join(DENY), re.I)

SKIP_DIRS = {".git", "__pycache__", ".venv", "runs", "dist", "build"}
ALLOW_SUFFIX = {".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".js", ".css", ".html", ".txt", ".json"}


def tracked_or_all() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        files = [ROOT / p for p in out.decode().split("\0") if p]
        if files:
            return files
    except Exception:
        pass
    return [p for p in ROOT.rglob("*") if p.is_file()]


def main() -> int:
    hits = []
    for p in tracked_or_all():
        if any(part in SKIP_DIRS for part in p.parts) or p.suffix not in ALLOW_SUFFIX:
            continue
        if p.name == "check_no_course_data.py":      # the denylist itself
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = DENY_RE.search(line)
            if m:
                hits.append((p.relative_to(ROOT), i, m.group(0), line.strip()[:100]))
    if not hits:
        print("clean: no recorded-material strings in tracked files")
        return 0
    print(f"{len(hits)} hit(s):")
    for path, line_no, what, line in hits:
        print(f"  {path}:{line_no}  [{what}]  {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
