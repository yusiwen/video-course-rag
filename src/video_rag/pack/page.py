#!/usr/bin/env python3
"""Generate viewer/index.html: one self-contained page, no server, no fetch.

Why inline everything: a page opened over file:// cannot fetch() a sibling
JSON file (CORS blocks it), so the index has to travel inside the HTML.

Two ways a course gets its video, and the page prefers the first that works:

1. ``src`` — a path written by this script (the teacher's own machine, or a
   relative path next to the page). Used when the file is simply there.
2. content addressing — the reader picks a video folder, the page hashes each
   candidate (size + sha1 of head||tail, 4MB each) and binds it to the course
   whose fingerprint matches. This is what a hand-out package relies on: the
   package ships an index and no video, and the reader's folder can hold any
   names, extra files or several courses.

The fingerprint is computed here with exactly the same rule the browser uses
(see ``fingerprint`` and ``sha1Of``), so the two must be changed together.

Search is BM25 over character bigrams, ported from search.py. Semantic search
needs bge-m3, which only exists in Python — that is the ask CLI (M4).

Ask runs in the browser: local BM25 picks the context, then the page calls an
OpenAI-compatible endpoint with the reader's own key (stored in localStorage,
shared by every file:// page on that machine). No key, no ask. The model is
told to cite chunk ids and never to print a time; times come from the index.

Usage: scripts/viewer.py [--out viewer/index.html] [--dist]
"""
import argparse
import hashlib
import json
import os
import urllib.parse
from pathlib import Path

from ..paths import Run, run as _run, runs_root

HASH_CHUNK = 4 * 1024 * 1024


def fingerprint(video: str):
    """(size, sha1(head||tail)) — must match sha1Of() in the page template.

    ``os.stat`` follows symlinks, so a symlinked video hashes the
    real file, which is also what the browser reads once the reader picks it.
    """
    try:
        st = os.stat(video)
    except OSError:
        return None, None, None
    size = st.st_size
    h = hashlib.sha1()
    with open(video, "rb") as f:
        if size <= HASH_CHUNK:
            h.update(f.read())
        else:
            h.update(f.read(HASH_CHUNK))
            f.seek(size - HASH_CHUNK)
            h.update(f.read(HASH_CHUNK))
    return size, h.hexdigest(), os.path.basename(video)


def rel_video(video, outdir):
    try:
        return os.path.relpath(video, outdir)
    except ValueError:
        return video


def load_assets() -> dict:
    """The page is one file, but its parts live in readable files.

    Inlining happens here and nowhere else: a packaged page must not fetch
    anything, and a reader of this repository should not have to read JavaScript
    out of a Python string literal.
    """
    here = Path(__file__).resolve().parent / "assets"
    out = {}
    for name, key in (("template.html", "template"), ("style.css", "css"), ("app.js", "js")):
        out[key] = (here / name).read_text(encoding="utf-8")
    return out


def build_page(entries, out_path: Path, dist: bool) -> dict:
    """Assemble one HTML file from the catalog entries; returns counts."""
    assets = load_assets()
    courses = []
    for entry in entries:
        r = _run(entry["run_id"])
        d = r.index_dir
        chapters = json.load(open(d / "chapters.json", encoding="utf-8"))["chapters"]
        units = json.load(open(d / "units.json", encoding="utf-8"))["units"]
        chunks = [json.loads(l) for l in open(d / "chunks.jsonl", encoding="utf-8")]
        size, sha1, fname = fingerprint(entry["video"])
        src = None if dist else urllib.parse.quote(
            rel_video(entry["video"], out_path.parent.resolve()))
        courses.append({
            "run_id": r.id,
            "title": entry["title"],
            "duration": entry["duration"],
            "src": src,
            "file": fname,
            "bytes": size,
            "sha1": sha1,
            "chapters": [{"id": c["id"], "start": c["start"], "end": c["end"],
                          "jump": c["jump"], "title": c["title"],
                          "summary": c.get("summary", ""),
                          # measured dead time: the caption must say so instead
                          # of repeating the previous slide's OCR text
                          "q": c.get("source") == "silence",
                          "sub": c.get("sub") or []} for c in chapters],
            "chunks": [{"i": i, "start": c["start"], "end": c["end"], "jump": c["jump"],
                        "t": c["text"], "ch": c["chapter_id"], "m": c["mode"],
                        "n": c["n_sent"]} for i, c in enumerate(chunks)],
            "n_units": len(units),
            "sentences": [[round(x["start"] / 1000, 2), round(x["end"] / 1000, 2), x["text"]]
                          for x in json.load(open(r.asr_sent, encoding="utf-8"))["items"]],
        })

    data = json.dumps(courses, ensure_ascii=False).replace("</", "<\\/")
    html = (assets["template"]
            .replace("__CSS__", assets["css"])
            .replace("__JS__", assets["js"])
            .replace("__DATA__", data)
            .replace("__SVC_PROBE__", "false" if dist else "true"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {"courses": len(courses),
            "chunks": sum(len(c["chunks"]) for c in courses),
            "chapters": sum(len(c["chapters"]) for c in courses),
            "missing_fingerprint": [c["run_id"] for c in courses if not c["sha1"]]}


def catalog(runs: str = "") -> list:
    cat = json.load(open(runs_root() / "catalog.json", encoding="utf-8"))
    entries = cat["courses"]
    wanted = [r.strip() for r in runs.split(",") if r.strip()]
    if wanted:
        order = {r: i for i, r in enumerate(wanted)}
        entries = [e for e in entries if e["run_id"] in order]
        entries.sort(key=lambda e: order[e["run_id"]])
        missing = [r for r in wanted if r not in {e["run_id"] for e in entries}]
        if missing:
            raise SystemExit(f"catalog has no entry for: {', '.join(missing)}")
    return entries


def main():
    ap = argparse.ArgumentParser(prog="vrag page")
    ap.add_argument("--out", default=None, help="output file (default: runs/page.html)")
    ap.add_argument("--dist", action="store_true",
                    help="packaged build: drop local video paths, require the reader to pick a folder")
    ap.add_argument("--runs", default="", help="comma separated run ids (default: all)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else runs_root() / "page.html"
    info = build_page(catalog(a.runs), out, a.dist)
    print(f"{out} ({out.stat().st_size/1024:.0f} KB) — {info['courses']} course(s), "
          f"{info['chunks']} chunks, {info['chapters']} chapters"
          + (f"; no fingerprint: {', '.join(info['missing_fingerprint'])}"
             if info["missing_fingerprint"] else ""))


if __name__ == "__main__":
    main()
