#!/usr/bin/env python3
"""Local ask service for the viewer (and anything else on this machine).

The viewer is a file:// page, so it cannot fetch a sibling JSON — but it can
call http://127.0.0.1:8765 if the response carries CORS headers that accept
the "null" origin. No framework, no port wars: one process, Ctrl-C to stop.

Endpoints:
  GET /health              -> {"ok": true, "runs": [...]}
  GET /ask?q=...&run=...   -> {"answer": ..., "cites": [{label, jump, chapter, ...}]}
  GET /video?run=...       -> the mp4 with Range support

/video exists because a file:// page pointing at /Volumes needs the browser to
be allowed to read removable volumes, and Chromium's media stack is fussy about
it. Streaming over loopback removes that variable entirely.

Usage: scripts/serve.py [--port 8765]
"""
from ..paths import runs_root
import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import ask as A  # noqa: E402
from video_rag.chapter.llm_chapters import load_dotenv  # noqa: E402


def stream_video(handler, path):
    """Minimal single-range file server; players need 206 + Content-Range."""
    size = os.path.getsize(path)
    rng = handler.headers.get("Range", "")
    start, end, code = 0, size - 1, 200
    if rng.startswith("bytes="):
        part = rng[6:].split(",")[0]
        a, _, b = part.partition("-")
        if a:
            start = int(a)
        if b:
            end = int(b)
        code = 206
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    else:
        handler.send_response(200)
    handler.send_header("Content-Type", "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(end - start + 1))
    handler.end_headers()
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = f.read(min(262144, left))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except BrokenPipeError:
                return
            left -= len(chunk)


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):      # keep the terminal readable
        sys.stderr.write("  " + fmt % args + "\n")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            cat = json.load(open(runs_root() / "catalog.json", encoding="utf-8"))
            self._send({"ok": True, "runs": [c["run_id"] for c in cat["courses"]]})
            return
        if u.path == "/video":
            qs = parse_qs(u.query)
            run = (qs.get("run") or ["sample"])[0]
            cat = json.load(open(runs_root() / "catalog.json", encoding="utf-8"))
            hit = next((c for c in cat["courses"] if c["run_id"] == run), None)
            if not hit or not os.path.exists(hit["video"]):
                self._send({"error": f"no video for {run}"}, 404)
                return
            stream_video(self, hit["video"])
            return
        if u.path == "/ask":
            qs = parse_qs(u.query)
            q = (qs.get("q") or [""])[0].strip()
            run = (qs.get("run") or ["sample"])[0]
            if not q:
                self._send({"error": "empty query"}, 400)
                return
            try:
                res = A.ask(q, run)
            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
                return
            cites = []
            for lbl in res["cites"]:
                c = res["by_label"][lbl]
                chap = next((x for x in res["chapters"] if x["id"] == c["chapter_id"]), None)
                cites.append({"label": lbl, "jump": c["jump"], "start": c["start"],
                              "end": c["end"], "chapter": c["chapter_title"],
                              "chapter_start": chap["start"] if chap else None,
                              "chapter_end": chap["end"] if chap else None,
                              "mode": c["mode"]})
            self._send({"answer": res["answer"], "cites": cites,
                        "insufficient": res["insufficient"],
                        "dropped": res["dropped_cites"],
                        "tokens": res["usage"].get("total_tokens") if res["usage"] else None})
            return
        self._send({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    load_dotenv()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
