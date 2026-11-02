#!/usr/bin/env python3
"""Promote a finished run into index/<run>/ and embed its chunks.

index/ is the product area: it survives deleting out/runs/ and is what the
viewer and the ask CLI read. Vectors come from bge-m3 through the local ollama
(1024-d, already L2-normalised, so cosine is a dot product).

Usage: scripts/index.py <run-id> [--title "..."]
"""
from ..paths import run as _run, runs_root
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

def ollama_embed() -> str:
    """Embedding endpoint. ``VIDEO_RAG_OLLAMA`` points it at another host/port.

    Default is the local daemon, which is what a single-machine setup has; a
    build box that keeps its models elsewhere can override just this URL.
    """
    return os.environ.get("VIDEO_RAG_OLLAMA", "http://localhost:11434").rstrip("/") + "/api/embed"


MODEL = "bge-m3"
BATCH = 16


def embed(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        body = json.dumps({"model": MODEL, "input": batch}).encode()
        r = subprocess.run(["curl", "-s", "--max-time", "120", ollama_embed(),
                            "-d", body], capture_output=True, check=True)
        d = json.loads(r.stdout)
        out.extend(d["embeddings"])
        print(f"  embedded {min(i+BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    run = a.run_id
    src = _run(run).root
    dst = _run(run).index_dir
    dst.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(l) for l in open(src / "chunks.jsonl", encoding="utf-8")]
    st = json.load(open(src / "states.json"))
    chapters = json.load(open(src / "chapters_fused.json"))

    digest = hashlib.sha1(open(src / "chunks.jsonl", "rb").read()).hexdigest()
    old = json.load(open(dst / "meta.json")) if (dst / "meta.json").exists() else {}

    for name, obj in (("chapters.json", chapters),
                      ("units.json", {"video": st["video"], "duration": st["duration"],
                                      "units": st["states"]}),
                      ("meta.json", {"run_id": run, "title": a.title or run,
                                     "video": st["video"], "duration": st["duration"],
                                     "n_chunks": len(chunks),
                                     "n_chapters": len(chapters["chapters"]),
                                     "model": MODEL, "dim": 1024,
                                     "chunks_sha1": digest,
                                     "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")})):
        json.dump(obj, open(dst / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    shutil.copyfile(src / "chunks.jsonl", dst / "chunks.jsonl")

    vec_path = dst / "vectors.f32"
    # vectors are only reusable if the chunk text is byte-identical: the same
    # count with different text silently serves stale embeddings
    if vec_path.exists() and old.get("chunks_sha1") == digest:
        print(f"vectors exist for {len(chunks)} chunks, reuse (delete to rebuild)")
    else:
        print(f"embedding {len(chunks)} chunks with {MODEL} via ollama")
        vecs = embed([c["text"] for c in chunks])
        with open(vec_path, "wb") as f:
            for v in vecs:
                f.write(struct.pack(f"{len(v)}f", *v))
        print(f"  -> {vec_path} ({vec_path.stat().st_size/1024/1024:.1f} MB)")

    cat_path = runs_root() / "catalog.json"
    cat = json.load(open(cat_path)) if cat_path.exists() else {"courses": []}
    entry = {"run_id": run, "title": a.title or run, "video": st["video"],
             "duration": st["duration"], "n_chapters": len(chapters["chapters"]),
             "n_chunks": len(chunks)}
    cat["courses"] = [c for c in cat["courses"] if c["run_id"] != run] + [entry]
    json.dump(cat, open(cat_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"indexed {run}: {len(chapters['chapters'])} chapters / {len(chunks)} chunks "
          f"/ {st['duration']/60:.1f} min")
    print(f"catalog: {len(cat['courses'])} course(s)")


if __name__ == "__main__":
    main()
