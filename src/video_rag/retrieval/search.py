#!/usr/bin/env python3
"""Search a course index: BM25 (char bigrams + ascii words) fused with bge-m3.

Chinese has no spaces, so BM25 runs over character bigrams — no jieba
dependency, and it beats word segmentation on short technical queries like
"gitlab 账号". Dense and sparse are combined with reciprocal-rank fusion,
which needs no score calibration between the two.

Usage: scripts/search.py "<query>" [--run sample] [--k 5] [--open]
"""
from ..paths import run as _run
import argparse
import json
import os
import math
import re
import struct
import subprocess
from pathlib import Path

def ollama_embed() -> str:
    """Embedding endpoint; ``VIDEO_RAG_OLLAMA`` overrides host and port."""
    return os.environ.get("VIDEO_RAG_OLLAMA", "http://localhost:11434").rstrip("/") + "/api/embed"


MODEL = "bge-m3"


def tokens(text):
    t = text.lower()
    toks = re.findall(r"[a-z0-9_.+#/]+", t)
    for seg in re.findall(r"[\u4e00-\u9fff]+", t):
        if len(seg) == 1:
            toks.append(seg)
        else:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.tf, self.df = [], {}
        for d in self.docs:
            c = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            self.tf.append(c)
            for t in c:
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(self.docs)

    def idf(self, t):
        df = self.df.get(t, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, q):
        qt = tokens(q)
        out = [0.0] * self.n
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            for t in qt:
                f = tf.get(t)
                if not f:
                    continue
                out[i] += self.idf(t) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return out


def embed_one(text):
    body = json.dumps({"model": MODEL, "input": [text]}).encode()
    r = subprocess.run(["curl", "-s", "--max-time", "60", ollama_embed(), "-d", body],
                       capture_output=True, check=True)
    return json.loads(r.stdout)["embeddings"][0]


def load(run):
    d = _run(run).index_dir
    chunks = [json.loads(l) for l in open(d / "chunks.jsonl", encoding="utf-8")]
    meta = json.load(open(d / "meta.json"))
    raw = (d / "vectors.f32").read_bytes()
    dim = meta["dim"]
    vecs = [struct.unpack(f"{dim}f", raw[i * dim * 4:(i + 1) * dim * 4])
            for i in range(len(chunks))]
    return chunks, meta, vecs


def ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    r = [0] * len(scores)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def retrieve(run, query, k=6, chapter_weight=0.6):
    """Rank chunks with three signals.

    chunk BM25 keeps exact terms; dense handles paraphrase and ASR-mangled
    product names; the chapter-title list is the curation channel — when a
    question names a tool ("GitLab 账号"), the chapter the lecturer devoted to
    that tool should win over a passing mention three chapters later.
    """
    chunks, meta, vecs = load(run)
    chapters = json.load(open(str(_run(run).index_dir / "chapters.json"), encoding="utf-8"))["chapters"]
    ctitle = {c["id"]: (c["title"] + " " + (c.get("summary") or "")) for c in chapters}

    bm = BM25([c["text"].split("[屏幕]")[0].strip() or c["text"] for c in chunks])
    rs = ranks(bm.scores(query))
    qv = embed_one(query)
    rd = ranks([sum(x * y for x, y in zip(qv, v)) for v in vecs])

    ids = [c["chapter_id"] for c in chunks]
    uniq = sorted(set(ids), key=lambda x: (x is None, x))
    cbm = BM25([ctitle.get(i, "") for i in uniq])
    cr = ranks(cbm.scores(query))
    rank_of_chap = {cid: cr[n] + 1 for n, cid in enumerate(uniq)}
    rc = [rank_of_chap.get(i, len(uniq) + 1) for i in ids]

    fused = [(1 / (60 + rs[i]) + 1 / (60 + rd[i]) + chapter_weight / (60 + rc[i]))
             * (1.0 if chunks[i]["n_sent"] else 0.6) for i in range(len(chunks))]
    top = sorted(range(len(chunks)), key=lambda i: -fused[i])[:k]
    return chunks, meta, chapters, top, {"bm25": rs, "dense": rd, "chapter": rc}


def mmss(t):
    return f"{int(t//60):02d}:{int(t%60):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--run", default="sample")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    chunks, meta, chapters, top, parts = retrieve(a.run, a.query, a.k)
    rs, rd = parts["bm25"], parts["dense"]
    by_chap = {}
    for i in top:
        by_chap[chunks[i]["chapter_id"]] = by_chap.get(chunks[i]["chapter_id"], 0) + 1
    chap_rank = sorted(by_chap.items(), key=lambda kv: -kv[1])[:3]
    titles = {c["id"]: c["title"] for c in chapters}

    print(f"{meta['title']}  ({meta['duration']/60:.0f} min, {len(chunks)} chunks)\n")
    print("章节:")
    for cid, sc in chap_rank:
        print(f"  {titles.get(cid, cid)}")
    print("\n片段:")
    for n, i in enumerate(top, 1):
        c = chunks[i]
        print(f"{n}. {mmss(c['jump'])}  [{c['mode']}] {c['chapter_title']}   "
              f"bm25#{rs[i]} dense#{rd[i]}")
        print(f"   {c['text'].replace(chr(10), ' | ')[:150]}")

    if a.open:
        c = chunks[top[0]]
        print(f"\nlaunching mpv at {mmss(c['jump'])}")
        subprocess.run(["mpv", f"--start={c['jump']}", "--pause", meta["video"]])


if __name__ == "__main__":
    main()
