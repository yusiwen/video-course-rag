"""Command line entry point.

    vrag ingest  <run-id> [video]     audio -> transcript -> screen units
    vrag title   <run-id>             measure title geometry, then name the units
    vrag chapter <run-id>             transcript chapters, fused with the pictures
    vrag units   <run-id>             unit-derived chapter table (no LLM needed)
    vrag index   <run-id>             chunks, glossary terms, embeddings
    vrag audit   [run-id ...]         does the table of contents pass inspection?
    vrag page    [--out FILE]         one self-contained HTML page for every run
    vrag pack    --out DIR            the deliverable a student receives
    vrag serve   <run-id>             optional local server (video over loopback)
    vrag ask     <run-id> "question"  answer from the index (needs an API key)
    vrag evaluate <run-id>            score answers against eval/questions.jsonl

Stages are invoked through the same functions the CLI exposes, so a hand-run step
and a scripted step cannot drift apart.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from .paths import Run, all_runs, run as _run, runs_root

STAGES = {
    "asr": "video_rag.ingest.asr",
    "sentences": "video_rag.ingest.sentences",
    "states": "video_rag.segment.states",
    "refine": "video_rag.segment.refine",
    "calibrate": "video_rag.title.calibrate",
    "retitle": "video_rag.title.retitle",
    "llm_chapters": "video_rag.chapter.llm_chapters",
    "fuse": "video_rag.chapter.fuse",
    "settle": "video_rag.chapter.settle",
    "table": "video_rag.chapter.table",
    "glossary": "video_rag.retrieval.glossary",
    "chunks": "video_rag.retrieval.chunks",
    "index": "video_rag.retrieval.index",
    "page": "video_rag.pack.page",
    "build": "video_rag.pack.build",
    "serve": "video_rag.pack.serve",
    "ask": "video_rag.retrieval.ask",
    "evaluate": "video_rag.retrieval.evaluate",
    "audit": "video_rag.title.audit",
}


def stage(name: str, *args) -> None:
    """Run one stage in-process with a synthetic argv.

    The stage modules own their argument parsing; calling them this way keeps a
    single code path between `vrag ingest` and `python -m video_rag.ingest.asr`.
    """
    mod = importlib.import_module(STAGES[name])
    old = sys.argv
    sys.argv = [f"vrag {name}"] + [str(a) for a in args]
    try:
        mod.main()
    finally:
        sys.argv = old


def _need_run(ns) -> Run:
    rid = ns.run or (all_runs()[-1].id if all_runs() else None)
    if not rid:
        raise SystemExit("no run given and runs/ is empty — try: vrag init <run-id> <video>")
    return _run(rid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vrag", description="Turn lecture recordings into a course package.")
    ap.add_argument("--runs-root", help="where runs live (default: ./runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a run directory from a video")
    p.add_argument("run")
    p.add_argument("video")

    for name, help_ in (("ingest", "audio, transcript, screen units"),
                        ("title", "calibrate + name the units"),
                        ("chapter", "transcript chapters fused with pictures"),
                        ("units", "unit-derived chapters (no LLM)"),
                        ("index", "chunks, glossary, embeddings")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("run", nargs="?")
        if name == "ingest":
            p.add_argument("video", nargs="?")

    p = sub.add_parser("audit", help="inspect the delivered table of contents")
    p.add_argument("runs", nargs="*")

    p = sub.add_parser("page", help="build the self-contained HTML page")
    p.add_argument("--out", default=None)
    p.add_argument("--dist", action="store_true", help="package mode: no local-service probe")

    p = sub.add_parser("pack", help="build the student deliverable")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--runs", default="", help="comma separated run ids (default: all)")

    p = sub.add_parser("serve", help="optional local server")
    p.add_argument("run", nargs="?")

    p = sub.add_parser("ask", help="answer a question from the index")
    p.add_argument("run")
    p.add_argument("question")

    p = sub.add_parser("evaluate", help="score against runs/<id>/eval/questions.jsonl")
    p.add_argument("run")

    p = sub.add_parser("stage", help="run a single low-level step (advanced)")
    p.add_argument("name", choices=sorted(STAGES))
    p.add_argument("args", nargs=argparse.REMAINDER)

    ns = ap.parse_args(argv)
    if ns.runs_root:
        import os
        os.environ["VIDEO_COURSE_RAG_RUNS"] = ns.runs_root

    if ns.cmd == "init":
        from .ingest.source import describe
        run = Run.get(ns.run).ensure()
        run.write_json(run.source, describe(ns.video))
        print(f"created {run.root} — {ns.video}")
        return 0

    if ns.cmd == "ingest":
        run = _need_run(ns)
        run.ensure()
        if ns.video:
            from .ingest.source import describe
            run.write_json(run.source, describe(ns.video))
        elif not run.source.exists():
            raise SystemExit("no source.json — run `vrag init <run-id> <video>` first")
        if not run.audio.exists():
            from .ingest.source import extract_audio
            extract_audio(run)
        stage("asr", run.id)
        stage("sentences", run.id)
        stage("states", run.id)
        stage("refine", str(run.root))
        return 0

    if ns.cmd == "title":
        run = _need_run(ns)
        stage("calibrate", run.id)
        stage("retitle", str(run.root))
        return 0

    if ns.cmd == "chapter":
        run = _need_run(ns)
        stage("settle", str(run.root))
        stage("table", "6.0", run.id)
        stage("llm_chapters", run.id)
        stage("fuse", run.id)
        return 0

    if ns.cmd == "units":
        run = _need_run(ns)
        stage("settle", str(run.root))
        stage("table", "6.0", run.id)
        return 0

    if ns.cmd == "index":
        run = _need_run(ns)
        stage("glossary", run.id)
        stage("chunks", run.id)
        stage("index", run.id)
        return 0

    if ns.cmd == "audit":
        args = ns.runs or [r.id for r in all_runs()]
        stage("audit", *args)
        return 0

    if ns.cmd == "page":
        args = []
        if ns.out:
            args += ["--out", ns.out]
        if ns.dist:
            args += ["--dist"]
        stage("page", *args)
        return 0

    if ns.cmd == "pack":
        args = ["--out", ns.out, "--title", ns.title]
        if ns.runs:
            args += ["--runs", ns.runs]
        stage("build", *args)
        return 0

    if ns.cmd == "serve":
        stage("serve", *([ns.run] if ns.run else []))
        return 0

    if ns.cmd == "ask":
        stage("ask", ns.question, "--run", ns.run)
        return 0

    if ns.cmd == "evaluate":
        stage("evaluate", "--run", ns.run)
        return 0

    if ns.cmd == "stage":
        stage(ns.name, *ns.args)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
