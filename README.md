# video-course-rag

[![ci](https://github.com/yusiwen/video-course-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/yusiwen/video-course-rag/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](#on-linux-no-apple-vision)
[![status](https://img.shields.io/badge/status-alpha-orange.svg)](docs/limitations.md)

**English** · [简体中文](README.zh-CN.md)

Turn recorded lectures into a **searchable course package** a student can open by
double-clicking one HTML file — no server, no Python, no account.

Status: **alpha**. Built and tested on macOS (the OCR step links Apple's Vision
framework); the ASR step is Chinese-first. See `docs/limitations.md` before you
point it at your own material.

## What it produces

```
runs/<id>/                 one directory per recording, all artifacts inside
  asr_sent.json            transcript, sentence level, absolute ms
  profile.json             measured title geometry for this recording
  chapters_fused.json      the table of contents that ships
  chunks.jsonl             retrieval units
  index/                   embeddings + page metadata
  eval/questions.jsonl     your own evaluation questions
  page/index.html          the page for this run
dist/<package>/index.html  the deliverable (all runs, one file, ~1.5 MB for 6 hours)
```

The package never contains videos. It matches a video by **content** (size plus a
SHA-1 of the first and last 4 MB), so renaming or moving the file changes nothing;
the student picks the folder once and the page finds the videos itself.

## How it works, in one paragraph

Audio is transcribed, the screen is segmented into *units* (a slide that stays put
is one unit), each unit is OCR'd, and a title is taken from a band of the frame
that `vrag title` **measures** rather than assumes. Chapters come from the
transcript (an LLM proposes boundaries by sentence id, then a local repair pass
fixes length violations before anything is re-asked), and are snapped to the
nearest picture change. Retrieval is BM25 over the speech plus BM25 over the
chapter titles, fused by reciprocal rank; the page shows the lecturer's words and
keeps the stale on-screen text underneath, dimmed.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[asr]"          # funasr + torch only if you want to transcribe
swiftc -O native/ocr/ocr.swift -o native/ocr/ocr

make demo                        # synthetic lecture, no real recording needed
vrag --runs-root examples/runs title demo
vrag --runs-root examples/runs page
open examples/runs/page.html
```

With your own recording:

```bash
vrag init lecture-1 /path/to/recording.mp4
vrag ingest lecture-1            # audio -> transcript -> screen units
vrag title  lecture-1            # measure the title band, then name the units
vrag chapter lecture-1           # transcript chapters, fused with the pictures
vrag index  lecture-1            # chunks, glossary, embeddings
vrag audit  lecture-1            # does the table of contents pass inspection?
vrag page                        # or: vrag pack --out dist/course
```

`vrag audit` is the gate: it fails a delivered table of contents whose titles are
mostly neutral labels, mostly shell text, or too often missing. A run that fails
should not be shipped.

### On Linux (no Apple Vision)

macOS reads the screen with Vision, compiled from `native/ocr/ocr.swift` on the
spot. On Linux the same job goes to RapidOCR, whose models ship inside the wheel,
so nothing is downloaded at run time:

```bash
sudo apt install ffmpeg
python3 -m venv .venv && . .venv/bin/activate
uv pip install -e ".[asr,ocr-linux]"

# with an NVIDIA card, swap torch for the CUDA wheels your driver matches
uv pip install --index-strategy unsafe-best-match \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.11.0+cu130" "torchaudio==2.11.0+cu130"
```

Three environment variables decide where the work runs:

| variable | meaning |
|---|---|
| `VIDEO_RAG_OCR` | `vision` or `rapidocr`; default is by platform |
| `VIDEO_RAG_ASR_DEVICE` | `cpu`, `cuda` or `mps`; macOS stays on cpu, elsewhere CUDA is used when present |
| `VIDEO_RAG_OLLAMA` | embedding endpoint; default `localhost:11434` |

Measured on the same 92-minute lecture (macOS: Vision + CPU; aarch64 GB10:
RapidOCR + CUDA):

- transcription is 18x faster (19.7 s for 92 minutes of audio) and the transcript
  is 0.9998 similar to the macOS one
- all 35 chapters land on the macOS boundary with 35/35 identical titles — but
  only because the chapter table carries `start_text` (see `docs/limitations.md`)
- screen segmentation is 7x slower (22 min vs 3 min); OCR is the only bottleneck
  on Linux
- calibration often finds no title band on Linux, so unit titles fall back to
  neutral labels. That is deliberate: naming a chapter after a toolbar button is
  worse than admitting it has no name, and chapter titles come from the LLM table
  anyway

## Repository layout

```
src/video_rag/        pipeline stages, one package per layer
  ingest/             audio, ASR, sentences
  segment/            screen units and boundary refinement
  title/              calibration, per-unit titles, the judge, the audit
  chapter/            transcript chapters, fusion, the unit-derived table
  retrieval/          chunks, glossary, embeddings, search, ask, evaluation
  pack/               the page (HTML/CSS/JS in assets/), the package builder
  ocr/                macOS Vision backend
native/ocr/ocr.swift  OCR helper source (compiled, never committed)
tests/                invariants, zero third-party dependencies
tools/                leak scan, page measurement
docs/                 English; *.zh-CN.md is the Chinese version
```

## Known limits (alpha)

- **Both macOS and Linux can build.** macOS reads frames with Vision, Linux with
  RapidOCR. The two do not agree word for word: screen segmentation is about 7x
  slower on Linux and unit titles are weaker there (see "On Linux" above).
- **Chinese first.** Sentence splitting, title heuristics and the UI copy were
  designed around Chinese lecture recordings. English material is untested and
  the title filter currently discards non-CJK lines.
- **Search is lexical.** The page has no query encoder, so it matches the words
  the lecturer used plus chapter titles. Same-meaning-different-words queries
  will miss; `vrag ask` (needs your own API key) reads the retrieved passages and
  can answer where search alone cannot.
- **The page is one file**, which is the point — 6 hours of lectures is about
  1.5 MB, and 120 hours stays under 0.3 s to first paint.

## Licence

MIT. No recorded material is included; `runs/` is ignored and a leak scan runs in
CI.
