# Limits (alpha)

Read this before running the tool on your own recordings. Everything here was
observed while building the pipeline, not guessed.

## Platform

- **The build host must be macOS.** `native/ocr/ocr.swift` links Apple's Vision
  framework and is compiled with `swiftc`. Titles, calibration and the audit all
  depend on OCR line boxes and confidences, so a different engine is a different
  tool: PaddleOCR or Tesseract are not drop-in replacements — the heuristics were
  tuned against Vision's coordinates.
- Transcription (`funasr`) is cross-platform; it is the only part of `ingest`
  that installs third-party code (about 1.2 GB with torch).

## Language

- Sentence splitting, the title judge and the UI copy assume Chinese lectures.
  The title filter drops lines that contain no CJK characters unless they are in
  the acronym set, so an English-language course produces neutral chapter labels
  rather than titles. This is the first thing to fix for non-Chinese material.

## Retrieval

- The shipped page is **lexical**: BM25 over the lecturer's words, plus BM25 over
  chapter titles, fused by reciprocal rank. There is no query encoder in the
  browser, so a paraphrase can fail even when the answer is in the transcript.
  Measured on recorded questions: adding the chapter channel moves rank-1 hits
  from 1/14 to 6/14 on paraphrases; a dense channel would add more, but it cannot
  run under `file://` (bge-m3 is 566M parameters).
- `vrag ask` uses the same retrieval and then lets an LLM answer from the
  retrieved passages. It needs an API key that the reader supplies; the key lives
  in `localStorage`, which every `file://` page on that machine can read. The
  page says so in its settings dialog, and the package README repeats it.
- Do not put a key in a package. `vrag pack` scans the output for key-shaped
  strings and refuses to finish if it finds one.

## Packaging

- The deliverable is a single HTML file by design. Measured: 120 hours of
  lectures is ~28 MB and still paints in 235 ms; retrieval cost does not grow with
  the number of courses because the index is rebuilt per course, lazily.
- Chapter titles are only as good as the screen. On material whose UI has no
  stable title row (application windows, desktop operations) the tool emits
  neutral labels like `幻灯片（37 分钟）` on purpose — inventing a topic reads worse
  than admitting there is none — and `vrag audit` reports how much of the table of
  contents that is.

## Two failure modes that look like success

Both of these were found by comparing this tool's output against the pipeline it
was ported from, and both produced a package that looked fine and was wrong.

- **A reused name hides a module.** `from ..ocr import backend as ocr` in a file
  that already defines `ocr(png)` binds the module to a name the local function
  then shadows: every `ocr.read(...)` raises `AttributeError`, an outer
  `try/except` turns that into `[]`, and segmentation runs with no screen text
  at all. Nothing crashes; the unit count just changes and every unit is
  classified `desktop`. Do not alias an import to a name the module already uses.
- **A cache trusts its own contents.** The calibration cache stored every frame
  it had seen, including frames read while OCR was broken. Empty entries never
  expire, so the poisoned vocabulary outlived the bug and kept changing which
  title band could be found. Cache only successful results, and treat an empty
  result as a miss.
