# Limits (alpha)

Read this before running the tool on your own recordings. Everything here was
observed while building the pipeline, not guessed.

## Platform

- **macOS uses Vision, Linux uses RapidOCR, and they are not interchangeable.**
  `native/ocr/ocr.swift` links Apple's Vision framework and is compiled with
  `swiftc`; on Linux `ocr/rapid.py` runs RapidOCR, whose models ship in the wheel.
  Both return the same shape — `{text, conf, x, y, w, h}`, normalised, origin
  bottom-left — because the consumers rely on it: a header is `y > 0.86`, a logo
  in the corner is `x > 0.78 and y > 0.82`. RapidOCR reports top-left pixel
  corners, so `rapid.py` flips `y` on the way out.
- **RapidOCR is slower and noisier, in ways the heuristics notice.** On the same
  frames it takes 1.0–1.8 s where Vision takes 0.1–0.2 s, and it finds 4–11 more
  lines, because it reads toolbar glyphs as text (`PAQ`, `最小化`). Segmentation
  of a 92-minute lecture therefore takes 22 minutes instead of 3.
- **A different engine finds a different title band.** Calibration builds each
  mode's title band out of the tallest line per frame. On PP-OCR the tallest boxes
  in a PowerPoint window are ribbon labels, so the band settled at y 0.885–0.965
  and 233 of 259 units were named after a toolbar button ("剪切", "文字方向") or
  the product header. Those bands are now rejected as window chrome and the units
  fall back to neutral labels — a chapter called "toolbar" is worse than one
  called "slides (4 min)".
- Transcription (`funasr`) is cross-platform; it is the only part of `ingest`
  that installs third-party code (about 1.2 GB with torch). With CUDA it runs at
  RTF 0.0036 (19.7 s for 92 minutes) against 0.32 on CPU, and the text is
  identical to the CPU run.
- **On the screen side the GPU is not the lever; the recognition batch is.**
  RapidOCR's torch engine runs the same PP-OCR models on CUDA with no extra
  package (torch is already there for ASR), and it halves the per-frame cost —
  but a first A/B showed only 1.6-2x, which is not worth a second inference path.
  Splitting the call explained why: detection is 130 ms of a 1072 ms frame on
  CUDA, so recognition is ~87%, and recognition was running in batches of 6 —
  about 21 small forward passes for a 121-line screen. At batch 32 the same
  frame takes 343 ms (median of 5 runs) against 2550 ms for onnxruntime on CPU,
  and the CPU engine alone improves from 2550 to 1495 ms. Batch 16 measured
  *slower* than 6 in one session, so the shape of this curve is worth re-measuring
  rather than assuming it is monotone.
- **The torch engine needs a first-use download; onnxruntime does not.** Torch
  checkpoints carry neither the character dictionary nor the metadata
  onnxruntime embeds, so RapidOCR fetches weights (10.2 MB det, 21.3 MB rec,
  0.6 MB cls) and `ppocrv6_dict.txt` from ModelScope into
  `VIDEO_RAG_OCR_MODELS` (~32 MB once). A machine without that reachability
  fails at construction, not at the first frame.
- **On RapidOCR the audit's chrome vocabulary absorbs real topics, and the rule
  had to learn not to condemn them.** Chrome is learned by counting OCR lines
  that recur across units; on Vision those lines are window furniture, but on
  RapidOCR the unit's lines include ribbon labels, so a phrase that is *also* a
  slide topic clears the threshold — `模型定义` did on a lecture where it is both.
  The variant rule then flagged delivered titles that merely contained it
  (`业务模型定义`, `模型内容定义`) as `chrome_variant`, which is an EXPO finding and
  therefore blocked delivery. The window was a flat "within six characters at
  0.75 similarity", so a four-character label sat inside a six-character title.
  It is now proportional — one character of slack at 0.85 — which is what OCR
  jitter actually moves (`SACME 平台名称` over `ACME 平台名称` scores 0.93), and a
  title that embeds a label is a composite, not a variant. After the change the
  five lectures audit at EXPO 0/0/0/0/0, unchanged titles in every one, and the
  fallback-layer number (RISK) stays high on RapidOCR — 128/19/75/0/0 against
  0-3 on Vision — because that layer *is* the unit titles, and on this platform
  they are ribbon labels.
- **`jump` is only defined for a unit that has a name.** The jump scan looks for
  the second where OCR reads the unit's own title again, but the loop scanned
  every unit anyway: on the reference lecture 231 of 246 units have no title, so
  each scored zero on all nine candidate frames and returned the start it
  already had. That was 1795 ffmpeg grabs plus 1795 OCR calls, ~22 minutes, to
  reproduce an answer the early return gives for free. Skipping unnamed units
  takes the stage to 21 frames and 15.6 s with **zero differences** across all
  246 units' `jump`, `jump_lag` and `jump_title_match`.
- **Know what a stage is made of before optimizing it.** `chapter` runs settle +
  table + llm_chapters + fuse, measured on a CUDA box at 340 s / 0.06 s / 0.04 s
  / 0.40 s **before** the unnamed-unit skip described above — the whole command is
  21 s after it. The other three links are rounding error either way, so "chapter
  is slow" is a statement about settle and nothing else. The cost there is frames × (grab +
  OCR); on the GPU path a frame is ~0.74 s, of which the OCR is ~0.34 s and a
  fresh ffmpeg seek the rest.
- **Switching the engine moves unit merges, not boundaries.** Onnxruntime/CPU vs
  torch/CUDA over the same lecture: 243/243 units matched within 2 s and 243/243
  kept their mode, but 71 unit titles differed — both sides being toolbar or
  product-header junk in a mode that has no title band — so the "merge
  neighbours with identical titles" rule produced 259 units against 249. The
  delivered table is what matters: 35/35 chapters, 35/35 identical titles, 18/35
  identical start times (max shift 11.6 s, the rest under 1 s, from snapping to
  a different unit onset).

## Reusing a chapter table on another machine

- **`start_sentence` is only valid in the run that produced it.** The LLM table
  stores sentence numbers, and a different machine (or another torch build, or
  another FunASR version) splits sentences slightly differently: the same audio
  gave 867 sentences on macOS and 873 on Linux, with a transcript that is 0.9998
  identical. Every chapter start after the first moved split then pointed at the
  wrong sentence, and the error accumulated — 12 of 35 chapters drifted, up to 47
  seconds by the end of the lecture, while all the titles stayed right. It reads
  as sloppy editing, not as a bug.
- Tables now also store `start_text`: the sentence the chapter opened on, widened
  to at least eight characters when that sentence is thin ("嗯"). The fuse layer
  matches it against the transcript, so the boundary is found by what was said.
  Old tables keep working through the number fallback.
- **What still differs after that fix:** visual snapping can move a boundary to a
  nearby unit onset, and unit segmentation itself is engine-dependent (246 units
  on macOS, 259 on Linux), so four chapters still start within 14 s of the macOS
  position. Chunk text similarity is about 0.68 because chunks group the same
  speech into different windows when the sentence split moves; the words are the
  same.

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

## The leak gate earned its keep

The first sweep of this repository was declared clean while it still carried real
material: a phrase from a recorded question in the page's empty-state copy, a
quoted transcript sentence in a function comment, the lecturer's name in a title
judge comment, an ASR mishearing in a docstring, real chapter titles in two test
fixtures, and a path from someone's Windows desktop in a gate comment. None of it
was in `runs/`; all of it was pasted into code as an illustrative example, which
is exactly how this material travels.

Two things now make that harder to repeat. The denylist holds those strings, so
they cannot come back silently; and the gate no longer skips `pack/assets/`,
which is where one of them was hiding — a directory skip is a hole you cannot see
from the exit code.
