"""RapidOCR — the non-Apple side of :mod:`video_rag.ocr.backend`.

PP-OCR detection + recognition running on onnxruntime, so it works on any Linux
box with a CPU and needs no CUDA. Two things have to be normalised here so the
rest of the pipeline cannot tell which engine produced a line:

* geometry — RapidOCR reports four corner points in pixels with the origin at
  the **top left**; every caller expects a normalised box with the origin at the
  **bottom left** (``x``, ``y``, ``w``, ``h`` belonging to the bottom-left
  corner's vertical extent), because that is Vision's convention.
* junk — the detector also finds toolbar glyphs of one or two characters
  (``□``, ``×``), which Vision merges or drops. They are not filtered out by
  content here, only by the same ``min_conf`` threshold the Vision path uses,
  but empty strings are dropped since a caller sorting lines by box size should
  never be handed a blank candidate.

The model files ship inside the ``rapidocr`` wheel, so the first call is offline
after installation; nothing is downloaded at run time.
"""
from __future__ import annotations

from pathlib import Path

_ENGINE = None


def engine():
    """The process-wide RapidOCR instance (model load is slow, so do it once)."""
    global _ENGINE
    if _ENGINE is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as e:            # pragma: no cover - install-time hint
            raise SystemExit(
                "RapidOCR is not installed: pip install 'video-course-rag[ocr-linux]'"
            ) from e
        _ENGINE = RapidOCR()
    return _ENGINE


def to_vision_boxes(points, width: float, height: float, text: str,
                    conf: float) -> dict:
    """One RapidOCR line -> Vision's ``{text, conf, x, y, w, h}``.

    ``points`` are the four corner points in pixels (any winding order), so the
    box is taken as their bounding rectangle. The vertical coordinate is flipped
    because the two engines disagree about where zero is.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)          # y0 is the top edge, y1 the bottom edge
    return {
        "text": text,
        "conf": float(conf),
        "x": x0 / width,
        "y": 1.0 - (y1 / height),
        "w": (x1 - x0) / width,
        "h": (y1 - y0) / height,
    }


WINDOW_BUTTONS = {"最小化", "最大化", "还原", "关闭", "还原窗口"}


def _noise(text: str) -> bool:
    """True for lines a title picker must never see.

    PP-OCR happily turns toolbar glyphs into "text": a one- or two-character
    ASCII token ("PAQ", "aa", "×") or a window-control label from the top-right
    corner. Vision drops these, so keeping them here would hand the calibration
    step a fake "tallest line" — measured on a real frame, the tallest detector
    box was 'PAQ' while the slide's own heading sat further down, which made the
    whole title band disappear and every chapter lose its name.

    A real Chinese slide title is never three characters or fewer, so the rule
    costs nothing that matters and does not touch the macOS path.
    """
    if text in WINDOW_BUTTONS:
        return True
    if len(text) <= 3 and not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    if not any(ch.isalnum() for ch in text):
        return True
    return False


def read(png, min_conf: float = 0.25) -> list:
    """OCR one image and return lines in Vision's shape."""
    from PIL import Image

    path = Path(png)
    with Image.open(path) as im:
        width, height = im.size

    out = engine()(str(path))
    # numpy arrays here: `x or []` would raise "truth value of an array is
    # ambiguous", which is how this line first failed on a real frame.
    texts = [] if out.txts is None else list(out.txts)
    scores = [] if out.scores is None else list(out.scores)
    boxes = [] if out.boxes is None else list(out.boxes)

    lines = []
    for i, text in enumerate(texts):
        text = (text or "").strip()
        if not text or _noise(text):
            continue
        conf = float(scores[i]) if i < len(scores) else 0.0
        if conf < min_conf:
            continue
        if i >= len(boxes):
            continue
        lines.append(to_vision_boxes(boxes[i], width, height, text, conf))
    return lines
