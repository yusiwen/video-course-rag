"""RapidOCR — the non-Apple side of :mod:`video_rag.ocr.backend`.

PP-OCR detection + recognition, normalised so the rest of the pipeline cannot
tell which engine produced a line:

* geometry — RapidOCR reports four corner points in pixels with the origin at
  the **top left**; every caller expects a normalised box with the origin at the
  **bottom left** (``x``, ``y``, ``w``, ``h`` belonging to the bottom-left
  corner's vertical extent), because that is Vision's convention.
* junk — the detector also finds toolbar glyphs of one or two characters
  (``□``, ``×``, ``PAQ``), which Vision merges or drops. They are dropped here,
  because the title picker takes each frame's tallest line as a title candidate
  and a fake tall box from a ribbon is enough to erase a whole mode's title band.

Two inference engines are supported, selected by environment (see :func:`options`):

* ``onnxruntime`` (default) — models ship inside the ``rapidocr`` wheel, so the
  first call is offline; CPU only.
* ``torch`` — runs the same PP-OCR models on CUDA when ``VIDEO_RAG_OCR_DEVICE``
  is ``cuda``. Measured on a 121-line 1080p frame: 343 ms against 2550 ms for
  onnxruntime on CPU, and detection alone drops from 317 ms to 130 ms. It costs
  a one-time download of the weights (~32 MB) and the character dictionary from
  ModelScope into ``VIDEO_RAG_OCR_MODELS``, because torch checkpoints carry
  neither.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENGINE = None
_DEFAULT_BATCH = 32       # recognition batches; see options() for why this size
_DEFAULT_MODELS = Path.home() / ".cache" / "video-course-rag" / "ocr"


def models_dir() -> Path:
    """Where the torch engine keeps downloaded weights and the character dict."""
    return Path(os.environ.get("VIDEO_RAG_OCR_MODELS") or _DEFAULT_MODELS)


def options() -> tuple:
    """Read the engine choice out of the environment.

    Returns ``(engine, device, batch)``. Kept free of any ``rapidocr`` import so
    it stays testable where the package is not installed (macOS runs Vision).

    ``VIDEO_RAG_OCR_ENGINE``  onnxruntime (default) | torch
    ``VIDEO_RAG_OCR_DEVICE``  cpu (default) | cuda
    ``VIDEO_RAG_OCR_BATCH``   recognition batch size, 32 for torch by default

    The batch default is not cosmetic: recognition is 87% of a dense frame and
    the stock batch of 6 makes ~21 small forward passes for a 121-line screen.
    Raising it to 32 measured 1076 -> 343 ms on CUDA, and 2550 -> 1495 ms on the
    CPU engine, so the number is worth having even without a GPU.
    """
    engine = (os.environ.get("VIDEO_RAG_OCR_ENGINE") or "onnxruntime").strip().lower()
    if engine not in ("onnxruntime", "torch"):
        raise SystemExit(
            f"VIDEO_RAG_OCR_ENGINE={engine!r} is not one of onnxruntime, torch")
    device = (os.environ.get("VIDEO_RAG_OCR_DEVICE") or "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        raise SystemExit(f"VIDEO_RAG_OCR_DEVICE={device!r} is not one of cpu, cuda")

    raw = os.environ.get("VIDEO_RAG_OCR_BATCH")
    if raw:
        try:
            batch = int(raw)
        except ValueError as e:
            raise SystemExit(f"VIDEO_RAG_OCR_BATCH={raw!r} is not an integer") from e
        if batch < 1:
            raise SystemExit(f"VIDEO_RAG_OCR_BATCH={batch} must be >= 1")
    else:
        batch = _DEFAULT_BATCH if engine == "torch" else None
    return engine, device, batch


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
        name, device, batch = options()
        params = {}
        if name == "torch":
            try:
                from rapidocr.utils.typings import EngineType
            except ImportError as e:        # pragma: no cover - install-time hint
                raise SystemExit(
                    "RapidOCR's torch engine is unavailable: pip install torch"
                ) from e
            # engine_type must be the enum, not its string value; RapidOCR
            # rejects a plain string with a TypeError.
            for task in ("Det", "Cls", "Rec"):
                params[f"{task}.engine_type"] = EngineType.TORCH
            params["EngineConfig.torch.use_cuda"] = device == "cuda"
            params["Global.model_root_dir"] = str(models_dir())
            params["Rec.rec_batch_num"] = batch or _DEFAULT_BATCH
        elif batch:
            params["Rec.rec_batch_num"] = batch
        _ENGINE = RapidOCR(params=params) if params else RapidOCR()
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
