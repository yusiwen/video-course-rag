"""Where OCR comes from, and what shape it must return.

Two engines, one contract. On macOS the helper is a Swift file linking Apple's
Vision framework, compiled on first use. Everywhere else it is RapidOCR
(PP-OCR models on onnxruntime). Callers never branch on the platform: they call
``read()`` and get the same list of lines.

The contract is ``[{text, conf, x, y, w, h}]`` with normalised coordinates and
the origin at the **bottom left**, which is how Vision reports boxes. The title
heuristics depend on it — a page header is ``y > 0.86``, a logo corner is
``x > 0.78 and y > 0.82`` — so an engine whose boxes start at the top left has
to flip them before returning (see ``rapid.py``).

Failures raise. Swallowing an OCR outage turns it into a silently different
segmentation: every unit becomes ``desktop``, no unit gets a title, and the
package looks normal while being wrong.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "native" / "ocr" / "ocr.swift"
BINARY = SOURCE.with_suffix("")          # native/ocr/ocr

ENGINES = ("vision", "rapidocr")


def engine() -> str:
    """Which engine to use: ``VIDEO_RAG_OCR`` if set, else the platform default."""
    want = os.environ.get("VIDEO_RAG_OCR", "").strip().lower()
    if want:
        if want not in ENGINES:
            raise SystemExit(f"VIDEO_RAG_OCR={want!r} is not one of {ENGINES}")
        return want
    return "vision" if sys.platform == "darwin" else "rapidocr"


def helper() -> str:
    """Absolute path to the compiled Vision helper, building it if necessary."""
    if not BINARY.exists():
        if not SOURCE.exists():
            raise SystemExit(f"OCR helper source missing: {SOURCE}")
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", str(SOURCE), "-o", str(BINARY)], check=True)
    return str(BINARY)


def read(png: str, min_conf: float = 0.25) -> list:
    """OCR one image with whichever engine this machine has."""
    path = Path(png)
    if not path.exists():
        raise RuntimeError(f"OCR: no such image {path}")
    if engine() == "vision":
        return _vision(path, min_conf)
    from . import rapid
    return rapid.read(path, min_conf)


def _vision(path: Path, min_conf: float) -> list:
    out = subprocess.run([helper(), str(path), str(min_conf)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"OCR failed on {path} (exit {out.returncode}): "
                           f"{out.stderr.strip()[:300]}")
    try:
        return json.loads(out.stdout or "[]")
    except ValueError as e:
        raise RuntimeError(f"OCR returned unparseable output for {path}: "
                           f"{out.stdout[:200]!r}") from e
