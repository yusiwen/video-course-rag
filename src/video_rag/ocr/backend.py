"""Where the OCR helper lives, and what to do when it is not built yet.

The helper is a Swift file that links Apple's Vision framework, so it has to be
compiled on the machine that runs it. Callers ask this module for the path; it
compiles on first use instead of assuming a build step happened.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "native" / "ocr" / "ocr.swift"
BINARY = SOURCE.with_suffix("")          # native/ocr/ocr


def helper() -> str:
    """Absolute path to the compiled helper, building it if necessary."""
    if not BINARY.exists():
        if not SOURCE.exists():
            raise SystemExit(f"OCR helper source missing: {SOURCE}")
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", str(SOURCE), "-o", str(BINARY)], check=True)
    return str(BINARY)


def read(png: str, min_conf: float = 0.25) -> list:
    """OCR one image.

    Failures raise: swallowing them turns an OCR outage into a silently
    different segmentation, which looks like a normal run but is wrong.
    """
    import json

    path = Path(png)
    if not path.exists():
        raise RuntimeError(f"OCR: no such image {path}")
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
