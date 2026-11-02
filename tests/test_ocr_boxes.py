"""OCR boxes: the one thing every engine has to agree about.

The pipeline reads boxes, not just text — a page header is "the line above
y=0.86", a logo corner is "x>0.78 and y>0.82", and a title candidate is ranked
by box height. Vision reports those numbers with the origin at the bottom left;
RapidOCR reports pixels with the origin at the top left. If the flip is wrong
the OCR text still looks perfect in a log and every title silently disappears,
which is the failure mode this suite already learned to fear.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.ocr import backend  # noqa: E402
from video_rag.ocr.rapid import to_vision_boxes  # noqa: E402


def box(pts, w=1000, h=500, text="标题", conf=0.9):
    return to_vision_boxes(pts, w, h, text, conf)


def test_top_left_becomes_bottom_left():
    # a 40px-tall line sitting 50px from the top of a 500px image
    b = box([[100, 50], [300, 50], [300, 90], [100, 90]])
    assert abs(b["h"] - 40 / 500) < 1e-9
    assert abs(b["y"] - (1 - 90 / 500)) < 1e-9, b      # 0.82: near the header band
    assert abs(b["x"] - 0.1) < 1e-9
    assert abs(b["w"] - 0.2) < 1e-9


def test_header_line_lands_in_the_header_band():
    # a real header: 20px tall, 5px from the top
    b = box([[10, 5], [400, 5], [400, 25], [10, 25]], h=720)
    assert b["y"] > 0.86, b


def test_line_at_the_bottom_is_not_a_header():
    b = box([[10, 690], [400, 690], [400, 715], [10, 715]], h=720)
    assert b["y"] < 0.10, b


def test_corner_order_does_not_matter():
    pts = [[100, 50], [300, 50], [300, 90], [100, 90]]
    a = box(pts)
    b = box(list(reversed(pts)))
    c = box([[300, 90], [100, 50], [100, 90], [300, 50]])
    assert a == b == c, (a, b, c)


def test_engine_defaults_by_platform():
    import os
    saved = os.environ.pop("VIDEO_RAG_OCR", None)
    try:
        expected = "vision" if sys.platform == "darwin" else "rapidocr"
        assert backend.engine() == expected
        os.environ["VIDEO_RAG_OCR"] = "rapidocr"
        assert backend.engine() == "rapidocr"
        os.environ["VIDEO_RAG_OCR"] = "vision"
        assert backend.engine() == "vision"
    finally:
        os.environ.pop("VIDEO_RAG_OCR", None)
        if saved is not None:
            os.environ["VIDEO_RAG_OCR"] = saved


def test_missing_image_raises():
    try:
        backend.read("/nonexistent/nope.png")
    except RuntimeError as e:
        assert "no such image" in str(e)
    else:
        raise AssertionError("a missing image must raise, not return []")

def test_toolbar_glyphs_are_not_text():
    """PP-OCR turns icons into 1-3 character tokens; Vision does not.

    The calibration step picks each frame's tallest line as a title candidate,
    so a fake tall box from a toolbar is enough to erase the whole title band.
    """
    from video_rag.ocr.rapid import _noise

    for junk in ("PAQ", "aa", "×", "□", "—", "最小化", "最大化", "关闭"):
        assert _noise(junk), junk
    for real in ("差旅报销", "业务模型定义", "表单设计器", "报销", "Q&A 环节"):
        assert not _noise(real), real
