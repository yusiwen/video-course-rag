"""`jump` is only meaningful for a unit we actually named.

The scan exists to find the second where OCR reads the unit's own title again.
For a unit with no title every frame scores zero, so the loop spent nine
ffmpeg grabs and nine OCR calls to return the boundary it already had — 2214
calls on a lecture where 3 of 249 units carry a title.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.chapter import settle  # noqa: E402


def test_unnamed_unit_is_never_read():
    frames = []

    def read_at(t):
        frames.append(t)
        return ["工具栏", "剪切"]

    assert settle.scan_for_title(10.0, 200.0, "", read_at) is None
    assert settle.scan_for_title(10.0, 200.0, "   ", read_at) is None
    assert frames == [], f"an unnamed unit must not be scanned, read {len(frames)} frames"


def test_named_unit_stops_at_the_first_frame_that_matches():
    seen = []

    def read_at(t):
        seen.append(t)
        return ["工具栏", "学生名单"] if t >= 13 else ["工具栏"]

    got = settle.scan_for_title(10.0, 200.0, "学生名单", read_at)
    assert got is not None
    assert got[0] == 13
    assert seen == [10.0, 11.0, 12.0, 13.0], seen


def test_scan_never_leaves_the_unit():
    seen = []

    def read_at(t):
        seen.append(t)
        return ["别的东西"]

    assert settle.scan_for_title(10.0, 11.5, "学生名单", read_at) is None
    assert seen == [10.0, 11.0], seen


def test_block_match_rescues_a_different_biggest_line():
    # the tall line at t differs from the one at the unit midpoint, so the
    # joined top-5 block is what matches
    def read_at(t):
        return ["课程安排", "第一章 业务模型定义", "讲师"]

    got = settle.scan_for_title(10.0, 200.0, "第一章 业务模型定义", read_at)
    assert got and got[0] == 10.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
