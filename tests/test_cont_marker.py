"""A continuation marker must not be stacked on top of a numbered one.

`cont()` appended 「（续）」 unless the title already ended with that exact string,
so a chapter the table had numbered 「…（续5）」 came back as 「…（续5）（续）」.
That shipped: the audit calls it `stacked_suffix` and blocks delivery on it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.chapter import fuse, llm_chapters as L  # noqa: E402


def test_a_numbered_marker_counts_as_already_marked():
    for mod in (fuse, L):
        assert mod.cont("自定义服务环境与组件（续5）") == "自定义服务环境与组件（续5）"
        assert mod.cont("自定义服务环境与组件（续）") == "自定义服务环境与组件（续）"


def test_a_plain_title_still_gets_one_marker():
    for mod in (fuse, L):
        assert mod.cont("表单设计器") == "表单设计器（续）"


def test_double_stacking_is_still_collapsed():
    for mod in (fuse, L):
        assert mod.cont("表单设计器（续）（续）") == "表单设计器（续）"


def test_splitting_one_chapter_twice_numbers_the_repeats():
    """The delivered table must not read 「A（续）」「A（续）」."""
    chapters = [{"start": 0, "title": "表单设计器"},
                {"start": 60, "title": "表单设计器（续）"},
                {"start": 120, "title": "表单设计器（续）"},
                {"start": 180, "title": "列表设计"}]
    fuse.number_repeats(chapters)
    names = [c["title"] for c in chapters]
    assert names == ["表单设计器", "表单设计器（续）", "表单设计器（续2）", "列表设计"], names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
