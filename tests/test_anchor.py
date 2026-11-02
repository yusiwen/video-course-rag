"""The chapter table must survive an ASR run that splits sentences differently.

A 92-minute lecture produced 867 sentences on one machine and 873 on another
with the same audio and the same text; every chapter start after the first
moved split then pointed at the wrong sentence and slid by up to 47 seconds.
These tests pin the fix: boundaries are found by what was said, and the stored
number is only a fallback.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.chapter.fuse import resolve_start  # noqa: E402


def sent(t):
    return {"text": t, "start": 0, "end": 0}


def test_anchor_beats_a_shifted_number():
    sents = [sent("大家好我们开始上课"), sent("今天讲第一件事"), sent("嗯"), sent("接着讲第二件事")]
    # the table was written against a split without the "嗯" sentence:
    # it says sentence 3, which is now "嗯" — the anchor says otherwise.
    c = {"start_sentence": 3, "start_text": "接着讲第二件事"}
    assert resolve_start(c, sents) == 3  # 0-based index of "接着讲第二件事"


def test_number_is_used_when_no_anchor():
    sents = [sent("甲"), sent("乙"), sent("丙")]
    assert resolve_start({"start_sentence": 2}, sents) == 1


def test_unmatched_anchor_falls_back_to_the_number():
    sents = [sent("甲"), sent("乙")]
    c = {"start_sentence": 2, "start_text": "这段话根本不在转写里"}
    assert resolve_start(c, sents) == 1


def test_floor_keeps_chapters_in_order():
    sents = [sent("重复出现的一句话"), sent("中间"), sent("重复出现的一句话")]
    c = {"start_sentence": 1, "start_text": "重复出现的一句话"}
    # with a floor, a repeated phrase must not drag a later chapter backwards
    assert resolve_start(c, sents, floor=1) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
