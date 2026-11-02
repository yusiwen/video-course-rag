"""A chapter table is portable only because of its anchors.

Table rows used to carry a sentence number and nothing else, which is meaningful
only on the machine whose ASR produced it. Reading such a table has to derive
the anchor from the transcript in front of it, or every boundary after the first
differing split silently points at the wrong place.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.chapter import llm_chapters as L  # noqa: E402

SENTS = [{"text": "第一章 业务模型定义", "start": 0, "end": 1000},
         {"text": "我们从表单设计器开始", "start": 1000, "end": 2000},
         {"text": "第二个主题是列表", "start": 2000, "end": 3000}]


def test_a_table_without_anchors_gets_them_from_the_local_transcript():
    table = [{"start_sentence": 1, "end_sentence": 1, "title": "业务模型", "summary": ""},
             {"start_sentence": 3, "end_sentence": 3, "title": "列表", "summary": ""}]
    out, added = L.anchor_table(table, SENTS)
    assert added == 2
    assert out[0]["start_text"].startswith("第一章")
    assert "第二个主题" in out[1]["start_text"]


def test_existing_anchors_are_kept_verbatim():
    table = [{"start_sentence": 1, "end_sentence": 3, "title": "x", "summary": "",
              "start_text": "别人写好的一句话"}]
    out, added = L.anchor_table(table, SENTS)
    assert added == 0
    assert out[0]["start_text"] == "别人写好的一句话"


def test_a_row_past_the_transcript_keeps_no_anchor_and_says_so():
    table = [{"start_sentence": 99, "end_sentence": 99, "title": "越界", "summary": ""}]
    out, added = L.anchor_table(table, SENTS)
    assert added == 0
    assert "start_text" not in out[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
