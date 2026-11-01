"""How many model calls one chapter table costs.

The retry loop used to fire on every duration violation, even though `parse()`
already normalises ordering and coverage and both duration classes have a local
repairer. On the indexed corpus that meant three calls per course (initial + two
retries) followed by the same local repair — ~46k wasted input tokens on the
longest lecture. These tests pin the order: repair first, call again only when
the repair cannot close the gap.
"""
import sys
from pathlib import Path

from video_rag.chapter import llm_chapters as A  # noqa: E402


def sents(n=600, step_ms=2000):
    """A recording whose sentences are two seconds apart, one per sentence."""
    out = []
    for i in range(n):
        t = i * step_ms
        out.append({"start": t, "end": t + step_ms, "text": f"sentence {i+1}"})
    return out


def table(*ranges):
    return {"chapters": [{"start_sentence": a, "end_sentence": b,
                          "title": f"S{a}-S{b}", "summary": "x"} for a, b in ranges]}


def answer(obj_or_text):
    return obj_or_text if isinstance(obj_or_text, str) else __import__("json").dumps(obj_or_text)


class Caller:
    """A fake model: returns queued answers and counts how often it was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return answer(self.answers[min(self.calls - 1, len(self.answers) - 1)])


def durations(chapters, s):
    return [round((s[c["end_sentence"] - 1]["end"] - s[c["start_sentence"] - 1]["start"]) / 1000, 1)
            for c in chapters]


def test_duration_violations_are_repaired_without_a_second_call():
    s = sents()
    # one 600s chapter and one 10s fragment: both are local-repair cases
    bad = table((1, 300), (301, 306), (307, 600))
    call = Caller(bad)
    ch = A.settle("prompt", call, s, retries=2)
    assert call.calls == 1, f"{call.calls} calls: the model was asked to fix what the code can fix"
    assert all(40 <= d <= 300 for d in durations(ch, s)), durations(ch, s)
    assert ch[0]["start_sentence"] == 1 and ch[-1]["end_sentence"] == len(s)


def test_unparsable_output_still_costs_a_retry():
    s = sents()
    call = Caller("这不是 JSON", table((1, 300), (301, 600)))
    ch = A.settle("prompt", call, s, retries=2)
    assert call.calls == 2, "a response that is not JSON has to be asked for again"
    assert ch[0]["start_sentence"] == 1 and ch[-1]["end_sentence"] == len(s)


def test_unsplittable_chapter_is_accepted_not_retried():
    """Three sentences over 400 s cannot be split by the code, and two re-prompts
    did not fix the same class of problem on any of the five real courses."""
    s = [{"start": 0, "end": 2000, "text": "a"},
         {"start": 400000, "end": 402000, "text": "b"},
         {"start": 404000, "end": 406000, "text": "c"}]
    call = Caller(table((1, 3)))
    ch = A.settle("prompt", call, s, retries=2)
    assert call.calls == 1
    assert len(ch) == 1


def test_parse_clamps_a_small_overshoot_and_refuses_a_wild_one():
    notes = []
    ch = A.parse(__import__("json").dumps(table((1, 105))), 100, notes)
    assert ch[-1]["end_sentence"] == 100
    assert notes and "clamped" in notes[0]
    notes = []
    ch = A.parse(__import__("json").dumps(table((1, 100), (101, 105))), 100, notes)
    assert ch[-1]["end_sentence"] == 100
    assert any("past the last sentence" in n for n in notes)
    try:
        A.parse(__import__("json").dumps(table((1, 40), (41, 300))), 100, [])
    except ValueError as e:
        assert "out of bounds" in str(e)
    else:
        raise AssertionError("a wild overshoot must not be silently clamped")


def test_violations_only_knows_durations():
    """Documents the code fact the repair-first order rests on."""
    s = sents(200)
    ch = A.parse(__import__("json").dumps(table((150, 200), (1, 50), (51, 149))), 200, [])
    # out-of-order input is normalised by parse(); nothing structural survives to
    # violate, so the only thing left to complain about can be a duration
    assert A.violations(ch, s) == [] or all("时长" in v for v in A.violations(ch, s))
