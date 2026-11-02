#!/usr/bin/env python3
"""Transcript -> chapter boundaries, then snap them onto picture events.

Step A (this file): send numbered sentences to an LLM, get back chapters that
reference SENTENCE IDS only, normalize/validate, write data/llm/<id>_chapters.json.
Step B: scripts/fuse_chapters.py snaps those onto visual onsets and produces
out/runs/<id>/chapters_fused.json.

Backends:
  --backend file            read data/llm/<id>_chapters.json (already written)
  --backend deepseek        OpenAI-compatible chat endpoint, needs DEEPSEEK_API_KEY
  --backend ollama          local http://localhost:11434, model via --model
  --backend openai          OPENAI_BASE_URL + OPENAI_API_KEY

The model must never emit timestamps: the prompt forbids it, and the parser
accepts only integers that index the sentence list.
"""
from ..paths import run as _run
import argparse
import json
import os
import re
import sys
from pathlib import Path


def load_dotenv():
    """Load <repo>/.env into os.environ without overriding real env vars.

    Kept dependency-free on purpose: the project already avoids pip packages
    for anything the pipeline touches at runtime.
    """
    p = Path(__file__).resolve().parents[3] / ".env"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
    return p

PROMPT = """你是课程视频的编排助手。下面是某门课的字幕，每行以 Sxxx 编号开头，后接时间戳和这句话的内容。

任务：把这段课划分为若干章（章节数控制在 {min_ch}–{max_ch} 个）。

硬性要求：
1. 只能输出**句子编号**，绝对不能输出时间（秒/分:秒都不行）。
2. 章节必须首尾相接、不重叠、覆盖全部句子。
3. 每章时长 60–300 秒（按时间戳估算），内容确实是同一个主题才合并。
4. 每章给一个 4–14 字的中文标题（不要"第一部分"这种序号），以及一句话摘要（≤40 字）。
5. 主题切换点通常出现在这些信号处：讲师说"我们再看一下""然后另一个是""第一/第二/第三"、开始介绍一个新工具或新概念。
6. 讲师在同一主题下反复演示操作时，不要切碎。

只输出 JSON，形如：
{{"chapters":[{{"start_sentence":1,"end_sentence":8,"title":"…","summary":"…"}}]}}

字幕：
{transcript}
"""


def load_sentences(run_id):
    return json.load(open(str(_run(run_id).asr_sent)))["items"]


def transcript_block(sents):
    lines = []
    for i, s in enumerate(sents, 1):
        t = int(s["start"] / 1000)
        lines.append(f"S{i:03d} [{t//60:02d}:{t%60:02d}] {s['text']}")
    return "\n".join(lines)


def call_deepseek(prompt):
    import urllib.request
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    body = json.dumps({
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    u = d.get("usage") or {}
    if u:
        print(f"[deepseek] prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} "
              f"total={u.get('total_tokens')} "
              f"cache_hit={u.get('prompt_cache_hit_tokens', '-')}", file=sys.stderr)
    return d["choices"][0]["message"]["content"]


def call_ollama(prompt, model):
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json", "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["response"]


def call_openai(prompt):
    import urllib.request
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not key:
        sys.exit("OPENAI_API_KEY not set")
    body = json.dumps({"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.2}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


SLACK = 12          # how far past the last sentence id we accept and clamp
SUFFIX = "（续）"


def tidy_title(t):
    """Collapse a title the model or the splitter marked up twice."""
    t = str(t).strip()
    while t.endswith(SUFFIX * 2):
        t = t[: -len(SUFFIX)]
    return t


def cont(title):
    """Mark a chapter as the continuation of the one before it, without stacking.

    Deriving twice (an over-long chapter split, then split again) used to produce
    "员工信息案例数据对象（续）（续）" in the table of contents.
    """
    t = tidy_title(title)
    return t if t.endswith(SUFFIX) else t + SUFFIX


def parse(text, n_sent, notes=None):
    txt = text.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt.split("\n", 1)[1] if "\n" in txt else txt
    obj = json.loads(txt)
    chapters = obj["chapters"] if isinstance(obj, dict) else obj
    out = []
    for c in chapters:
        s, e = int(c["start_sentence"]), int(c["end_sentence"])
        # On a 2000+ sentence lecture the model routinely writes ids past the end
        # (2248-2276 of 2275). Throwing the whole answer away for that costs a
        # full re-prompt (~46k tokens) and usually does not fix it, so:
        #   whole chapter past the end -> drop it
        #   start inside, small overshoot -> clamp the end
        #   start inside, absurd overshoot -> the model lost the scale, retry
        if s > n_sent:
            if notes is not None:
                notes.append(f"S{s:03d}-S{e:03d} is past the last sentence, chapter dropped")
            continue
        if e > n_sent + SLACK:
            raise ValueError(f"sentence range out of bounds: {s}-{e} (n={n_sent})")
        if e < s:
            if notes is not None:
                notes.append(f"S{s:03d}-S{e:03d} ends before it starts, chapter dropped")
            continue
        if s < 1:
            if notes is not None:
                notes.append(f"S{s:03d} start < 1, clamped to 1")
            s = 1
        if e > n_sent:
            if notes is not None:
                notes.append(f"S{s:03d}-S{e:03d} end clamped to S{n_sent:03d}")
            e = n_sent
        out.append({"start_sentence": s, "end_sentence": e,
                    "title": tidy_title(c.get("title", "")),
                    "summary": str(c.get("summary", "")).strip()})
    out.sort(key=lambda c: c["start_sentence"])
    # normalise: contiguous coverage
    for i, c in enumerate(out):
        if i == 0:
            c["start_sentence"] = 1
        else:
            c["start_sentence"] = out[i - 1]["end_sentence"] + 1
        if i == len(out) - 1:
            c["end_sentence"] = n_sent
    out = [c for c in out if c["end_sentence"] >= c["start_sentence"]]
    if not out:
        raise ValueError("no usable chapter after clamping")
    return out


MIN_S, MAX_S = 40.0, 300.0


def violations(chapters, sents):
    """Duration problems, phrased so the model can act on them."""
    out = []
    for c in chapters:
        a = sents[c["start_sentence"] - 1]["start"] / 1000
        b = sents[c["end_sentence"] - 1]["end"] / 1000
        if b - a > MAX_S:
            out.append(f"S{c['start_sentence']:03d}-S{c['end_sentence']:03d} 时长 {b-a:.0f}s "
                       f"超过 {MAX_S:.0f}s（这一章里显然有多个主题，请在中间的主题切换处切开）")
        elif b - a < MIN_S:
            out.append(f"S{c['start_sentence']:03d}-S{c['end_sentence']:03d} 时长 {b-a:.0f}s "
                       f"短于 {MIN_S:.0f}s（除非确实是独立小主题，否则并入相邻同主题章节）")
    return out


def repair_long(chapters, sents, max_s=MAX_S):
    """Split over-long chapters at the widest pause inside them.

    The model ignoring the 60-300s rule is common; retrying it costs a call and
    often changes nothing. Sentences now carry real per-sentence times, so the
    widest silence inside a chapter is a defensible place to cut.
    """
    out = []
    for c in chapters:
        a, b = c["start_sentence"], c["end_sentence"]
        dur = (sents[b - 1]["end"] - sents[a - 1]["start"]) / 1000
        if dur <= max_s or b - a < 4:
            out.append(c)
            continue
        # widest pause strictly INSIDE the chapter. k must leave a whole
        # sentence on each side: starting at k = a measured the gap before the
        # chapter, so on evenly spaced speech every recursion cut off one
        # sentence and 600 s became 297 one-sentence chapters.
        mid = (a + b) / 2
        best, at, best_gap = -1, None, None
        for k in range(a + 2, b - 1):
            gap = sents[k]["start"] - sents[k - 1]["end"]
            if gap > best or (gap == best and abs(k - mid) < abs(at - mid)):
                best, at, best_gap = gap, k, gap
        if at is None:
            at = int(mid)
        elif best_gap is not None and best_gap < 250:
            # no real pause anywhere (continuous speech): cut at the middle, the
            # widest-gap rule would otherwise pick an arbitrary tie
            at = int(mid)

        left = {**c, "end_sentence": at}
        right = {**c, "start_sentence": at + 1, "title": cont(c["title"])}
        out.extend(repair_long([left, right], sents, max_s))
    return out


CONT_RE = r"（续\d*）$"


def repair_to_fixpoint(chapters, sents, rounds=4):
    """Apply the local repairs until they stop changing anything."""
    prev, cur = None, chapters
    for _ in range(rounds):
        if prev is not None and len(cur) == len(prev):
            break
        prev = cur
        cur = repair_all(cur, sents)
    return cur, violations(cur, sents)


def settle(prompt, call, sents, retries=2):
    """Turn one model answer into a chapter table, calling again only when the
    answer cannot be parsed at all.

    Everything this pipeline can detect is a duration problem — too long or too
    short — because `parse()` already normalises ordering and contiguous
    coverage, and both duration classes have a local repairer. Re-prompting for
    them therefore bought nothing: measured on the corpus, all five courses spent
    three calls each (initial + two retries, ~46k input tokens per call on the
    longest) and were then repaired locally anyway, producing the same table.
    A residual long chapter survives into the fuse layer, which already cuts
    over-long chapters at the widest pause.
    """
    raw, chapters, last_err = call(prompt), None, None
    attempt = 0
    while True:
        notes = []
        try:
            chapters = parse(raw, len(sents), notes)
            for n in notes:
                print(f"  clamped: {n}")
            bad = violations(chapters, sents)
            if bad:
                before = len(chapters)
                chapters, left = repair_to_fixpoint(chapters, sents)
                print(f"  repaired locally: {before} -> {len(chapters)} chapters "
                      f"({len(bad)} duration problem(s) reported)")
                if left:
                    print(f"  accepted {len(left)} residual outlier(s) rather than "
                          f"re-prompting with the whole transcript:")
                    for v in left[:5]:
                        print(f"    {v}")
            return chapters
        except Exception as e:
            last_err = f"输出无法解析：{e}"
        attempt += 1
        if attempt > retries:
            print(f"giving up after {retries} retries, last problem:\n{last_err}")
            if chapters is None:
                sys.exit("no parsable output")
            before = len(chapters)
            chapters, _ = repair_to_fixpoint(chapters, sents)
            print(f"  repaired locally: {before} -> {len(chapters)} chapters")
            return chapters
        print(f"  retry {attempt}: {last_err.splitlines()[0]}")
        raw = call(prompt + f"\n\n上一次输出有以下问题，请修正后重新输出完整 JSON：\n{last_err}")


def repair_all(chapters, sents):
    """Everything the code can fix without asking the model again."""
    return merge_short(repair_long(chapters, sents), sents)


def unique_titles(chapters):
    """Two rows with the same title make a table of contents unreadable.

    Splitting one over-long chapter twice legitimately yields "A", "A（续）",
    "A（续）" — same base, so number the repeats instead of shipping duplicates.
    """
    import re
    seen = {}
    for c in chapters:
        t = tidy_title(c["title"]) or "未命名"
        n = seen.get(t, 0) + 1
        seen[t] = n
        c["title"] = t if n == 1 else re.sub(CONT_RE, "", t) + f"（续{n}）"
    return chapters


def auto_range(dur_s):
    return max(2, int(dur_s / 360)), max(4, int(dur_s / 100))


CN_DIGIT = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
            "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def apply_glossary(chapters, run_id):
    """Fix titles with the per-course term map and digit spellings.

    The transcript is the only input, so ASR homophones leak straight into
    titles (a homophone in the transcript can outvote the real spelling). Terms come
    from data/glossary/<run>.json, which is filled from OCR text + a human
    pass, not from the ASR itself.
    """
    gp = _run(run_id).glossary
    terms = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
    # longest first so a term wins over one of its own substrings
    keys = sorted(terms, key=len, reverse=True)
    for c in chapters:
        for field in ("title", "summary"):
            v = c.get(field, "")
            for k in keys:
                v = v.replace(k, terms[k])
            # 六点五 -> 6.5, 三 -> 3 only inside a version-like token
            v = re.sub(r"([零一二两三四五六七八九])点([零一二两三四五六七八九])",
                       lambda m: CN_DIGIT[m.group(1)] + "." + CN_DIGIT[m.group(2)], v)
            c[field] = v
    return len(terms)


def merge_short(chapters, sents, min_s=40.0):
    """Absorb fragments into a neighbour, forwards first then backwards.

    A fragment has to go somewhere: before this only the previous chapter was
    tried, so a short piece that could not legally extend it (it would pass the
    ceiling) stayed in the table as its own row. The budget allows a little
    slack — a row that is six seconds long is not prettier than a chapter that is
    six seconds over.
    """
    def t0(c):
        return sents[c["start_sentence"] - 1]["start"] / 1000

    def t1(c):
        return sents[c["end_sentence"] - 1]["end"] / 1000

    ceil = MAX_S * 1.1
    out, chs, i = [], list(chapters), 0
    while i < len(chs):
        c = chs[i]
        if t1(c) - t0(c) >= min_s or len(chs) == 1:
            out.append(c)
            i += 1
            continue
        if out and t1(c) - t0(out[-1]) <= ceil:
            out[-1]["end_sentence"] = c["end_sentence"]
            i += 1
            continue
        if i + 1 < len(chs) and t1(chs[i + 1]) - t0(c) <= ceil:
            chs[i + 1]["start_sentence"] = c["start_sentence"]
            i += 1
            continue
        out.append(c)
        i += 1
    return out


def anchor(text: str, n: int = 32) -> str:
    """A short comparable prefix of the sentence a chapter starts on.

    Sentence numbers only mean something inside the run that produced them: with
    a different machine or ASR build the split moves, and every number after the
    first moved split points at the wrong sentence — measured on a 92-minute
    lecture, that drifted chapter starts by up to 47 seconds by the end. The
    anchor lets the fuse layer find the boundary again by what was said.
    """
    t = "".join(ch for ch in text if not ch.isspace())
    return t[:n]


MIN_ANCHOR = 8


def anchor_at(sents, i: int, n: int = 32) -> str:
    """Anchor for the sentence at index i, widened when that sentence is thin.

    A chapter can open on "嗯" or "好"; matching eight characters against the
    whole transcript would hit a dozen wrong sentences. Pull the next sentence
    in until the anchor is long enough to be distinctive.
    """
    t = "".join(ch for ch in sents[i]["text"] if not ch.isspace())
    j = i + 1
    while len(t) < MIN_ANCHOR and j < len(sents):
        t += "".join(ch for ch in sents[j]["text"] if not ch.isspace())
        j += 1
    return t[:n]


def anchor_table(ch, sents):
    """Give every chapter the text its boundary said, using this transcript.

    A table that knows only sentence numbers is correct on the machine that
    produced it and wrong everywhere else: the same lecture splits into 873
    sentences there and 867 here, so every boundary past the first differing
    split points somewhere else. Tables written before `start_text` existed — or
    by hand — get their anchors derived here rather than being trusted.
    """
    out, added = [], 0
    n = len(sents)
    for c in ch:
        i = max(1, int(c["start_sentence"])) - 1
        j = int(c["end_sentence"])
        if i >= n or j > n:
            # The same lecture splits into 2275 sentences on one machine and
            # 2272 on another, so a table written there can point past the end
            # of the transcript here. Clamp and say so — the alternative is an
            # IndexError three stages later, inside merge_short.
            print(f"warning: chapter {c.get('title')!r} spans sentences {i + 1}-{j} "
                  f"but this transcript has {n}; clamped")
            i = min(i, n - 1)
            j = n
        j = max(j, i + 1)
        text = (c.get("start_text") or "").strip()
        if not text and i < n:
            text = anchor_at(sents, i)
            added += 1
        out.append({"start_sentence": i + 1,
                    "end_sentence": j,
                    "title": c["title"], "summary": c["summary"],
                    **({"start_text": text} if text else {})})
    return out, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--backend", default="file",
                    choices=["file", "deepseek", "ollama", "openai"])
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--min-chapters", type=int, default=0,
                    help="0 = derive from duration (≈no chapter longer than 6 min)")
    ap.add_argument("--max-chapters", type=int, default=0,
                    help="0 = derive from duration (≈no chapter shorter than 100 s)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--check-key", action="store_true",
                    help="print whether the API key is visible (masked) and exit")
    a = ap.parse_args()

    env_file = load_dotenv()
    if a.check_key:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        src = f"from {env_file}" if env_file else "no .env found"
        if key:
            print(f"DEEPSEEK_API_KEY visible: {key[:4]}…{key[-2:]} "
                  f"(len {len(key)}) [{src}]")
        else:
            print(f"DEEPSEEK_API_KEY not set [{src}]")
        return

    sents = load_sentences(a.run_id)
    out_path = str(_run(a.run_id).chapters_llm)

    if a.backend == "file":
        ch = json.load(open(out_path))["chapters"]
        print(f"read {len(ch)} chapters from {out_path}")
        chapters, added = anchor_table(ch, sents)
        if added:
            print(f"anchored {added}/{len(ch)} chapters from this machine's transcript")
    else:
        dur_s = (sents[-1]["end"] - sents[0]["start"]) / 1000
        lo, hi = (a.min_chapters, a.max_chapters)
        if not lo or not hi:
            d_lo, d_hi = auto_range(dur_s)
            lo, hi = (lo or d_lo), (hi or d_hi)
        print(f"duration {dur_s/60:.1f} min -> asking for {lo}-{hi} chapters")

        prompt = PROMPT.format(min_ch=lo, max_ch=hi, transcript=transcript_block(sents))
        call = {"deepseek": lambda p: call_deepseek(p),
                "ollama": lambda p: call_ollama(p, a.model),
                "openai": lambda p: call_openai(p)}[a.backend]

        chapters = settle(prompt, call, sents, a.retries)

    chapters = merge_short(chapters, sents)
    chapters = unique_titles(chapters)
    n_terms = apply_glossary(chapters, a.run_id)
    if n_terms:
        print(f"glossary: {n_terms} term(s) applied")
    # Record what each boundary said, not only its sentence number: a table is
    # routinely reused on a machine whose ASR splits sentences differently, and
    # then every number after the first moved split is wrong.
    for c in chapters:
        c.setdefault("start_text", anchor_at(sents, c["start_sentence"] - 1))
    json.dump({"video": a.run_id, "backend": a.backend,
               "chapters": chapters}, open(out_path, "w"), ensure_ascii=False, indent=1)

    def mmss(ms):
        t = int(ms / 1000)
        return f"{t//60:02d}:{t%60:02d}"

    print(f"{len(chapters)} chapters -> {out_path}")
    for c in chapters:
        s = sents[c["start_sentence"] - 1]["start"]
        e = sents[c["end_sentence"] - 1]["end"]
        print(f"  S{c['start_sentence']:03d}-S{c['end_sentence']:03d} "
              f"{mmss(s)}-{mmss(e)} {c['title']}")


if __name__ == "__main__":
    main()
