#!/usr/bin/env python3
"""Shared judgement about what counts as a usable chapter title.

Used by chapters.py and fuse_chapters.py (to decide when a title is NOT worth
shipping) and by title_audit.py (to measure how often that happens). Keeping one
implementation matters: a gate that disagrees with the code it guards is worse
than no gate.

The rules are deliberately objective — they only look at the run's own artifacts:

  persistence   a line present in a large share of all units is shell text, a
                product header or a watermark; a slide title turns over.
  deck names    a line carrying a file name came from a window title bar.
  shape         too short, symbol-only, or pure ASCII debris.

Nothing here decides whether a title is *good*; it decides whether it is
plausibly a topic at all. A title that fails gets replaced by a neutral label
describing what was on screen, which is honest: 「桌面操作（3 分钟）」 says what a
reader needs (nothing was taught here, someone was operating) without inventing
a topic out of OCR debris.
"""
import re
from collections import Counter

CHROME_SHARE = 0.25        # share of all units a line must appear in to be chrome
CHROME_MIN_UNITS = 5
MODE_DOMINANCE = 0.60      # share of ITS OWN mode that makes a line chrome there
MODE_MIN_UNITS = 5         # below this a mode is too small to judge dominance in
ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
DECK_RE = re.compile(r"\.(pptx|ppt|docx|doc|pdf|xlsx)|PowerPoint|【兼容模式】|WPS", re.I)
SYMBOL_RE = re.compile(r"^[\W\d_]+$")
CJK = re.compile(r"[\u4e00-\u9fff]")
# Whole-string matches only. Substring matching on these flagged real topics
# («导出报表» contains 导出, «批量删除» contains 删除) and made the gate noise.
UI_WORDS = frozenset((
    "未登录", "分享", "收藏", "搜索", "登录", "注册", "更多", "设置", "帮助",
    "文件", "开始", "插入", "设计", "切换", "动画", "幻灯片放映", "审阅",
    "视图", "保存", "另存为", "打印", "全屏", "返回", "首页", "上一页",
    "下一页", "确定", "取消", "编辑", "删除", "新建", "打开", "属性",
))
SEP = " · "
MAX_TITLE_CHARS = 40
CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
# A window/app title bar ("… · System Dashboard", "x.pptx - PowerPoint",
# "localhost:8080/…") names the application, not what is being taught.
WINDOW_RE = re.compile(r"\s[-|—]\s|localhost|https?://|System Dashboard|\.pptx|PowerPoint", re.I)
MODE_ZH = {"ppt": "幻灯片", "ppt-fs": "幻灯片", "browser": "浏览器操作",
           "desktop": "桌面操作", "other-app": "窗口操作"}


def _verdict_single(t, chrome, acronyms):
    if DECK_RE.search(t):
        return ("deck_name", "文件名或窗口标题栏")
    # Composites FIRST. A title built as "A · B" by the dedupe pass is never in
    # the chrome set as a whole string, so waiting for the whole to fail lets two
    # pieces of chrome through as one plausible title.
    if SEP in t:
        parts = [p.strip() for p in t.split(SEP) if p.strip()]
        if len(parts) > 1:
            if not any(_verdict_single(p, chrome, acronyms) is None for p in parts):
                return ("composite_chrome", "拼接起来的两半都是外壳/垃圾")
    tn = norm(t)
    if tn in chrome:
        return ("chrome_title", "高持续率文本，是外壳或水印")
    # OCR reads the same header differently each time («ACME 平台名称»,
    # «XACME 平台名称», «，ACME 平台名称»), so an exact set is not
    # enough: a near-match to a known chrome string is chrome too.
    for c in chrome:
        if abs(len(c) - len(tn)) <= 6 and similar(tn, c):
            return ("chrome_variant", "与外壳文字近似（OCR 变体）")
    if tn in UI_WORDS or t in UI_WORDS:
        return ("ui_word", "整条就是界面元素文字")
    if len(t) < 3 or SYMBOL_RE.match(t):
        return ("short", "太短或只有符号")
    if len(t) > MAX_TITLE_CHARS:
        return ("long", f"过长（{len(t)} 字）")
    if t.isascii() and t.upper() not in acronyms:
        return ("ascii_only", "纯 ASCII 且不是本课高频缩写")
    return None


def verdict(title, chrome=(), acronyms=()):
    """(kind, why) when this string is not a plausible topic, else None.

    A chapter title is often two fragments joined by " · " (the dedupe pass
    appends a distinguishing line to a repeated deck header). Judging that
    string whole lets two pieces of chrome pass as one plausible title —
    "搜索一下，我就知道 · •ACME企业应用" is a browser tab bar, not a topic — so a
    composite is acceptable only if at least one of its parts is.
    """
    t = (title or "").strip()
    if not t:
        return ("empty", "空标题")
    return _verdict_single(t, chrome, acronyms)


def norm(t):
    """Normalise before any comparison: OCR jitter must not split one string."""
    return "".join(ch for ch in (t or "").lower()
                   if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def chrome_and_acronyms(units):
    """Which OCR lines are chrome on this course, and which short caps are jargon.

    Two scopes, because one is not enough: a line can dominate the WHOLE run
    (a product header on every screen) or dominate its own mode (the account
    name «超级管理员» sitting in an app header or a watermark
    overlaid on 92% of the slides). Measured on real runs: the second case stays
    invisible to a global count when its mode is a minority of the run.
    """
    n = max(1, len(units))
    count = Counter()
    per_mode_n = Counter()
    per_mode = {}
    for u in units:
        m = u.get("mode", "?")
        per_mode_n[m] += 1
        bucket = per_mode.setdefault(m, Counter())
        for line in {norm(x) for x in (u.get("ocr_lines") or [])}:
            if line:
                count[line] += 1
                bucket[line] += 1
    # acronyms are detected on the RAW lines: normalisation lowercases, and
    # ACRONYM_RE requires capitals, so counting the normalised text made the
    # jargon set permanently empty and every pure-ASCII title "debris".
    raw = Counter()
    for u in units:
        for line in {x.strip() for x in (u.get("ocr_lines") or [])}:
            if line:
                raw[line] += 1
    chrome = {l for l, c in count.items() if c >= max(CHROME_MIN_UNITS, CHROME_SHARE * n)}
    mode_chrome = {}
    for m, bucket in per_mode.items():
        if per_mode_n[m] < MODE_MIN_UNITS:
            continue
        mode_chrome[m] = {l for l, c in bucket.items()
                          if c >= 3 and c / per_mode_n[m] >= MODE_DOMINANCE}
    acronyms = {l.upper() for l, c in raw.items() if ACRONYM_RE.match(l) and c >= 3}
    return chrome, mode_chrome, acronyms


def chrome_for(mode, chrome, mode_chrome):
    """The chrome set that applies to one unit/chapter."""
    return chrome | mode_chrome.get(mode, set())


def suspect(title):
    """(kind, why) for a title the rules above cannot judge, but a human would.

    These pass every shape test and are still not topics. Listed rather than
    rejected: rejecting on such weak evidence would throw away real titles
    (a slide literally titled with a schedule). They are the measured target of
    the calibration step, which can tell where the text sits and whether it
    turns over — signals this function does not have.
    """
    t = (title or "").strip()
    if CLOCK_RE.search(t):
        return ("clock_in_title", "含时分秒，像聊天水印/时间戳而不是主题")
    return None


def similar(a, b, threshold=0.75):
    """OCR reads the same header differently («数据平台» / «数据IE平台»)."""
    import difflib
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def dur_txt(sec):
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} 秒"
    m = sec / 60.0
    return f"{m:.0f} 分钟" if m >= 2 else f"{m:.1f} 分钟"


def neutral_title(mode, dur):
    """What to call a chapter whose title could not be trusted."""
    return f"{MODE_ZH.get(mode, '片段')}（{dur_txt(dur)}）"


# The shape neutral_title produces. Anything matching this in a *shipped* table of
# contents means a reader gets «幻灯片（3 分钟）» instead of a topic.
NEUTRAL_RE = re.compile("^(?:%s)（" % "|".join(sorted(set(MODE_ZH.values()) | {"片段"})))
