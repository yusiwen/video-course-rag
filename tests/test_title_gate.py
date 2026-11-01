"""The title judge: what counts as a usable chapter title, and what does not.

Every case here is a failure that actually shipped or nearly shipped on the six
indexed courses, so a regression shows up as a test rather than as a table of
contents nobody reads closely.
"""
import sys
from pathlib import Path

from video_rag.title import gate as G  # noqa: E402
from video_rag.title.gate import (chrome_and_acronyms, chrome_for, dur_txt, neutral_title,  # noqa: E402
                       norm, similar, verdict)


def units(*lines, mode="ppt"):
    """A run where each unit saw one of the given line-sets."""
    return [{"mode": mode, "ocr_lines": list(ls)} for ls in lines]


def test_ui_word_rejected_whole_string_only():
    assert verdict("未登录", set(), set())[0] == "ui_word"
    # «打印设计器» contains 设计 and «扫码登录» contains 登录: substring matching
    # flagged both and turned two correct titles into errors.
    assert verdict("打印设计器", set(), set()) is None
    assert verdict("扫码登录", set(), set()) is None


def test_deck_name_and_shape():
    assert verdict("扫描仪.pptx", set(), set())[0] == "deck_name"
    assert verdict("【兼容模式】", set(), set())[0] == "deck_name"
    assert verdict("兼容模式 WPS", set(), set())[0] == "deck_name"
    assert verdict("ab", set(), set())[0] == "short"
    assert verdict("—— ★", set(), set())[0] == "short"
    assert verdict("pogle", set(), set())[0] == "ascii_only"
    assert verdict("ACME", set(), {"ACME"}) is None          # jargon is allowed
    assert verdict("acme", set(), {"ACME"}) is None          # case must not matter
    assert verdict("", set(), set())[0] == "empty"


def test_persistence_chrome_global_and_per_mode():
    # A header on every screen; a title that turns over; an app account name that
    # only dominates ITS OWN mode (a global count misses it when the mode is a
    # minority of the run).
    us = [{"mode": "ppt-fs", "ocr_lines": ["ACME企业应用", "首页", "账号甲"]} for _ in range(8)]
    us += [{"mode": "ppt", "ocr_lines": ["ACME企业应用", "幻灯片一"]} for _ in range(8)]
    # 8/16 units is chrome globally, and that is the point of the global scope
    chrome, mode_chrome, acr = chrome_and_acronyms(us)
    assert norm("ACME企业应用") in chrome
    assert verdict("ACME企业应用", chrome_for("ppt", chrome, mode_chrome), acr)[0] == "chrome_title"
    # a minority mode: 6 of 36 units, below the 25% global share but dominant
    # inside its own mode — this is the account name case, invisible globally
    few = [{"mode": "browser", "ocr_lines": ["服务设计器", "租户乙"]} for _ in range(6)]
    many = [{"mode": "ppt", "ocr_lines": ["标题甲", "标题乙"]} for _ in range(30)]
    chrome2, mc2, _ = chrome_and_acronyms(few + many)
    assert norm("租户乙") in mc2.get("browser", set())
    assert norm("租户乙") not in chrome2


def test_chrome_variants_are_chrome():
    # OCR reads one header several ways; counting raw strings fragmented it so
    # nothing reached the threshold and the product name named 24 chapters.
    us = [{"mode": "ppt", "ocr_lines": ["ACME开放服务云平台"]} for _ in range(8)]
    chrome, mc, acr = chrome_and_acronyms(us)
    for variant in ("，ACME开放服务云平台", "SACME开放服务云平台", "3ACME开放服务云平台"):
        v = verdict(variant, chrome_for("ppt", chrome, mc), acr)
        assert v is not None and v[0] in ("chrome_title", "chrome_variant"), variant


def test_composite_judged_fragment_by_fragment():
    chrome = {norm("百度一下，我就知道"), norm("ACME企业应用")}
    v = verdict("百度一下，我就知道 · ACME企业应用", chrome, set())
    assert v is not None and v[0] == "composite_chrome"
    # one good fragment is enough to name the chapter
    assert verdict("表单管理-分级字典案例 · 心找稻売模板", chrome, set()) is None


def test_similarity_separates_noise_from_neighbours():
    # short readings of one name sit at ~0.83: the picker excludes the band's
    # dominant text at 0.75, while the header rule clusters chapters at 0.90
    assert similar(norm("品开发平台"), norm("品IE开发平台"), 0.75)
    assert not similar(norm("品开发平台"), norm("品IE开发平台"), 0.90)
    assert similar(norm("ACME开放服务云平台"), norm("3ACME开放服务云平台"), 0.90)
    # three genuinely different chapters on one theme must NOT cluster
    a, b = norm("服务管理自定义通用服务服务代码书写"), norm("服务管理自定义通用服务表单配置服务")
    assert not similar(a, b, 0.90)


def test_neutral_labels():
    assert neutral_title("browser", 185) == "浏览器操作（3 分钟）"
    assert neutral_title("desktop", 42) == "桌面操作（42 秒）"
    assert neutral_title("mystery", 90) == "片段（1.5 分钟）"
    assert dur_txt(59) == "59 秒"
    assert dur_txt(360) == "6 分钟"


def test_neutral_labels_are_recognisable_and_silence_rows_are_not():
    """The shipped-TOC check keys on this pattern: a package mostly made of
    「幻灯片（3 分钟）」 must be refused, while a measured silence row must not be
    mistaken for one."""
    for mode in ("ppt", "ppt-fs", "browser", "desktop", "other-app", "???"):
        for dur in (5, 61, 599):
            t = G.neutral_title(mode, dur)
            assert G.NEUTRAL_RE.match(t), f"{mode}/{dur}: {t} not recognised as neutral"
    for t in ("休息（无内容 6 分钟）", "操作演示（无讲解 3 分钟）",
              "幻灯片设计", "桌面操作技巧", "创建订单管理两表"):
        assert not G.NEUTRAL_RE.match(t), f"{t} was taken for a neutral label"
