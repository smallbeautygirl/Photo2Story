"""Hanzi -> Bopomofo (注音符號) conversion for Traditional Chinese captions.

Used by src/stages/zhuyin_render.py to annotate each character of a zh-tw
caption with its phonetic reading, for early readers.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypinyin import Style, load_phrases_dict, load_single_dict, pinyin

_TONE_MARKS = {"ˊ", "ˇ", "ˋ", "˙"}

# pypinyin's default dictionary gets several patterns wrong that are common
# in simple narrative Chinese (confirmed against real generated storybook
# pages, not just theoretical cases):
#
# 1. 著 is a four-way heteronym (zhe/zhuo2/zhao2/zhu4). pypinyin's default
#    reading is zhu4 ("write/compose"), but the overwhelmingly common case
#    in narration is the "V著" continuous-aspect grammatical particle
#    (neutral-tone "zhe"), e.g. "看著" (looking at), "笑著" (smiling). A short
#    allow-list protects the compounds where 著 legitimately keeps a
#    different reading; every other occurrence gets the particle reading.
# 2. Reduplicated kinship/pet address terms (爸爸, 寶寶, 狗狗, 奶奶, ...)
#    conventionally take neutral tone on the second syllable in everyday
#    Mandarin, but pypinyin's dictionary only has this for some (媽媽, 哥哥),
#    not others -- the rest fall back to two independent full-tone
#    single-character lookups.
# 3. 和 as the conjunction "and" is read han4 in Taiwan Mandarin, but
#    pypinyin's default reading is always he2 (its mainland/dictionary
#    reading), e.g. "貓和狗" (cat and dog). A short allow-list protects the
#    compounds where 和 legitimately keeps its he2/he4 reading.
#
# All lists are curated, not exhaustive -- add more compounds/terms here as
# they're found wrong in real generated output. This is an inherently
# long-tail problem (Mandarin heteronyms); the pattern -- one single-char
# override for the common case, plus a short protected-compound allow-list
# for the exceptions -- is the general fix for any newly found character.
_ZHU_COMPOUNDS: dict[str, list[list[str]]] = {
    "著名": [["zhu4"], ["ming2"]],
    "著作": [["zhu4"], ["zuo4"]],
    "著急": [["zhao2"], ["ji2"]],
    "顯著": [["xian3"], ["zhu4"]],
    "附著": [["fu4"], ["zhuo2"]],
    "土著": [["tu3"], ["zhu4"]],
    "執著": [["zhi2"], ["zhuo2"]],
}

_REDUPLICATED_ADDRESS_TERMS: dict[str, list[list[str]]] = {
    "爸爸": [["ba4"], ["ba"]],
    "寶寶": [["bao3"], ["bao"]],
    "狗狗": [["gou3"], ["gou"]],
    "奶奶": [["nai3"], ["nai"]],
    "姊姊": [["jie3"], ["jie"]],
    "姐姐": [["jie3"], ["jie"]],
    "弟弟": [["di4"], ["di"]],
    "妹妹": [["mei4"], ["mei"]],
    "爺爺": [["ye2"], ["ye"]],
}

_HE_COMPOUNDS: dict[str, list[list[str]]] = {
    "和平": [["he2"], ["ping2"]],
    "和諧": [["he2"], ["xie2"]],
    "溫和": [["wen1"], ["he2"]],
    "總和": [["zong3"], ["he2"]],
    "附和": [["fu4"], ["he4"]],
    "和好": [["he2"], ["hao3"]],
    "共和": [["gong4"], ["he2"]],
    "和解": [["he2"], ["jie3"]],
    "柔和": [["rou2"], ["he2"]],
    "隨和": [["sui2"], ["he2"]],
    "和睦": [["he2"], ["mu4"]],
    "調和": [["tiao2"], ["he2"]],
    "和氣": [["he2"], ["qi4"]],
}

# Registered once at import time, matching this codebase's existing
# one-time-setup-at-import convention (see gemini_client.py's load_dotenv).
load_phrases_dict({**_ZHU_COMPOUNDS, **_REDUPLICATED_ADDRESS_TERMS, **_HE_COMPOUNDS})
load_single_dict({ord("著"): "zhe", ord("和"): "han4"})


@dataclass(frozen=True)
class ZhuyinChar:
    """One character of input text, with its Bopomofo reading split out.

    `main` is the stacked initial/medial/final letters, tone mark excluded.
    `tone_mark` is one of "" (1st/flat tone), "ˊ", "ˇ", "ˋ", "˙" (neutral).
    Non-Hanzi characters (punctuation, Latin, digits, spaces) get
    `main=""`, `tone_mark=""` -- there is nothing to annotate.
    """

    char: str
    main: str
    tone_mark: str


def annotate(text: str) -> list[ZhuyinChar]:
    """Convert each character of `text` to its Bopomofo reading.

    pypinyin groups consecutive non-Hanzi characters into a single returned
    "reading" (e.g. the run "Hi world" comes back as one list entry, not one
    per character), so a naive zip(text, readings) misaligns as soon as any
    non-Hanzi text is present. This walks `text` by consuming exactly
    `len(reading)` characters per entry, re-expanding grouped passthrough
    runs back to one ZhuyinChar per original character.

    Leading/trailing whitespace is stripped first, matching
    stage3_assemble._wrap_to_width's own internal .strip() -- callers that
    wrap the plain-text path and the Zhuyin path with the same raw caption
    should see the same leading/trailing whitespace handling either way.
    """
    text = text.strip()
    if not text:
        return []

    readings = pinyin(text, style=Style.BOPOMOFO, heteronym=False)
    result: list[ZhuyinChar] = []
    pos = 0
    for (reading,) in readings:
        if text[pos : pos + len(reading)] == reading:
            # pypinyin returned this run unchanged: not Hanzi, no annotation.
            for ch in reading:
                result.append(ZhuyinChar(char=ch, main="", tone_mark=""))
            pos += len(reading)
        else:
            # A real Bopomofo reading, always for exactly one Hanzi character.
            char = text[pos]
            if reading and reading[-1] in _TONE_MARKS:
                result.append(ZhuyinChar(char=char, main=reading[:-1], tone_mark=reading[-1]))
            else:
                result.append(ZhuyinChar(char=char, main=reading, tone_mark=""))
            pos += 1
    return result
