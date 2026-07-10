"""Hanzi -> Bopomofo (注音符號) conversion for Traditional Chinese captions.

Used by src/stages/zhuyin_render.py to annotate each character of a zh-tw
caption with its phonetic reading, for early readers.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypinyin import Style, pinyin

_TONE_MARKS = {"ˊ", "ˇ", "ˋ", "˙"}


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
