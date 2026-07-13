# tests/test_zhuyin.py
from src.utils.zhuyin import ZhuyinChar, annotate


def test_annotate_returns_one_entry_per_character():
    text = "陶樂蒂的開學日"
    result = annotate(text)
    assert len(result) == len(text)
    assert [zc.char for zc in result] == list(text)


def test_annotate_extracts_all_five_tones():
    # 媽(1st,no mark) 麻(2nd,ˊ) 馬(3rd,ˇ) 罵(4th,ˋ) 嗎(neutral,˙)
    result = annotate("媽麻馬罵嗎")
    assert result[0] == ZhuyinChar(char="媽", main="ㄇㄚ", tone_mark="")
    assert result[1] == ZhuyinChar(char="麻", main="ㄇㄚ", tone_mark="ˊ")
    assert result[2] == ZhuyinChar(char="馬", main="ㄇㄚ", tone_mark="ˇ")
    assert result[3] == ZhuyinChar(char="罵", main="ㄇㄚ", tone_mark="ˋ")
    assert result[4] == ZhuyinChar(char="嗎", main="ㄇㄚ", tone_mark="˙")


def test_annotate_passes_through_punctuation_and_latin():
    result = annotate("我愛Hi world的開學日123")
    assert len(result) == len("我愛Hi world的開學日123")
    non_hanzi = [zc for zc in result if zc.char in "Hi world123"]
    assert all(zc.main == "" and zc.tone_mark == "" for zc in non_hanzi)
    # the Hanzi around the passthrough run still got real readings
    assert result[0] == ZhuyinChar(char="我", main="ㄨㄛ", tone_mark="ˇ")


def test_annotate_empty_string_returns_empty_list():
    assert annotate("") == []


def test_annotate_strips_leading_and_trailing_whitespace():
    """Matches stage3_assemble._wrap_to_width's own internal .strip(), so the
    plain-text and Zhuyin-annotated paths treat surrounding whitespace the
    same way."""
    result = annotate("  陶樂蒂  ")
    assert [zc.char for zc in result] == list("陶樂蒂")


def test_annotate_uses_particle_reading_for_zhe_after_verb():
    """著 defaults to zhu4 ("write/compose") in pypinyin's dictionary even in
    the extremely common "V著" continuous-aspect grammatical particle
    pattern -- confirmed wrong for narrative sentences like "看著" (looking
    at), which a real generated storybook page used."""
    result = annotate("看著")
    assert result[1] == ZhuyinChar(char="著", main="ㄓㄜ", tone_mark="˙")


def test_annotate_keeps_correct_reading_for_zhu_compounds():
    """A short allow-list protects the compounds where 著 legitimately keeps
    a non-particle reading, so the particle-reading override above doesn't
    regress these."""
    assert annotate("著名")[0] == ZhuyinChar(char="著", main="ㄓㄨ", tone_mark="ˋ")
    assert annotate("著急")[0] == ZhuyinChar(char="著", main="ㄓㄠ", tone_mark="ˊ")


def test_annotate_uses_neutral_tone_for_reduplicated_address_terms():
    """爸爸/寶寶/狗狗/奶奶 conventionally take neutral tone on the second
    syllable in everyday Mandarin (confirmed wrong in a real generated
    storybook using "寶寶"); pypinyin's default dictionary only gets this
    right for some reduplicated terms (媽媽, 哥哥), not others."""
    result = annotate("寶寶")
    assert result[0] == ZhuyinChar(char="寶", main="ㄅㄠ", tone_mark="ˇ")
    assert result[1] == ZhuyinChar(char="寶", main="ㄅㄠ", tone_mark="˙")
