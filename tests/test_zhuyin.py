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
