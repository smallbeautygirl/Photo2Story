"""Stage 1 output checks: did the story come back in the right shape, language and length?"""

from __future__ import annotations

import pytest

from src.eval.stage1_metrics import cjk_ratio, page_length_stats, stage0_result_from_selection


def test_cjk_ratio_counts_han_characters_among_letters():
    assert cjk_ratio("小寶寶看著大魚缸。") == pytest.approx(1.0)
    assert cjk_ratio("Hello world") == 0.0
    assert cjk_ratio("小魚 says hi") == pytest.approx(2 / 8)
    assert cjk_ratio("") == 0.0


def test_page_length_stats_reports_max_and_how_many_pages_exceed_the_limit():
    pages = ["一二三四五", "一二三", "", "一二三四五六七"]

    stats = page_length_stats(pages, max_chars=5)

    assert stats == {"max_chars": 7, "mean_chars": pytest.approx(15 / 4), "over_limit": 1, "empty": 1}


def test_stage0_result_from_selection_keeps_ordered_paths_and_their_descriptions():
    selection = {
        "ordered_paths": ["b.jpg", "a.jpg"],
        "selected_paths": ["a.jpg", "b.jpg"],
        "descriptions": {"a.jpg": "desc a", "b.jpg": "desc b", "c.jpg": "unused"},
        "effective_context": "a day at the zoo",
    }

    result = stage0_result_from_selection(selection)

    assert list(result["descriptions"].items()) == [("b.jpg", "desc b"), ("a.jpg", "desc a")]
    assert result["ordered_paths"] == ["b.jpg", "a.jpg"]
    assert result["effective_context"] == "a day at the zoo"
