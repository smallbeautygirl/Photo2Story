# tests/test_stage1_story.py
from src.stages.stage1_story import infer_causal_narrative, generate_story, run_stage1


def test_infer_causal_narrative_returns_string(mocker):
    mocker.patch(
        "src.stages.stage1_story.generate_text",
        return_value="They went to the beach first, then had lunch.",
    )
    result = infer_causal_narrative(
        descriptions={"p1.jpg": "beach scene", "p2.jpg": "restaurant"},
        context="beach holiday",
        model_name="gemini-2.5-flash",
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_story_returns_k_pages(mocker):
    mocker.patch(
        "src.stages.stage1_story.generate_text",
        return_value='["Once upon a time.", "They played all day.", "Finally they went home."]',
    )

    result = generate_story(
        descriptions={"p1.jpg": "beach", "p2.jpg": "hotel", "p3.jpg": "food"},
        narrative="Fun trip narrative.",
        context="holiday",
        style="ghibli",
        language="en",
        model_name="gemini-2.5-flash",
    )
    assert len(result) == 3
    assert all(isinstance(p, str) for p in result)


def test_generate_story_strips_markdown_code_fence(mocker):
    """LLM often wraps the JSON array in ```json ... ``` fences; parser must unwrap it."""
    mocker.patch(
        "src.stages.stage1_story.generate_text",
        return_value=(
            "**Page text:**\n"
            "```json\n"
            '["Once upon a time.", "They played all day.", "Finally they went home."]\n'
            "```"
        ),
    )

    result = generate_story(
        descriptions={"p1.jpg": "a", "p2.jpg": "b", "p3.jpg": "c"},
        narrative="arc",
        context="ctx",
        style="ghibli",
        language="en",
        model_name="gemini-2.5-flash",
    )
    assert result == [
        "Once upon a time.",
        "They played all day.",
        "Finally they went home.",
    ]


def test_generate_story_fallback_on_bad_json(mocker):
    """If LLM returns invalid JSON, fall back to line splitting."""
    mocker.patch(
        "src.stages.stage1_story.generate_text",
        return_value="Page one text.\nPage two text.\nPage three text.",
    )

    result = generate_story(
        descriptions={"p1.jpg": "a", "p2.jpg": "b", "p3.jpg": "c"},
        narrative="arc",
        context="ctx",
        style="anime",
        language="zh-tw",
        model_name="gemini-2.5-flash",
    )
    assert len(result) == 3


def test_run_stage1_with_causal_inference(mocker):
    stage0_result = {
        "ordered_paths": ["p1.jpg", "p2.jpg", "p3.jpg"],
        "descriptions": {"p1.jpg": "beach", "p2.jpg": "hotel", "p3.jpg": "food"},
    }
    mock_narrative = mocker.patch(
        "src.stages.stage1_story.infer_causal_narrative", return_value="They had a great trip."
    )
    mock_story = mocker.patch(
        "src.stages.stage1_story.generate_story",
        return_value=["Page 1.", "Page 2.", "Page 3."],
    )
    config = {"model": "gemini-2.5-flash", "context_mode": "full", "use_causal_inference": True}

    result = run_stage1(
        stage0_result, context="beach holiday", style="ghibli", language="en", config=config
    )

    assert result["pages"] == ["Page 1.", "Page 2.", "Page 3."]
    assert result["narrative"] == "They had a great trip."
    mock_narrative.assert_called_once()
    mock_story.assert_called_once()


def test_run_stage1_without_causal_inference(mocker):
    stage0_result = {
        "ordered_paths": ["p1.jpg", "p2.jpg"],
        "descriptions": {"p1.jpg": "beach", "p2.jpg": "hotel"},
    }
    mock_narrative = mocker.patch("src.stages.stage1_story.infer_causal_narrative")
    mock_story = mocker.patch(
        "src.stages.stage1_story.generate_story", return_value=["Page 1.", "Page 2."]
    )
    config = {"model": "gemini-2.5-flash", "context_mode": "full", "use_causal_inference": False}

    result = run_stage1(
        stage0_result, context="trip", style="anime", language="zh-tw", config=config
    )

    assert result["narrative"] == ""
    mock_narrative.assert_not_called()
    mock_story.assert_called_once()
