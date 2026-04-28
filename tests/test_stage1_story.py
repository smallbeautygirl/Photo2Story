# tests/test_stage1_story.py
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

torch = pytest.importorskip("torch")
from src.stages.stage1_story import infer_causal_narrative, generate_story, run_stage1


def _make_llm_mock(decode_output: str):
    """Return fake AutoTokenizer and AutoModelForCausalLM."""
    mock_tok = MagicMock()
    mock_tok.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    mock_tok.decode.return_value = decode_output

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    return mock_transformers


def test_infer_causal_narrative_returns_string(mocker):
    mock_tr = _make_llm_mock("They went to the beach first, then had lunch.")
    mocker.patch.dict(sys.modules, {"transformers": mock_tr})
    mocker.patch("src.stages.stage1_story.torch.cuda.empty_cache")

    result = infer_causal_narrative(
        descriptions={"p1.jpg": "beach scene", "p2.jpg": "restaurant"},
        context="beach holiday",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_story_returns_k_pages(mocker):
    json_out = '["Once upon a time.", "They played all day.", "Finally they went home."]'
    mock_tr = _make_llm_mock(json_out)
    mocker.patch.dict(sys.modules, {"transformers": mock_tr})
    mocker.patch("src.stages.stage1_story.torch.cuda.empty_cache")

    result = generate_story(
        descriptions={"p1.jpg": "beach", "p2.jpg": "hotel", "p3.jpg": "food"},
        narrative="Fun trip narrative.",
        context="holiday",
        style="watercolor",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    assert len(result) == 3
    assert all(isinstance(p, str) for p in result)


def test_generate_story_fallback_on_bad_json(mocker):
    """If LLM returns invalid JSON, fall back to line splitting."""
    mock_tr = _make_llm_mock("Page one text.\nPage two text.\nPage three text.")
    mocker.patch.dict(sys.modules, {"transformers": mock_tr})
    mocker.patch("src.stages.stage1_story.torch.cuda.empty_cache")

    result = generate_story(
        descriptions={"p1.jpg": "a", "p2.jpg": "b", "p3.jpg": "c"},
        narrative="arc",
        context="ctx",
        style="anime",
        model_name="Qwen/Qwen2.5-7B-Instruct",
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
    config = {"model": "Qwen/Qwen2.5-7B-Instruct", "context_mode": "full", "use_causal_inference": True}

    result = run_stage1(stage0_result, context="beach holiday", style="watercolor", config=config)

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
    config = {"model": "Qwen/Qwen2.5-7B-Instruct", "context_mode": "full", "use_causal_inference": False}

    result = run_stage1(stage0_result, context="trip", style="anime", config=config)

    assert result["narrative"] == ""
    mock_narrative.assert_not_called()
    mock_story.assert_called_once()
