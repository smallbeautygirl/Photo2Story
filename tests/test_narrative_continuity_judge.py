# tests/test_narrative_continuity_judge.py
from src.eval.narrative_continuity_judge import (
    TAXONOMY,
    classify_transition,
    score_book_continuity,
)


def test_classify_transition_returns_taxonomy_category(mocker):
    mocker.patch(
        "src.eval.narrative_continuity_judge.generate_text",
        return_value="consequence_of_prior_action",
    )

    result = classify_transition("Page one.", "Page two.", model_name="gemini-2.5-flash")

    assert result == "consequence_of_prior_action"
    assert result in TAXONOMY


def test_classify_transition_accepts_none_classification(mocker):
    """A judge classifying a self-contained transition as 'none' must not be
    coerced into a positive category (docs/eval/reference_continuity_calibration.md)."""
    mocker.patch(
        "src.eval.narrative_continuity_judge.generate_text",
        return_value="none",
    )

    result = classify_transition("I am a giraffe.", "I am a buffalo.", model_name="gemini-2.5-flash")

    assert result == "none"


def test_classify_transition_defaults_to_none_on_unrecognized_output(mocker):
    mocker.patch(
        "src.eval.narrative_continuity_judge.generate_text",
        return_value="something the model made up",
    )

    result = classify_transition("Page one.", "Page two.", model_name="gemini-2.5-flash")

    assert result == "none"


def test_score_book_continuity_aggregates_transitions(mocker):
    mocker.patch(
        "src.eval.narrative_continuity_judge.classify_transition",
        side_effect=["recurring_character_object", "none", "consequence_of_prior_action"],
    )

    result = score_book_continuity(
        ["Page 1.", "Page 2.", "Page 3.", "Page 4."], model_name="gemini-2.5-flash"
    )

    assert result["transitions"] == [
        "recurring_character_object",
        "none",
        "consequence_of_prior_action",
    ]
    assert result["continuous_count"] == 2
    assert result["total_transitions"] == 3
