# tests/test_pipeline.py
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline import StoryPipeline

MOCK_STAGE0 = {
    "selected_paths": ["a.jpg", "b.jpg"],
    "descriptions": {"a.jpg": "beach", "b.jpg": "hotel"},
    "ordered_paths": ["a.jpg", "b.jpg"],
}
MOCK_STAGE1 = {"pages": ["Page one.", "Page two."], "narrative": "Fun trip."}
MOCK_STAGE2 = {"illustration_paths": ["out/page_01.png", "out/page_02.png"]}
MOCK_STAGE3 = {"pdf_path": "out/storybook.pdf"}


def test_pipeline_run_calls_all_stages(tmp_path, demo_config, tmp_images, mocker):
    mocker.patch("src.pipeline.run_stage0", return_value=MOCK_STAGE0)
    mocker.patch("src.pipeline.run_stage1", return_value=MOCK_STAGE1)
    mocker.patch("src.pipeline.run_stage2", return_value=MOCK_STAGE2)
    mocker.patch("src.pipeline.run_stage3", return_value=MOCK_STAGE3)

    pipeline = StoryPipeline(demo_config)
    result = pipeline.run(
        image_paths=tmp_images,
        context="北海道家庭旅遊",
        style="ghibli",
        output_dir=str(tmp_path),
    )

    assert result["pdf_path"] == "out/storybook.pdf"
    assert result["pages"] == ["Page one.", "Page two."]
    assert result["narrative"] == "Fun trip."
    assert result["selected_paths"] == ["a.jpg", "b.jpg"]


def test_pipeline_passes_config_to_stages(tmp_path, demo_config, tmp_images, mocker):
    mock_s0 = mocker.patch("src.pipeline.run_stage0", return_value=MOCK_STAGE0)
    mocker.patch("src.pipeline.run_stage1", return_value=MOCK_STAGE1)
    mocker.patch("src.pipeline.run_stage2", return_value=MOCK_STAGE2)
    mocker.patch("src.pipeline.run_stage3", return_value=MOCK_STAGE3)

    pipeline = StoryPipeline(demo_config)
    pipeline.run(
        image_paths=tmp_images,
        context="test",
        style="anime",
        output_dir=str(tmp_path),
    )

    call_kwargs = mock_s0.call_args
    assert call_kwargs[0][2] == demo_config["stage0"]
