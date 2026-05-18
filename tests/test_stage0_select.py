# tests/test_stage0_select.py
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import piexif

torch = pytest.importorskip("torch")  # skip entire file if torch not available

from src.stages.stage0_select import clip_cluster_select  # noqa: E402


def _make_open_clip_mock(features_np: np.ndarray):
    """Return a fake open_clip module whose model.encode_image returns a real tensor."""
    mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.tensor(features_np)

    mock_oc = MagicMock()
    mock_oc.create_model_and_transforms.return_value = (mock_model, MagicMock(), mock_preprocess)
    return mock_oc


def test_clip_cluster_select_returns_k_paths(tmp_images):
    features = np.random.rand(5, 512).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(features)}):
        result = clip_cluster_select(tmp_images, k=3, model_name="ViT-B-32", pretrained="openai")

    assert len(result) == 3
    for path in result:
        assert path in tmp_images


def test_clip_cluster_select_no_duplicates(tmp_images):
    # Use identity-like features so each image is in its own cluster
    features = np.eye(5, 512).astype(np.float32)

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(features)}):
        result = clip_cluster_select(tmp_images, k=3, model_name="ViT-B-32", pretrained="openai")

    assert len(result) == len(set(result))


from src.stages.stage0_select import sort_by_exif, run_stage0


def test_sort_by_exif_no_exif_returns_all(tmp_images):
    """Images without EXIF should be returned unchanged (all at end)."""
    result = sort_by_exif(tmp_images)
    assert len(result) == len(tmp_images)
    assert set(result) == set(tmp_images)


def test_sort_by_exif_with_timestamps(tmp_path):
    """Images with EXIF timestamps should be sorted chronologically."""
    from PIL import Image as PILImage

    paths = []
    timestamps = ["2024:07:01 10:00:00", "2024:07:01 08:00:00", "2024:07:01 12:00:00"]
    for i, ts in enumerate(timestamps):
        p = tmp_path / f"img_{i}.jpg"
        PILImage.new("RGB", (64, 64), color=(i*80, i*80, i*80)).save(p)
        exif_dict = {"0th": {}, "Exif": {piexif.ExifIFD.DateTimeOriginal: ts.encode()}, "1st": {}, "thumbnail": None, "GPS": {}}
        piexif.insert(piexif.dump(exif_dict), str(p))
        paths.append(str(p))

    result = sort_by_exif(paths)
    # Should be ordered: img_1 (08:00) < img_0 (10:00) < img_2 (12:00)
    assert result[0] == paths[1]
    assert result[1] == paths[0]
    assert result[2] == paths[2]


def test_run_stage0_random_mode(tmp_images, demo_config, mocker):
    """random mode selects k images without calling VLM."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "random"

    mock_describe = mocker.patch("src.stages.stage0_select.describe_photos_with_vlm",
                                  return_value={p: f"desc {i}" for i, p in enumerate(tmp_images)})

    result = run_stage0(tmp_images, context="test trip", config=cfg)

    assert len(result["selected_paths"]) == cfg["k"]
    assert "descriptions" in result
    assert "ordered_paths" in result
    assert len(result["ordered_paths"]) == cfg["k"]
    mock_describe.assert_called_once()  # still called to get descriptions


def test_run_stage0_hybrid_empty_context_infers_theme(tmp_images, demo_config, mocker):
    """When context is empty, hybrid mode should infer a theme and still run scoring."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "hybrid"

    features = np.eye(5, 512).astype(np.float32)
    mock_oc = _make_open_clip_mock(features)

    mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        return_value={p: f"desc {i}" for i, p in enumerate(tmp_images[:4])},
    )
    mock_infer = mocker.patch(
        "src.stages.stage0_select.infer_theme_from_descriptions",
        return_value="a child's swimming lesson",
    )
    mock_score = mocker.patch(
        "src.stages.stage0_select.score_relevance_with_llm",
        return_value={p: 0.8 for p in tmp_images[:4]},
    )

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        result = run_stage0(tmp_images, context="", config=cfg)

    mock_infer.assert_called_once()
    mock_score.assert_called_once()
    assert result["effective_context"] == "a child's swimming lesson"


def test_parse_score_handles_decorations():
    """Score parser tolerates markdown / trailing commentary."""
    from src.stages.stage0_select import _parse_score

    assert _parse_score("0.85") == 0.85
    assert _parse_score("**0.85**") == 0.85
    assert _parse_score("0.85 (high relevance)") == 0.85
    assert _parse_score("Score: 0.42") == 0.42
    assert _parse_score("1.5") == 1.0  # clamped
    assert _parse_score("nope") is None


def test_run_stage0_hybrid_mode_calls_vlm_and_scores(tmp_images, demo_config, mocker):
    """hybrid mode calls VLM describe and LLM scoring."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "hybrid"

    features = np.eye(5, 512).astype(np.float32)
    mock_oc = _make_open_clip_mock(features)

    mock_describe = mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        return_value={p: f"desc {i}" for i, p in enumerate(tmp_images[:4])},
    )
    mock_score = mocker.patch(
        "src.stages.stage0_select.score_relevance_with_llm",
        return_value={p: 0.8 for p in tmp_images[:4]},
    )

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        result = run_stage0(tmp_images, context="beach holiday", config=cfg)

    assert len(result["ordered_paths"]) == cfg["k"]
    mock_describe.assert_called_once()
    mock_score.assert_called_once()
