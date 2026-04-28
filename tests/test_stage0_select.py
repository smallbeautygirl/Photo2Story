# tests/test_stage0_select.py
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

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
