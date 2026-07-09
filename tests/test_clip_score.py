# tests/test_clip_score.py
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip entire file if torch not available


def _make_open_clip_mock(image_features: np.ndarray, text_features: np.ndarray):
    """Fake open_clip module returning fixed image/text feature vectors."""
    mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.tensor(image_features).unsqueeze(0)
    mock_model.encode_text.return_value = torch.tensor(text_features).unsqueeze(0)

    mock_tokenizer = MagicMock(side_effect=lambda texts: torch.zeros(len(texts), 77, dtype=torch.long))

    mock_oc = MagicMock()
    mock_oc.create_model_and_transforms.return_value = (mock_model, MagicMock(), mock_preprocess)
    mock_oc.get_tokenizer.return_value = mock_tokenizer
    return mock_oc


def test_compute_clip_score_identical_vectors_scores_one(tmp_images):
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(vec, vec)}):
        from src.eval.clip_score import compute_clip_score

        score = compute_clip_score(tmp_images[0], "a red square")

    assert score == pytest.approx(1.0, abs=1e-4)


def test_compute_clip_score_orthogonal_vectors_scores_zero(tmp_images):
    image_vec = np.zeros(512, dtype=np.float32)
    image_vec[0] = 1.0
    text_vec = np.zeros(512, dtype=np.float32)
    text_vec[1] = 1.0

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(image_vec, text_vec)}):
        from src.eval.clip_score import compute_clip_score

        score = compute_clip_score(tmp_images[0], "unrelated caption")

    assert score == pytest.approx(0.0, abs=1e-4)


def test_score_spreads_returns_one_score_per_page(tmp_images):
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(vec, vec)}):
        from src.eval.clip_score import score_spreads

        scores = score_spreads(tmp_images[:3], ["a.", "b.", "c."])

    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_score_spreads_wrong_count_raises(tmp_images):
    from src.eval.clip_score import score_spreads

    with pytest.raises(AssertionError):
        score_spreads(tmp_images[:2], ["only one caption"])
