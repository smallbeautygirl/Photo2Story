import pytest
from unittest.mock import MagicMock, patch
from src.model_manager import ModelManager

def test_load_calls_loader():
    mgr = ModelManager()
    mock_model = MagicMock()
    loader = MagicMock(return_value=mock_model)
    result = mgr.load("vlm", loader)
    loader.assert_called_once()
    assert result is mock_model

def test_load_same_name_does_not_reload():
    mgr = ModelManager()
    loader = MagicMock(return_value=MagicMock())
    mgr.load("vlm", loader)
    mgr.load("vlm", loader)
    assert loader.call_count == 1

def test_load_different_name_unloads_first():
    mgr = ModelManager()
    model_a = MagicMock()
    model_b = MagicMock()
    mgr.load("a", lambda: model_a)
    with patch("src.model_manager.torch") as mock_torch:
        mgr.load("b", lambda: model_b)
        mock_torch.cuda.empty_cache.assert_called_once()

def test_unload_clears_state():
    mgr = ModelManager()
    mgr.load("vlm", lambda: MagicMock())
    with patch("src.model_manager.torch"):
        mgr.unload()
    assert mgr._current_name is None
    assert mgr._current_model is None
