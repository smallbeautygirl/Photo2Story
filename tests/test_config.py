# tests/test_config.py
import pytest
from src.config import load_config, PipelineConfig

def test_load_demo_config():
    cfg = load_config("configs/demo.yaml")
    assert cfg["stage0"]["mode"] == "hybrid"
    assert cfg["stage0"]["k"] == 4
    assert cfg["stage1"]["use_causal_inference"] is True

def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")
