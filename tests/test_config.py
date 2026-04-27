# tests/test_config.py
import pytest
from pathlib import Path
from src.config import load_config, PipelineConfig

PROJECT_ROOT = Path(__file__).parents[1]

def test_load_demo_config():
    cfg = load_config(PROJECT_ROOT / "configs/demo.yaml")  # use Path, not str
    assert cfg["stage0"]["mode"] == "hybrid"
    assert cfg["stage0"]["k"] == 4
    assert cfg["stage1"]["use_causal_inference"] is True

def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")

def test_load_config_invalid_mode(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage0:\n  mode: invalid\n  clip_model: x\n  clip_pretrained: y\n  vlm_model: x\n  llm_model: x\n  k: 4\nstage1:\n  model: x\n  context_mode: full\n  use_causal_inference: true\nstage2:\n  base_model: x\n  use_ipadapter: true\n  use_stylealigned: false\n  style_image_path: null\nstage3:\n  output_format: pdf\n  page_layout: image_top_text_bottom\n")
    with pytest.raises(ValueError, match="stage0.mode"):
        load_config(bad)

def test_load_config_missing_stage(tmp_path):
    bad = tmp_path / "missing_stage.yaml"
    bad.write_text("stage0:\n  mode: hybrid\n")
    with pytest.raises(ValueError, match="missing required section"):
        load_config(bad)
