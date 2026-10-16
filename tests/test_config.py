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

def test_stage3_accepts_picture_book_layout(tmp_path):
    import yaml
    from src.config import load_config

    cfg = {
        "stage0": {"mode": "random", "clip_model": "x", "clip_pretrained": "x",
                   "vlm_model": "x", "llm_model": "x", "k": 2},
        "stage1": {"model": "x", "context_mode": "none", "use_causal_inference": False},
        "stage2": {"base_model": "x", "use_ipadapter": False, "use_stylealigned": False,
                   "style_image_path": None},
        "stage3": {"output_format": "pdf", "page_layout": "picture_book_spread"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    loaded = load_config(p)

    assert loaded["stage3"]["page_layout"] == "picture_book_spread"


def _write_config(tmp_path, stage2_extra: dict) -> Path:
    import yaml

    cfg = {
        "stage0": {"mode": "random", "clip_model": "x", "clip_pretrained": "x",
                   "vlm_model": "x", "llm_model": "x", "k": 2},
        "stage1": {"model": "x", "context_mode": "none", "use_causal_inference": False},
        "stage2": {"base_model": "x", "use_ipadapter": False, "use_stylealigned": False,
                   "style_image_path": None, **stage2_extra},
        "stage3": {"output_format": "pdf", "page_layout": "picture_book_spread"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_character_reference_defaults_to_none_when_omitted(tmp_path):
    loaded = load_config(_write_config(tmp_path, {}))

    assert loaded["stage2"]["character_reference"] == "none"


@pytest.mark.parametrize("mode", ["none", "previous_spread", "character_sheet", "source_photo"])
def test_character_reference_accepts_each_condition(tmp_path, mode):
    loaded = load_config(_write_config(tmp_path, {"character_reference": mode}))

    assert loaded["stage2"]["character_reference"] == mode


def test_character_reference_rejects_unknown_condition(tmp_path):
    with pytest.raises(ValueError, match="stage2.character_reference"):
        load_config(_write_config(tmp_path, {"character_reference": "anchor_photo"}))
