from typing import TypedDict, Literal
import yaml
from pathlib import Path


class Stage0Config(TypedDict):
    mode: Literal["random", "clip_only", "llm_only", "hybrid"]
    clip_model: str
    clip_pretrained: str
    vlm_model: str
    llm_model: str
    k: int


class Stage1Config(TypedDict):
    model: str
    context_mode: Literal["none", "keyword", "full"]
    use_causal_inference: bool


class Stage2Config(TypedDict):
    base_model: str
    use_ipadapter: bool
    use_stylealigned: bool
    style_image_path: str | None


class Stage3Config(TypedDict):
    output_format: Literal["pdf"]
    page_layout: Literal["image_top_text_bottom"]


class PipelineConfig(TypedDict):
    stage0: Stage0Config
    stage1: Stage1Config
    stage2: Stage2Config
    stage3: Stage3Config


def load_config(path: str) -> PipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)
