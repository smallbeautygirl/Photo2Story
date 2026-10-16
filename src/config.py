from typing import Literal, NotRequired, TypedDict, Union
import yaml
from pathlib import Path


class Stage0Config(TypedDict):
    mode: Literal["random", "clip_only", "llm_only", "hybrid", "hybrid_v2"]
    clip_model: str
    clip_pretrained: str
    vlm_model: str
    llm_model: str
    k: int


class Stage1Config(TypedDict):
    model: str
    context_mode: Literal["none", "keyword", "full"]
    use_causal_inference: bool
    # Controls story brevity / vocabulary; defaults to "standard" when omitted.
    reading_level: NotRequired[Literal["simple", "standard"]]


class Stage2Config(TypedDict):
    base_model: str
    use_ipadapter: bool
    use_stylealigned: bool
    style_image_path: str | None
    # Image generation backend; defaults to local SD 1.5 when omitted.
    backend: NotRequired[Literal["local", "fal"]]
    # Cross-spread character conditioning for the fal backend (see
    # docs/adr/0004-flux-kontext-character-reference.md); defaults to "none".
    character_reference: NotRequired[
        Literal["none", "previous_spread", "character_sheet", "source_photo"]
    ]
    # Text model that describes the book's recurring characters for every fal
    # prompt; without it the photo descriptions are concatenated instead.
    character_description_model: NotRequired[str]


class Stage3Config(TypedDict):
    output_format: Literal["pdf"]
    page_layout: Literal["image_top_text_bottom", "picture_book", "picture_book_spread"]
    # Optional override for the CJK font file; defaults to the bundled
    # jf-openhuninn font when omitted (see docs/adr/0003-fixed-font-across-pipeline.md).
    font_path: NotRequired[str]


class PipelineConfig(TypedDict):
    stage0: Stage0Config
    stage1: Stage1Config
    stage2: Stage2Config
    stage3: Stage3Config


_VALID_MODES = {"random", "clip_only", "llm_only", "hybrid", "hybrid_v2"}
_VALID_CONTEXT_MODES = {"none", "keyword", "full"}
_VALID_CHARACTER_REFERENCES = {"none", "previous_spread", "character_sheet", "source_photo"}


def _validate(cfg: dict) -> None:
    for stage in ("stage0", "stage1", "stage2", "stage3"):
        if stage not in cfg:
            raise ValueError(f"Config missing required section: '{stage}'")
    if cfg["stage0"].get("mode") not in _VALID_MODES:
        raise ValueError(f"stage0.mode must be one of {_VALID_MODES}")
    if cfg["stage1"].get("context_mode") not in _VALID_CONTEXT_MODES:
        raise ValueError(f"stage1.context_mode must be one of {_VALID_CONTEXT_MODES}")
    cfg["stage2"].setdefault("character_reference", "none")
    if cfg["stage2"]["character_reference"] not in _VALID_CHARACTER_REFERENCES:
        raise ValueError(
            f"stage2.character_reference must be one of {_VALID_CHARACTER_REFERENCES}"
        )


def load_config(path: Union[str, Path]) -> PipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg
