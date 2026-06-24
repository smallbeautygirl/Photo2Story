"""Fixed illustration-style presets for the storybook.

Each preset binds a human-facing label (used in the story prompt) to a curated
Stable Diffusion prompt fragment, style-specific negatives, and an optional
reference image for IP-Adapter so the whole book keeps one master's look.

Unknown style strings fall back to a free-form passthrough preset, so callers
can still experiment with ad-hoc styles without registering them first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Reference images live here as <key>.png; missing files degrade to text-only prompts.
STYLE_REFS_DIR = Path("assets/style_refs")


class StyleKey(StrEnum):
    GHIBLI = "ghibli"
    WATERCOLOR = "watercolor"
    INK_WASH = "ink_wash"
    PIXAR = "pixar"
    FLAT_PASTEL = "flat_pastel"
    CRAYON = "crayon"


@dataclass(frozen=True)
class StylePreset:
    key: str
    label: str  # human-friendly name woven into the story prompt
    sd_prompt: str  # rich positive style fragment for Stable Diffusion
    negative: str  # style-specific additions to the base negative prompt
    style_image: Path | None  # IP-Adapter reference image, if one is shipped
    # Stylisation weights. vanilla SD 1.5 + a text prompt does not reproduce a
    # master's look, so a preset can point at either a full fine-tuned checkpoint
    # (replaces base_model) or a LoRA loaded on top of base_model.
    model: str | None = None  # HF checkpoint id that replaces stage2.base_model
    lora: str | None = None  # HF LoRA repo id loaded on top of base_model


def _ref(key: StyleKey) -> Path:
    return STYLE_REFS_DIR / f"{key.value}.png"


STYLE_PRESETS: dict[str, StylePreset] = {
    StyleKey.GHIBLI: StylePreset(
        key=StyleKey.GHIBLI,
        # "ghibli style" is the trigger token the Ghibli-Diffusion checkpoint was trained on.
        label="Studio Ghibli",
        sd_prompt="ghibli style, Studio Ghibli anime, soft watercolor backgrounds, gentle light",
        negative="3d render, photorealistic, harsh shadows",
        style_image=_ref(StyleKey.GHIBLI),
        model="nitrosocke/Ghibli-Diffusion",
    ),
    StyleKey.WATERCOLOR: StylePreset(
        key=StyleKey.WATERCOLOR,
        label="soft watercolor",
        sd_prompt="soft watercolor painting, loose brush strokes, paper texture, pastel washes",
        negative="3d render, photographic, hard black outlines",
        style_image=_ref(StyleKey.WATERCOLOR),
    ),
    StyleKey.INK_WASH: StylePreset(
        key=StyleKey.INK_WASH,
        label="Chinese ink wash",
        sd_prompt="Chinese ink wash painting, sumi-e, flowing brush strokes, rice paper, gray gradients",
        negative="vivid saturated colors, 3d render, photographic",
        style_image=_ref(StyleKey.INK_WASH),
    ),
    StyleKey.PIXAR: StylePreset(
        key=StyleKey.PIXAR,
        # "modern disney style" is the trigger token of the mo-di-diffusion checkpoint.
        label="3D Pixar",
        sd_prompt="modern disney style, 3D Pixar animation, soft global illumination, smooth shading",
        negative="flat, 2d sketch, watercolor, rough pencil lines",
        style_image=_ref(StyleKey.PIXAR),
        model="nitrosocke/mo-di-diffusion",
    ),
    StyleKey.FLAT_PASTEL: StylePreset(
        key=StyleKey.FLAT_PASTEL,
        label="flat pastel",
        sd_prompt="flat vector illustration, clean geometric shapes, soft pastel palette, minimal shading",
        negative="photorealistic, 3d render, gritty texture, harsh shadows",
        style_image=_ref(StyleKey.FLAT_PASTEL),
    ),
    StyleKey.CRAYON: StylePreset(
        key=StyleKey.CRAYON,
        label="crayon drawing",
        sd_prompt="children's crayon drawing, waxy textured strokes, naive hand-drawn, bright colors",
        negative="3d render, photorealistic, digital smoothness, sharp vector lines",
        style_image=_ref(StyleKey.CRAYON),
    ),
}


def resolve_style(style: str) -> StylePreset:
    """Resolve a style string to a registered preset.

    Matching is case-insensitive and treats spaces/hyphens as underscores. Unknown
    styles become a free-form passthrough preset so ad-hoc styles still render.
    """
    key = style.strip().lower().replace(" ", "_").replace("-", "_")
    preset = STYLE_PRESETS.get(key)
    if preset is not None:
        return preset
    return StylePreset(
        key=key,
        label=style.strip(),
        sd_prompt=f"{style.strip()} art style",
        negative="",
        style_image=None,
    )


def available_styles() -> list[str]:
    """Registered style keys, for CLI help and validation messages."""
    return [k.value for k in StyleKey]
