"""Hosted FLUX.1-dev illustration backend (fal.ai).

Mirrors the local Stable Diffusion 1.5 backend in `stage2_illustrate.py` so
`run_stage2` can dispatch between them on `stage2.backend`. FLUX renders faces,
hands and scene context far more reliably than SD 1.5, at the cost of a per-image
API fee, so it is the preferred backend on machines without a large GPU.

Auth: `FAL_KEY` is read from the repo-root `.env` (env vars take precedence).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

from src.utils.prompt_templates import SD_PROMPT_TEMPLATE
from src.utils.styles import StylePreset, resolve_style

logger = logging.getLogger(__name__)

# Load .env once at import so fal_client sees FAL_KEY. Existing env wins.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# Base FLUX.1-dev for plain styles; the LoRA runner when a preset ships one.
FAL_BASE_ENDPOINT = "fal-ai/flux/dev"
FAL_LORA_ENDPOINT = "fal-ai/flux-lora"
# Image-conditioned FLUX.1 Kontext (dev) runner; takes one reference image plus
# the same style LoRAs. Used when `stage2.character_reference` is not "none".
FAL_KONTEXT_ENDPOINT = "fal-ai/flux-kontext-lora"

# FLUX.1-dev sampling defaults: low CFG, no negative prompt.
INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
# Landscape ratio matching an A4 spread (two 210mm-wide portrait pages at 297mm
# height, ideally ~1.4141:1). Rounded to multiples of 32 for FLUX; actual ratio
# (1408/992 ≈ 1.4194:1) is close but slightly wider than ideal.
SPREAD_IMAGE_SIZE = {"width": 1408, "height": 992}
LORA_SCALE = 1.0

# Character/setting continuity: cap how much of the combined per-photo
# descriptions gets folded into every spread's prompt, so it stays a light
# continuity nudge rather than crowding out that spread's own scene.
CHARACTER_REF_WORD_CAP = 60

# FLUX has no negative-prompt support (see sampling defaults above), so the
# common "extra hand"/bad-anatomy artifact in busy multi-child scenes has to
# be steered with positive reinforcement instead. Reduces but does not
# guarantee eliminating the artifact.
ANATOMY_QUALITY_HINT = (
    "anatomically correct hands with five fingers each, natural body proportions, "
    "no extra limbs"
)

# Same no-negative-prompt limitation produces illegible pseudo-text on signage
# and hallucinated watermark/signature marks in image corners. Positive
# reinforcement is the only lever available here too.
TEXT_QUALITY_HINT = "plain signage without legible text, no watermark, no signature"

# Kontext reads the prompt as an edit instruction for the reference image, so
# a bare scene description risks a near-copy of the reference. Each condition
# states what the reference is for; the rest of the prompt is unchanged. The
# previous-spread wording is deliberately blunt: a softer "draw a new scene with
# the same characters" kept the prior castle-and-fireworks on every spread.
PREVIOUS_SPREAD_INSTRUCTION = (
    "Use the reference image only for the characters' appearance (faces, hair, "
    "skin tone, clothing). Do not keep its background, setting, objects or "
    "composition; draw a completely different scene. Scene:"
)
SOURCE_PHOTO_INSTRUCTION = (
    "Redraw this photo as an illustration, keeping the people's appearance. Scene:"
)
REFERENCE_INSTRUCTIONS = {
    "previous_spread": PREVIOUS_SPREAD_INSTRUCTION,
    "source_photo": SOURCE_PHOTO_INSTRUCTION,
}

# FLUX occasionally returns an all-black frame (a NaN glitch on a bad seed).
# Re-running picks a fresh seed, so retry a few times before giving up.
MAX_ATTEMPTS = 3
# A non-black image has at least one channel brighter than this somewhere.
BLANK_MAX_LEVEL = 10


class IllustrationError(RuntimeError):
    """Raised when the fal image backend fails to produce an illustration."""


def _is_blank(image_bytes: bytes) -> bool:
    """True if the image is effectively all-black (the FLUX NaN failure mode)."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        extrema = img.convert("RGB").getextrema()
    return max(channel_max for _, channel_max in extrema) <= BLANK_MAX_LEVEL


def _build_character_description(descriptions: list[str]) -> str:
    """Combine all photo descriptions into one continuity string, deduped and
    capped, so every spread's prompt carries the same characters/setting.

    No new LLM call: reuses the per-photo descriptions already produced by
    stage0_select and passed in as `scenes`.
    """
    seen: list[str] = []
    for description in descriptions:
        description = description.strip()
        if description and description not in seen:
            seen.append(description)
    combined = " ".join(seen)
    words = combined.split()
    if len(words) > CHARACTER_REF_WORD_CAP:
        combined = " ".join(words[:CHARACTER_REF_WORD_CAP])
    return combined


def _build_prompt(scene: str, preset: StylePreset, character_description: str = "") -> str:
    """Compose the FLUX prompt.

    FLUX uses a T5 encoder (~512 tokens), so unlike the SD 1.5 path we do not trim
    the scene. The FLUX-tuned style fragment is preferred; it falls back to the SD
    fragment for presets that have not been tuned for FLUX yet. Some LoRAs also
    require a trailing trigger sentence, appended via flux_prompt_suffix.

    `character_description`, when given, is a book-wide continuity string (see
    `_build_character_description`) appended so recurring characters/settings stay
    descriptively consistent across independently generated spreads.
    """
    style_prompt = preset.flux_prompt or preset.sd_prompt
    prompt = SD_PROMPT_TEMPLATE.format(scene=scene, style_prompt=style_prompt)
    if preset.flux_prompt_suffix:
        prompt = f"{prompt} {preset.flux_prompt_suffix}"
    prompt = f"{prompt} {ANATOMY_QUALITY_HINT}, {TEXT_QUALITY_HINT}."
    if character_description:
        prompt = (
            f"{prompt} Recurring characters and setting across the book: {character_description}."
        )
    return prompt


def generate_illustrations_fal(
    scenes: list[str],
    style: str,
    config: dict,
    output_dir: str,
    source_photo_paths: list[str] | None = None,
) -> list[str]:
    """Generate one illustration per page via fal.ai hosted FLUX.1-dev.

    Mirrors `generate_illustrations` (the local SD 1.5 backend). The only config
    field read is `character_reference`: "previous_spread" conditions each
    spread after the first on the spread before it, "source_photo" conditions
    every spread on its own entry in `source_photo_paths` (one per scene).
    """
    import fal_client

    if not os.environ.get("FAL_KEY"):
        raise IllustrationError("FAL_KEY is not set; cannot reach the fal image backend")

    reference_mode = config.get("character_reference", "none")
    if reference_mode == "source_photo" and source_photo_paths is None:
        raise IllustrationError("character_reference 'source_photo' needs source_photo_paths")
    if source_photo_paths is not None and len(source_photo_paths) != len(scenes):
        raise IllustrationError("source_photo_paths must hold exactly one photo per scene")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preset = resolve_style(style)
    endpoint = FAL_LORA_ENDPOINT if preset.flux_lora else FAL_BASE_ENDPOINT
    logger.info(
        "Resolved fal illustration style",
        extra={
            "style": style,
            "preset": preset.key,
            "endpoint": endpoint,
            "lora": preset.flux_lora,
        },
    )

    character_description = _build_character_description(scenes)
    saved_paths: list[str] = []
    seeds_used: dict[str, int] = {}
    for i, scene in enumerate(scenes):
        arguments = _base_arguments(_build_prompt(scene, preset, character_description), preset)
        if source_photo_paths is not None:
            arguments["seed"] = _page_seed(Path(source_photo_paths[i]), i)
        page_endpoint = endpoint
        reference_path = _reference_image_path(
            reference_mode, i, saved_paths, source_photo_paths, output_path
        )
        if reference_path is not None:
            page_endpoint = FAL_KONTEXT_ENDPOINT
            _condition_on_reference(
                arguments,
                fal_client.upload_file(reference_path),
                REFERENCE_INSTRUCTIONS[reference_mode],
            )

        image_bytes = _generate_page(fal_client, page_endpoint, arguments, page=i + 1)
        out_path = output_path / f"page_{i + 1:02d}.png"
        out_path.write_bytes(image_bytes)
        saved_paths.append(str(out_path))
        if "seed" in arguments:
            seeds_used[out_path.name] = arguments["seed"]
        logger.info("Saved fal illustration", extra={"page": i + 1, "path": str(out_path)})

    if seeds_used:
        (output_path / "seeds.json").write_text(json.dumps(seeds_used, indent=2))
    return saved_paths


def _page_seed(photo_path: Path, page_index: int) -> int:
    """A seed fixed by album and page, shared by every character_reference
    condition so the conditions differ only in their reference image.

    The album is the photo's folder name. hashlib, not hash(), because Python
    salts str hashes per process and the seed must survive reruns.
    """
    key = f"{photo_path.parent.name}/{page_index + 1}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def _base_arguments(prompt: str, preset: StylePreset) -> dict:
    arguments: dict = {
        "prompt": prompt,
        "image_size": SPREAD_IMAGE_SIZE,
        "num_inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_images": 1,
        "enable_safety_checker": False,
    }
    if preset.flux_lora:
        arguments["loras"] = [{"path": preset.flux_lora, "scale": LORA_SCALE}]
    return arguments


def _reference_image_path(
    reference_mode: str,
    page_index: int,
    saved_paths: list[str],
    source_photo_paths: list[str] | None,
    output_path: Path,
) -> Path | None:
    """The image a spread is conditioned on, or None for text-only generation."""
    if reference_mode == "previous_spread" and saved_paths:
        return Path(saved_paths[-1])
    if reference_mode == "source_photo" and source_photo_paths is not None:
        return _save_spread_sized_photo(
            Path(source_photo_paths[page_index]),
            output_path / "references" / f"page_{page_index + 1:02d}_source.png",
        )
    return None


def _save_spread_sized_photo(photo_path: Path, out_path: Path) -> Path:
    """Centre-crop and resize a photo to the spread size.

    Kontext's "match_input" sizes the output after the reference, so an
    unprepared 500 px portrait photo would yield a small portrait illustration.
    """
    size = (SPREAD_IMAGE_SIZE["width"], SPREAD_IMAGE_SIZE["height"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(photo_path) as photo:
        upright = ImageOps.exif_transpose(photo).convert("RGB")
    ImageOps.fit(upright, size, method=Image.Resampling.LANCZOS).save(out_path)
    return out_path


def _condition_on_reference(arguments: dict, reference_url: str, instruction: str) -> None:
    """Switch text-to-image arguments to Kontext's image-conditioned form.

    Kontext has no free-form size; "match_input" keeps the reference's size,
    which is the spread size when the reference is a prior spread.
    """
    del arguments["image_size"]
    arguments["image_url"] = reference_url
    arguments["resolution_mode"] = "match_input"
    arguments["prompt"] = f"{instruction} {arguments['prompt']}"


def _generate_page(fal_client, endpoint: str, arguments: dict, page: int) -> bytes:
    """Run one fal generation, retrying when FLUX returns an all-black frame.

    A fixed seed would reproduce the same black frame, so each retry moves to
    the next seed; `arguments["seed"]` is left holding the seed that succeeded.
    """
    first_seed = arguments.get("seed")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if first_seed is not None:
            arguments["seed"] = first_seed + attempt - 1
        try:
            result = fal_client.run(endpoint, arguments=arguments)
            image_url = result["images"][0]["url"]
            image_bytes = urllib.request.urlopen(image_url).read()  # noqa: S310 (fal URL)
        except Exception as e:
            logger.exception(
                "fal image generation failed",
                extra={"page": page, "endpoint": endpoint, "attempt": attempt},
            )
            raise IllustrationError(f"fal failed to generate page {page}") from e

        if not _is_blank(image_bytes):
            return image_bytes

        logger.warning(
            "fal returned a blank image; retrying with a fresh seed",
            extra={"page": page, "attempt": attempt},
        )

    raise IllustrationError(
        f"fal returned a blank image for page {page} after {MAX_ATTEMPTS} attempts"
    )
