from __future__ import annotations
import logging
import torch
from pathlib import Path
from PIL import Image
from src.utils.prompt_templates import SD_PROMPT_TEMPLATE
from src.utils.image_utils import resize_for_sd, pil_to_rgb
from src.utils.styles import StylePreset, resolve_style

logger = logging.getLogger(__name__)

BASE_NEGATIVE = (
    "blurry, ugly, deformed, disfigured, mutated, malformed limbs, extra limbs, "
    "extra fingers, fused fingers, bad hands, bad anatomy, bad proportions, "
    "distorted face, deformed face, watermark, text, signature, "
    # SD 1.5 mangles tight face crops; steer toward wider, gentler framing.
    "close-up, extreme close-up, macro, giant face"
)

# A gentle, wider composition keeps faces small, which SD 1.5 renders far more reliably.
FRAMING_HINT = "wide gentle scene, full figures, soft composition"

INFERENCE_STEPS = 40
GUIDANCE_SCALE = 8.5

# CLIP encodes at most 77 tokens. The style fragment leads the prompt, so the
# scene must stay short enough that the trailing quality words survive too.
SCENE_WORD_CAP = 38


def _cap_scene(scene: str) -> str:
    """Trim an over-long VLM scene description so the full SD prompt fits CLIP's 77-token window."""
    words = scene.split()
    if len(words) <= SCENE_WORD_CAP:
        return scene
    return " ".join(words[:SCENE_WORD_CAP])


def _load_sd_pipeline(base_model: str):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = None
    return pipe


def _resolve_style_image(preset: StylePreset, config: dict) -> Path | None:
    """Pick the IP-Adapter reference image: explicit config path wins, else the preset's."""
    configured = config.get("style_image_path")
    if configured:
        return Path(configured)
    if preset.style_image and preset.style_image.exists():
        return preset.style_image
    if preset.style_image is not None:
        logger.warning(
            "Style preset reference image not found; falling back to text-only style",
            extra={"style": preset.key, "expected_path": str(preset.style_image)},
        )
    return None


def generate_illustrations(
    scenes: list[str],
    style: str,
    config: dict,
    output_dir: str,
) -> list[str]:
    """Generate one illustration per page from English scene descriptions.

    `scenes` are the VLM photo descriptions (English). Story page_text is intentionally
    not used here: SD 1.5 was trained on English captions and degrades on non-English
    prompts, so we keep the SD input in English regardless of the story language.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preset = resolve_style(style)
    effective_model = preset.model or config["base_model"]
    logger.info(
        "Resolved illustration style",
        extra={"style": style, "preset": preset.key, "model": effective_model, "lora": preset.lora},
    )

    pipe = _load_sd_pipeline(effective_model)
    if preset.lora:
        pipe.load_lora_weights(preset.lora)

    style_image = None
    ip_model = None
    style_image_path = _resolve_style_image(preset, config) if config["use_ipadapter"] else None
    if style_image_path is not None:
        style_image = resize_for_sd(pil_to_rgb(Image.open(style_image_path)))
        from ip_adapter import IPAdapter
        ip_model = IPAdapter(pipe, "ip_adapter/ip-adapter_sd15.bin", "cuda")

    negative = f"{BASE_NEGATIVE}, {preset.negative}" if preset.negative else BASE_NEGATIVE
    saved_paths: list[str] = []

    for i, scene in enumerate(scenes):
        prompt = SD_PROMPT_TEMPLATE.format(
            scene=_cap_scene(scene),
            style_prompt=f"{preset.sd_prompt}, {FRAMING_HINT}",
        )

        if ip_model is not None and style_image is not None:
            images = ip_model.generate(
                pil_image=style_image,
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images

        out_path = str(output_path / f"page_{i+1:02d}.png")
        images[0].save(out_path)
        saved_paths.append(out_path)

    del pipe
    torch.cuda.empty_cache()
    return saved_paths


def run_stage2(
    descriptions: list[str], style: str, config: dict, output_dir: str
) -> dict:
    """Returns: {"illustration_paths": list[str]}

    Dispatches to the hosted FLUX backend when `stage2.backend` is "fal";
    otherwise uses the local Stable Diffusion 1.5 pipeline.
    """
    if config.get("backend", "local") == "fal":
        from src.stages.illustrate_fal import generate_illustrations_fal

        paths = generate_illustrations_fal(descriptions, style, config, output_dir)
    else:
        paths = generate_illustrations(descriptions, style, config, output_dir)
    return {"illustration_paths": paths}
