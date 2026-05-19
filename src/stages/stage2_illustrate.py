from __future__ import annotations
import torch
from pathlib import Path
from PIL import Image
from src.utils.prompt_templates import SD_PROMPT_TEMPLATE
from src.utils.image_utils import resize_for_sd, pil_to_rgb


def _load_sd_pipeline(base_model: str):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = None
    return pipe


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

    pipe = _load_sd_pipeline(config["base_model"])

    style_image = None
    ip_model = None
    if config["use_ipadapter"] and config.get("style_image_path"):
        style_image = resize_for_sd(pil_to_rgb(Image.open(config["style_image_path"])))
        from ip_adapter import IPAdapter
        ip_model = IPAdapter(pipe, "ip_adapter/ip-adapter_sd15.bin", "cuda")

    negative = "blurry, ugly, bad anatomy, watermark, text, signature"
    saved_paths: list[str] = []

    for i, scene in enumerate(scenes):
        prompt = SD_PROMPT_TEMPLATE.format(scene=scene, style=style)

        if ip_model is not None and style_image is not None:
            images = ip_model.generate(
                pil_image=style_image,
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=30,
                guidance_scale=7.5,
            )
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=30,
                guidance_scale=7.5,
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
    """Returns: {"illustration_paths": list[str]}"""
    paths = generate_illustrations(descriptions, style, config, output_dir)
    return {"illustration_paths": paths}
