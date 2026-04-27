import pytest
from PIL import Image
from pathlib import Path
import tempfile, os

@pytest.fixture
def tmp_images(tmp_path):
    """Create 5 tiny 64x64 RGB images for testing."""
    paths = []
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    for i, color in enumerate(colors):
        p = tmp_path / f"photo_{i}.jpg"
        Image.new("RGB", (64, 64), color=color).save(p)
        paths.append(str(p))
    return paths

@pytest.fixture
def demo_config():
    return {
        "stage0": {
            "mode": "hybrid",
            "clip_model": "ViT-B-32",
            "clip_pretrained": "openai",
            "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
            "llm_model": "Qwen/Qwen2.5-7B-Instruct",
            "k": 4,
        },
        "stage1": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "context_mode": "full",
            "use_causal_inference": True,
        },
        "stage2": {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "use_ipadapter": True,
            "use_stylealigned": False,
            "style_image_path": None,
        },
        "stage3": {
            "output_format": "pdf",
            "page_layout": "image_top_text_bottom",
        },
    }
