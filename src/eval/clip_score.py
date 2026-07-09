"""CLIPScore: within-spread caption-image alignment.

Reuses the same CLIP model already used for photo selection in
src/stages/stage0_select.py's clip_cluster_select.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"


def compute_clip_score(
    image_path: str,
    caption: str,
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> float:
    """Cosine similarity between an image and its caption, in [-1.0, 1.0]."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    text = tokenizer([caption])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return float((image_features @ text_features.T).item())


def score_spreads(
    illustration_paths: list[str],
    pages: list[str],
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> list[float]:
    """Within-spread relevance: one CLIPScore per spread (caption vs its own illustration)."""
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    scores = [
        compute_clip_score(path, caption, model_name, pretrained)
        for path, caption in zip(illustration_paths, pages)
    ]
    logger.info(
        "Computed within-spread CLIPScores",
        extra={"count": len(scores), "mean": float(np.mean(scores)) if scores else None},
    )
    return scores
