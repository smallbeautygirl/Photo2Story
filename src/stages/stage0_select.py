from __future__ import annotations

import logging
import random
import re
from datetime import datetime

import numpy as np
import piexif
from PIL import Image
from sklearn.cluster import KMeans

from src.utils.gemini_client import describe_image, generate_text
from src.utils.prompt_templates import (
    LLM_INFER_THEME,
    LLM_SCORE_RELEVANCE,
    VLM_DESCRIBE_PHOTO,
)

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"[-+]?\d*\.?\d+")
_LOW_SCORE_THRESHOLD = 0.4
_DEFAULT_RELEVANCE = 0.5


def clip_cluster_select(
    image_paths: list[str],
    k: int,
    model_name: str,
    pretrained: str,
) -> list[str]:
    """Phase A: Select k diverse photos using CLIP embeddings + K-means."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()

    images = torch.stack([
        preprocess(Image.open(p).convert("RGB")) for p in image_paths
    ])

    with torch.no_grad():
        features = model.encode_image(images).cpu().numpy().astype(np.float32)

    features /= np.linalg.norm(features, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    selected: list[str] = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_feats = features[mask]
        center = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        local_best = int(np.argmin(dists))
        global_idx = int(np.where(mask)[0][local_best])
        selected.append(image_paths[global_idx])

    return selected


def describe_photos_with_vlm(image_paths: list[str], model_name: str) -> dict[str, str]:
    """Phase B step 1: Generate a text description for each photo via Gemini VLM."""
    logger.info(
        "Describing photos with Gemini VLM",
        extra={"model": model_name, "count": len(image_paths)},
    )
    return {
        path: describe_image(path, VLM_DESCRIBE_PHOTO, model_name)
        for path in image_paths
    }


def _parse_score(raw: str) -> float | None:
    """Extract first decimal number in [0, 1] from LLM output. Returns None if unparseable."""
    match = _SCORE_RE.search(raw)
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(0))))
    except ValueError:
        return None


def score_relevance_with_llm(
    descriptions: dict[str, str], context: str, model_name: str
) -> dict[str, float]:
    """Phase B step 2: Score each description's relevance to user context."""
    logger.info(
        "Scoring photo relevance with Gemini LLM",
        extra={"model": model_name, "count": len(descriptions), "context": context},
    )
    scores: dict[str, float] = {}
    for path, desc in descriptions.items():
        prompt = LLM_SCORE_RELEVANCE.format(context=context, description=desc)
        raw = generate_text(prompt, model_name, max_output_tokens=8)
        parsed = _parse_score(raw)
        if parsed is None:
            logger.warning(
                "Could not parse relevance score; defaulting to 0.5",
                extra={"path": path, "raw": raw[:64]},
            )
            parsed = _DEFAULT_RELEVANCE
        scores[path] = parsed
    return scores


def infer_theme_from_descriptions(
    descriptions: dict[str, str], model_name: str
) -> str:
    """Infer a short theme from VLM descriptions; used when user context is empty."""
    if not descriptions:
        return ""
    desc_block = "\n".join(f"- {d}" for d in descriptions.values())
    prompt = LLM_INFER_THEME.format(k=len(descriptions), descriptions=desc_block)
    logger.info(
        "Inferring theme from photo descriptions",
        extra={"model": model_name, "count": len(descriptions)},
    )
    raw = generate_text(prompt, model_name, max_output_tokens=32)
    return raw.strip().strip('"').strip("'")


def sort_by_exif(image_paths: list[str]) -> list[str]:
    """Phase C: Sort by EXIF DateTimeOriginal; images without EXIF go to end."""
    def get_dt(p: str) -> datetime | None:
        try:
            exif = piexif.load(p)
        except (FileNotFoundError, piexif.InvalidImageDataError, ValueError) as e:
            logger.warning("Could not load EXIF", extra={"path": p, "error": str(e)})
            return None
        raw = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
        if not raw:
            return None
        try:
            return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S")
        except (ValueError, AttributeError):
            logger.warning(
                "Could not parse DateTimeOriginal",
                extra={"path": p, "raw": str(raw)[:32]},
            )
            return None

    dated = [(p, get_dt(p)) for p in image_paths]
    with_dt = sorted([(p, dt) for p, dt in dated if dt], key=lambda x: x[1])
    without_dt = [p for p, dt in dated if not dt]
    return [p for p, _ in with_dt] + without_dt


def _replace_low_score_candidates(
    candidates: list[str],
    descriptions: dict[str, str],
    scores: dict[str, float],
    image_paths: list[str],
    effective_context: str,
    config: dict,
) -> None:
    """In place: swap any candidate scoring < threshold with best-scoring pool replacement."""
    low_score_paths = [p for p in candidates if scores.get(p, 1.0) < _LOW_SCORE_THRESHOLD]
    if not low_score_paths:
        return

    pool_paths = [p for p in image_paths if p not in candidates]
    if not pool_paths:
        return

    pool_descs = describe_photos_with_vlm(pool_paths, config["vlm_model"])
    pool_scores = score_relevance_with_llm(pool_descs, effective_context, config["llm_model"])

    for bad_path in low_score_paths:
        if not pool_paths:
            break
        best_replacement = max(pool_paths, key=lambda p: pool_scores.get(p, 0))
        idx = candidates.index(bad_path)
        candidates[idx] = best_replacement
        descriptions[best_replacement] = pool_descs[best_replacement]
        del descriptions[bad_path]
        pool_paths.remove(best_replacement)


def run_stage0(image_paths: list[str], context: str, config: dict) -> dict:
    """
    Full Stage 0: select K photos, describe them, order by time.

    Returns:
        {
            "selected_paths": list[str],     # k paths after selection
            "descriptions": dict[str, str],  # path -> VLM description (ordered)
            "ordered_paths": list[str],      # final time-ordered k paths
            "effective_context": str,        # user context, or LLM-inferred theme if empty
        }
    """
    mode = config["mode"]
    k = config["k"]
    descriptions: dict[str, str] = {}

    # Phase A: candidate selection
    if mode == "random":
        candidates = random.sample(image_paths, min(k, len(image_paths)))
    elif mode == "clip_only":
        candidates = clip_cluster_select(
            image_paths, k, config["clip_model"], config["clip_pretrained"]
        )
    elif mode == "llm_only":
        all_descs = describe_photos_with_vlm(image_paths, config["vlm_model"])
        scoring_ctx = context or infer_theme_from_descriptions(all_descs, config["llm_model"])
        all_scores = score_relevance_with_llm(all_descs, scoring_ctx, config["llm_model"])
        sorted_paths = sorted(all_scores, key=lambda p: all_scores[p], reverse=True)
        candidates = sorted_paths[:k]
        descriptions = {p: all_descs[p] for p in candidates}
    elif mode == "hybrid":
        candidates = clip_cluster_select(
            image_paths, k, config["clip_model"], config["clip_pretrained"]
        )
    else:
        raise ValueError(f"Unknown stage0 mode: {mode}")

    # Phase B: describe candidates with VLM (skip if llm_only already did this)
    if mode != "llm_only":
        descriptions = describe_photos_with_vlm(candidates, config["vlm_model"])

    # Determine effective context: fall back to LLM-inferred theme when user context is empty.
    # Why: hybrid scoring + downstream stages need a non-empty topic to ground the story.
    effective_context = context
    if not effective_context and descriptions:
        effective_context = infer_theme_from_descriptions(descriptions, config["llm_model"])
        logger.info(
            "Using inferred theme as effective context",
            extra={"theme": effective_context},
        )

    # Phase B (cont.): hybrid re-rank using effective context
    if mode == "hybrid" and effective_context:
        scores = score_relevance_with_llm(descriptions, effective_context, config["llm_model"])
        _replace_low_score_candidates(
            candidates, descriptions, scores, image_paths, effective_context, config
        )

    # Phase C: order by EXIF timestamp
    ordered = sort_by_exif(candidates)
    ordered_descriptions = {p: descriptions.get(p, "") for p in ordered}

    logger.info(
        "Stage 0 complete",
        extra={
            "mode": mode,
            "k": k,
            "selected": candidates,
            "effective_context": effective_context,
        },
    )

    return {
        "selected_paths": candidates,
        "descriptions": ordered_descriptions,
        "ordered_paths": ordered,
        "effective_context": effective_context,
    }
