from __future__ import annotations

import logging
import random
from datetime import datetime

import numpy as np
import pillow_heif
from PIL import Image
from sklearn.cluster import KMeans

from src.stages.stage0_llm import (
    _parse_score,  # re-exported for tests
    describe_photos_with_vlm,
    infer_theme_from_descriptions,
    score_relevance_with_llm,
    select_top_k_with_llm,
)

logger = logging.getLogger(__name__)

# Registers HEIC/HEIF as a Pillow-openable format, since iPhones export photos in
# this format by default and clip_cluster_select's CLIP embedding step needs
# Image.open() to work on them directly.
pillow_heif.register_heif_opener()

# EXIF tag ids (see the EXIF 2.3 spec): the Exif sub-IFD pointer, DateTimeOriginal, DateTime.
_EXIF_IFD_TAG = 0x8769
_DATETIME_ORIGINAL_TAG = 36867
_DATETIME_TAG = 306
_LOW_SCORE_THRESHOLD = 0.4


def encode_clip_features(image_paths: list[str], model_name: str, pretrained: str) -> np.ndarray:
    """L2-normalised CLIP image embeddings, one row per photo."""
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

    return features / np.linalg.norm(features, axis=1, keepdims=True)


def kmeans_select(
    features: np.ndarray, image_paths: list[str], k: int
) -> tuple[list[str], dict[str, int]]:
    """Phase A: K-means over CLIP features; pick the photo nearest each cluster centre.

    Returns the k selected paths and every photo's cluster label, so callers can
    see which clusters a selection covers.
    """
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

    cluster_labels = {p: int(label) for p, label in zip(image_paths, labels, strict=True)}
    return selected, cluster_labels


def clip_cluster_select(
    image_paths: list[str],
    k: int,
    model_name: str,
    pretrained: str,
) -> list[str]:
    """Select k diverse photos using CLIP embeddings + K-means."""
    features = encode_clip_features(image_paths, model_name, pretrained)
    selected, _ = kmeans_select(features, image_paths, k)
    return selected


def sort_by_exif(image_paths: list[str]) -> list[str]:
    """Phase C: Sort by EXIF DateTimeOriginal; images without EXIF go to end."""
    def get_dt(p: str) -> datetime | None:
        # PIL rather than piexif: piexif only parses JPEG/TIFF, so every iPhone HEIC
        # would be treated as undated and pushed to the end of the book.
        try:
            with Image.open(p) as image:
                exif = image.getexif()
        except (FileNotFoundError, OSError) as e:
            logger.warning("Could not load EXIF", extra={"path": p, "error": str(e)})
            return None
        raw = exif.get_ifd(_EXIF_IFD_TAG).get(_DATETIME_ORIGINAL_TAG) or exif.get(_DATETIME_TAG)
        if not raw:
            return None
        try:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
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
    *,
    all_descriptions: dict[str, str],
) -> list[dict[str, str | float]]:
    """In place: swap any candidate scoring < threshold with best-scoring pool replacement.

    Pool scores are merged into `scores` and pool descriptions into `all_descriptions`;
    returns one record per swap made.
    """
    low_score_paths = [p for p in candidates if scores.get(p, 1.0) < _LOW_SCORE_THRESHOLD]
    if not low_score_paths:
        return []

    pool_paths = [p for p in image_paths if p not in candidates]
    if not pool_paths:
        return []

    pool_descs = describe_photos_with_vlm(
        pool_paths, config["vlm_model"], cache_dir=config.get("description_cache_dir")
    )
    pool_scores = score_relevance_with_llm(pool_descs, effective_context, config["llm_model"])
    scores.update(pool_scores)
    all_descriptions.update(pool_descs)

    replacements: list[dict[str, str | float]] = []
    for bad_path in low_score_paths:
        if not pool_paths:
            break
        best_replacement = max(pool_paths, key=lambda p: pool_scores.get(p, 0))
        idx = candidates.index(bad_path)
        candidates[idx] = best_replacement
        descriptions[best_replacement] = pool_descs[best_replacement]
        del descriptions[bad_path]
        pool_paths.remove(best_replacement)
        replacements.append({
            "removed": bad_path,
            "removed_score": scores[bad_path],
            "added": best_replacement,
            "added_score": pool_scores[best_replacement],
        })

    logger.info("Replaced low-relevance candidates", extra={"replacements": replacements})
    return replacements


def run_stage0(image_paths: list[str], context: str, config: dict) -> dict:
    """
    Full Stage 0: select K photos, describe them, order by time.

    Returns:
        {
            "selected_paths": list[str],     # k paths after selection
            "descriptions": dict[str, str],  # path -> VLM description (ordered)
            "ordered_paths": list[str],      # final time-ordered k paths
            "effective_context": str,        # user context, or LLM-inferred theme if empty
            "diagnostics": {                 # why these photos: for RQ1 analysis
                "cluster_labels": dict[str, int],   # every photo's CLIP cluster (clip_only/hybrid)
                "scores": dict[str, float],         # LLM relevance of every scored photo
                "replacements": list[dict],         # low-score swaps made in hybrid mode
            },
        }
    """
    mode = config["mode"]
    k = config["k"]
    descriptions: dict[str, str] = {}
    cluster_labels: dict[str, int] = {}
    scores: dict[str, float] = {}
    replacements: list[dict[str, str | float]] = []
    all_descriptions: dict[str, str] = {}

    # Phase A: candidate selection
    if mode == "random":
        candidates = random.sample(image_paths, min(k, len(image_paths)))
    elif mode in ("clip_only", "hybrid"):
        features = encode_clip_features(image_paths, config["clip_model"], config["clip_pretrained"])
        candidates, cluster_labels = kmeans_select(features, image_paths, k)
    elif mode == "llm_only":
        all_descriptions = describe_photos_with_vlm(
            image_paths, config["vlm_model"], cache_dir=config.get("description_cache_dir")
        )
        scoring_ctx = context or infer_theme_from_descriptions(all_descriptions, config["llm_model"])
        # Per-photo scores are kept for diagnostics only; selection is the listwise call.
        scores = score_relevance_with_llm(all_descriptions, scoring_ctx, config["llm_model"])
        candidates = select_top_k_with_llm(
            all_descriptions, scoring_ctx, k=k, model=config["llm_model"], seed=config.get("seed", 42)
        )
        descriptions = {p: all_descriptions[p] for p in candidates}
    else:
        raise ValueError(f"Unknown stage0 mode: {mode}")

    # Phase B: describe candidates with VLM (skip if llm_only already did this)
    if mode != "llm_only":
        descriptions = describe_photos_with_vlm(
            candidates, config["vlm_model"], cache_dir=config.get("description_cache_dir")
        )
        all_descriptions.update(descriptions)

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
        replacements = _replace_low_score_candidates(
            candidates, descriptions, scores, image_paths, effective_context, config,
            all_descriptions=all_descriptions,
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
        "diagnostics": {
            "cluster_labels": cluster_labels,
            "scores": scores,
            "replacements": replacements,
            "descriptions": all_descriptions,
        },
    }
