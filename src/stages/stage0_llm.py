"""Stage 0's language-model calls: photo descriptions (with an on-disk cache),
relevance scoring, theme inference and listwise top-k selection.

Kept apart from the selection flow in `stage0_select.py` so that module stays
about *which* photos are chosen, and this one about *what the models are asked*.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from src.utils.llm_client import describe_image, generate_text
from src.utils.prompt_templates import (
    LLM_INFER_THEME,
    LLM_SCORE_RELEVANCE,
    LLM_SELECT_TOP_K,
    VLM_DESCRIBE_PHOTO,
)

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"[-+]?\d*\.?\d+")
_DEFAULT_RELEVANCE = 0.5


def describe_photos_with_vlm(
    image_paths: list[str], model_name: str, *, cache_dir: Path | None = None
) -> dict[str, str]:
    """Phase B step 1: a text description per photo from the VLM.

    Descriptions depend only on (photo, model): temperature is 0 and the prompt carries
    no context, so with `cache_dir` set they are stored per model and reused across runs.
    """
    cache_path = _description_cache_path(cache_dir, model_name)
    cached = _load_description_cache(cache_path)
    missing = [p for p in image_paths if str(Path(p).resolve()) not in cached]
    logger.info(
        "Describing photos with VLM",
        extra={"model": model_name, "count": len(image_paths), "cached": len(image_paths) - len(missing)},
    )
    for path in missing:
        cached[str(Path(path).resolve())] = describe_image(path, VLM_DESCRIBE_PHOTO, model_name)
    if missing and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=1))
    return {p: cached[str(Path(p).resolve())] for p in image_paths}


def _description_cache_path(cache_dir: Path | None, model_name: str) -> Path | None:
    if cache_dir is None:
        return None
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return Path(cache_dir) / f"{safe_model}.json"


def _load_description_cache(cache_path: Path | None) -> dict[str, str]:
    if cache_path is None or not cache_path.is_file():
        return {}
    return json.loads(cache_path.read_text())


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
        "Scoring photo relevance with LLM",
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


def select_top_k_with_llm(
    descriptions: dict[str, str], context: str, *, k: int, model: str, seed: int
) -> list[str]:
    """llm_only Phase A: one listwise call picks k photos from every description.

    Per-photo 0-1 scores saturate (most photos score 1.0 against a broad theme), and a
    top-k sort then silently falls back to filename order. Asking for k photo numbers in
    one call avoids that; any invalid or missing picks are filled by a seeded draw.
    """
    paths = list(descriptions)
    numbered = "\n".join(f"{i + 1}. {descriptions[p]}" for i, p in enumerate(paths))
    prompt = LLM_SELECT_TOP_K.format(context=context, n=len(paths), k=k, descriptions=numbered)
    logger.info("Selecting photos with listwise LLM call", extra={"model": model, "count": len(paths), "k": k})
    raw = generate_text(prompt, model, max_output_tokens=64, response_schema=list[int])

    picked: list[str] = []
    for number in (int(m) for m in re.findall(r"\d+", raw)):
        if 1 <= number <= len(paths) and paths[number - 1] not in picked:
            picked.append(paths[number - 1])
    picked = picked[:k]
    if len(picked) < k:
        logger.warning(
            "LLM named too few valid photos; filling by seeded draw",
            extra={"raw": raw[:64], "valid": len(picked), "k": k},
        )
        remaining = [p for p in paths if p not in picked]
        picked += random.Random(seed).sample(remaining, k - len(picked))
    return picked
