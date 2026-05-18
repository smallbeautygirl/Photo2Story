from __future__ import annotations

import json
import logging
import re

from src.utils.gemini_client import generate_text
from src.utils.prompt_templates import LLM_CAUSAL_INFERENCE, LLM_STORY_GENERATION

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_array(raw: str) -> list | None:
    """Pull a JSON array out of an LLM reply that may be wrapped in markdown fences or prose."""
    fence_match = _CODE_FENCE_RE.search(raw)
    candidate = fence_match.group(1).strip() if fence_match else raw

    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def infer_causal_narrative(
    descriptions: dict[str, str], context: str, model_name: str
) -> str:
    """Step 1: Ask LLM to describe the narrative arc across all photos."""
    k = len(descriptions)
    desc_block = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions.values()))
    prompt = LLM_CAUSAL_INFERENCE.format(k=k, context=context, descriptions=desc_block)
    logger.info("Inferring causal narrative", extra={"model": model_name, "k": k})
    return generate_text(prompt, model_name, max_output_tokens=256)


def generate_story(
    descriptions: dict[str, str],
    narrative: str,
    context: str,
    style: str,
    model_name: str,
) -> list[str]:
    """Step 2: Generate k-page story as a JSON list."""
    k = len(descriptions)
    desc_block = "\n".join(
        f"Page {i+1}: {desc}" for i, desc in enumerate(descriptions.values())
    )
    prompt = LLM_STORY_GENERATION.format(
        k=k, style=style, context=context, narrative=narrative, descriptions=desc_block
    )
    logger.info("Generating story pages", extra={"model": model_name, "k": k, "style": style})
    raw = generate_text(
        prompt,
        model_name,
        max_output_tokens=512,
        response_schema=list[str],
    )

    pages = _extract_json_array(raw)
    if pages is not None and len(pages) == k:
        return [str(p) for p in pages]

    logger.warning(
        "Story generation did not return a valid k-length JSON array; falling back to line split",
        extra={"k": k, "got": len(pages) if pages is not None else None},
    )

    lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.strip().startswith("```")]
    while len(lines) < k:
        lines.append("")
    return lines[:k]


def run_stage1(stage0_result: dict, context: str, style: str, config: dict) -> dict:
    """
    Returns:
        {"pages": list[str], "narrative": str}
    """
    descriptions = stage0_result["descriptions"]
    model_name = config["model"]
    narrative = ""

    if config["use_causal_inference"] and context:
        narrative = infer_causal_narrative(descriptions, context, model_name)

    pages = generate_story(descriptions, narrative, context, style, model_name)
    return {"pages": pages, "narrative": narrative}
