"""Thin Vertex AI Gemini client used by stage0/stage1.

Authentication: relies on Google Application Default Credentials.
Run `gcloud auth application-default login` once; the SDK picks up the
credentials at `~/.config/gcloud/application_default_credentials.json`.

Project / location are read from `.env` at the repo root, with env-var
overrides (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`). Location
falls back to `asia-east1` if neither is set.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_LOCATION = "asia-east1"

# Load .env once at import time. Existing env vars take precedence (override=False).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

_MIME_BY_EXT: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


@lru_cache(maxsize=1)
def _client():
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
    logger.info(
        "Initialising Vertex AI Gemini client",
        extra={"project": project, "location": location},
    )
    return genai.Client(vertexai=True, project=project, location=location)


def _guess_mime(path: str) -> str:
    return _MIME_BY_EXT.get(Path(path).suffix.lower(), "image/jpeg")


def generate_text(
    prompt: str,
    model: str,
    max_output_tokens: int = 512,
    response_schema: type | None = None,
    thinking_budget: int = 0,
) -> str:
    """Single-turn text generation. Returns the model's text reply (stripped).

    When `response_schema` is provided, the model is forced into JSON mode and the
    reply is a JSON document matching the schema (e.g. `list[str]`).

    `thinking_budget` defaults to 0 (disabled). Gemini 2.5 counts thinking tokens
    against `max_output_tokens`, so leaving thinking on silently truncates short
    structured outputs.
    """
    from google.genai import types

    config_kwargs: dict = {
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    response = _client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return (response.text or "").strip()


def describe_image(
    image_path: str,
    prompt: str,
    model: str,
    max_output_tokens: int = 256,
    thinking_budget: int = 0,
) -> str:
    """Multimodal: send one image + prompt, return the model's text reply."""
    from google.genai import types

    image_bytes = Path(image_path).read_bytes()
    response = _client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime(image_path)),
            prompt,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return (response.text or "").strip()
