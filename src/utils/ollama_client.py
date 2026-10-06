"""Self-hosted Ollama backend (e.g. Gemma 4 on the lab server).

Mirrors the two entry points of `gemini_client` so `llm_client` can dispatch
between them: `generate_text` and `describe_image`. Talks to Ollama's
`/api/chat` endpoint over plain HTTP; the host comes from `OLLAMA_HOST`.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

import httpx
import pillow_heif
from PIL import Image

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()

DEFAULT_HOST = "http://localhost:11434"
# A 31B model on a shared GPU takes ~15 s for a short image description and
# noticeably longer when another job holds the card.
REQUEST_TIMEOUT_S = 300.0
# Ollama decodes images with Go's image libraries, which do not read HEIC;
# anything outside this set is re-encoded as JPEG before upload.
_PASSTHROUGH_SUFFIXES = {".jpg", ".jpeg", ".png"}
_JSON_SCHEMA_BY_TYPE: dict[object, dict] = {
    list[str]: {"type": "array", "items": {"type": "string"}},
}


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def _chat(
    model: str,
    messages: list[dict],
    *,
    max_output_tokens: int,
    thinking_budget: int,
    response_format: dict | str | None = None,
) -> str:
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": thinking_budget > 0,
        "options": {"temperature": 0.0, "num_predict": max_output_tokens},
    }
    if response_format is not None:
        body["format"] = response_format
    response = httpx.post(f"{_host()}/api/chat", json=body, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return (response.json()["message"]["content"] or "").strip()


def generate_text(
    prompt: str,
    model: str,
    max_output_tokens: int = 512,
    response_schema: type | None = None,
    thinking_budget: int = 0,
) -> str:
    """Single-turn text generation; same contract as `gemini_client.generate_text`."""
    response_format: dict | str | None = None
    if response_schema is not None:
        response_format = _JSON_SCHEMA_BY_TYPE.get(response_schema, "json")
    return _chat(
        model,
        [{"role": "user", "content": prompt}],
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
        response_format=response_format,
    )


def describe_image(
    image_path: str,
    prompt: str,
    model: str,
    max_output_tokens: int = 256,
    thinking_budget: int = 0,
) -> str:
    """Multimodal: one image + prompt; same contract as `gemini_client.describe_image`."""
    image_b64 = base64.b64encode(_uploadable_bytes(image_path)).decode()
    return _chat(
        model,
        [{"role": "user", "content": prompt, "images": [image_b64]}],
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
    )


def _uploadable_bytes(image_path: str) -> bytes:
    path = Path(image_path)
    if path.suffix.lower() in _PASSTHROUGH_SUFFIXES:
        return path.read_bytes()
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
