"""Dispatch text / vision calls to the configured LLM backend.

Stages import `generate_text` and `describe_image` from here instead of from a
specific client. The backend is chosen per call from the model name (Gemini
names go to Vertex AI, anything else to the self-hosted Ollama server), or
forced for every call with the `LLM_BACKEND` environment variable.
"""

from __future__ import annotations

import os
from types import ModuleType

from src.utils import gemini_client, ollama_client

GEMINI_MODEL_PREFIX = "gemini"
_BACKENDS: dict[str, ModuleType] = {"gemini": gemini_client, "ollama": ollama_client}


def backend_for(model: str) -> str:
    forced = os.environ.get("LLM_BACKEND")
    if forced:
        return forced
    return "gemini" if model.startswith(GEMINI_MODEL_PREFIX) else "ollama"


def _module_for(model: str) -> ModuleType:
    backend = backend_for(model)
    try:
        return _BACKENDS[backend]
    except KeyError as e:
        raise ValueError(
            f"Unknown LLM backend {backend!r}; expected one of {sorted(_BACKENDS)}"
        ) from e


def generate_text(prompt: str, model: str, **kwargs) -> str:
    return _module_for(model).generate_text(prompt, model, **kwargs)


def describe_image(image_path: str, prompt: str, model: str, **kwargs) -> str:
    return _module_for(model).describe_image(image_path, prompt, model, **kwargs)
