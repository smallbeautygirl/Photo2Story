"""Backend dispatch: Gemini model names go to Vertex AI, anything else to Ollama,
unless LLM_BACKEND forces one."""

from __future__ import annotations

import pytest

from src.utils import llm_client


def test_gemini_model_names_route_to_gemini(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    assert llm_client.backend_for("gemini-2.5-flash") == "gemini"


def test_other_model_names_route_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    assert llm_client.backend_for("gemma4:31b-it-q4_K_M") == "ollama"
    assert llm_client.backend_for("qwen3-vl:32b") == "ollama"


def test_env_override_wins_over_model_name(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    assert llm_client.backend_for("gemini-2.5-flash") == "ollama"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")

    with pytest.raises(ValueError, match="openai"):
        llm_client.generate_text("hi", "gemini-2.5-flash")


def test_calls_are_forwarded_to_the_chosen_backend(monkeypatch, mocker):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    gemini_generate = mocker.patch("src.utils.gemini_client.generate_text", return_value="g")
    ollama_describe = mocker.patch("src.utils.ollama_client.describe_image", return_value="o")

    assert llm_client.generate_text("p", "gemini-2.5-flash", max_output_tokens=8) == "g"
    assert llm_client.describe_image("x.jpg", "p", "gemma4:31b") == "o"

    gemini_generate.assert_called_once_with("p", "gemini-2.5-flash", max_output_tokens=8)
    ollama_describe.assert_called_once_with("x.jpg", "p", "gemma4:31b")
