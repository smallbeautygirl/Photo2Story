"""Ollama backend: same two entry points as gemini_client, over the Ollama /api/chat HTTP API."""

from __future__ import annotations

import base64
import json

import pytest
from PIL import Image

from src.utils import ollama_client


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"role": "assistant", "content": self._content}}


@pytest.fixture
def captured_post(mocker):
    """Patch httpx.post; return the list of request bodies sent."""
    bodies: list[dict] = []

    def fake_post(url, *, json: dict, timeout):
        bodies.append({"url": url, **json})
        return _FakeResponse(fake_post.reply)

    fake_post.reply = "  A toddler at a fish tank.  "
    mocker.patch.object(ollama_client.httpx, "post", side_effect=fake_post)
    return bodies, fake_post


def test_describe_image_sends_base64_image_with_thinking_off(tmp_path, captured_post, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:11434")
    bodies, _ = captured_post
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(photo)

    text = ollama_client.describe_image(str(photo), "Describe.", "gemma4:31b", max_output_tokens=64)

    body = bodies[0]
    assert body["url"] == "http://ollama.test:11434/api/chat"
    assert body["model"] == "gemma4:31b"
    assert body["think"] is False
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 64}
    assert body["messages"][0]["content"] == "Describe."
    assert base64.b64decode(body["messages"][0]["images"][0]) == photo.read_bytes()
    assert text == "A toddler at a fish tank."


def test_generate_text_with_a_schema_never_sends_a_format_grammar(captured_post):
    """Ollama's grammar-constrained output hangs Gemma on CJK text (a 4-page Chinese story
    ran past 180 s and blocked the shared server; the same prompt without `format`
    returned valid fenced JSON in 53 s). Schemas are enforced by the prompt and the
    callers' parsers instead, so the request must carry no `format` at all."""
    bodies, fake_post = captured_post
    fake_post.reply = '```json\n["a", "b"]\n```'

    text = ollama_client.generate_text("Two words.", "gemma4:31b", response_schema=list[str])

    assert "format" not in bodies[0]
    assert json.loads(text.strip("`").removeprefix("json").strip()) == ["a", "b"]


def test_generate_text_without_schema_sends_no_format(captured_post):
    bodies, _ = captured_post

    ollama_client.generate_text("Say hi.", "gemma4:31b", max_output_tokens=8, thinking_budget=100)

    assert "format" not in bodies[0]
    assert bodies[0]["think"] is True
    assert bodies[0]["options"]["num_predict"] == 8


def test_generate_text_with_int_list_schema_also_sends_no_format(captured_post):
    bodies, fake_post = captured_post
    fake_post.reply = "[3, 1]"

    ollama_client.generate_text("Pick two.", "gemma4:31b", response_schema=list[int])

    assert "format" not in bodies[0]
