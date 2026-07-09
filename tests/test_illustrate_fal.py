from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from src.stages import illustrate_fal
from src.stages.illustrate_fal import IllustrationError, generate_illustrations_fal


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


COLOR_PNG = _png_bytes((180, 120, 60))
BLACK_PNG = _png_bytes((0, 0, 0))


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


@pytest.fixture
def fake_fal(monkeypatch):
    """Stub fal_client.run and the image download; record the calls made.

    The downloaded bytes are keyed off the URL so a test can make fal return a
    blank image by having fake_run hand back a URL ending in 'black'.
    """
    calls: list[tuple[str, dict]] = []

    def fake_run(endpoint: str, arguments: dict):
        calls.append((endpoint, arguments))
        return {"images": [{"url": "https://fal.example/color.png"}]}

    def fake_urlopen(url: str):
        return _FakeResponse(BLACK_PNG if "black" in url else COLOR_PNG)

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("fal_client.run", fake_run)
    monkeypatch.setattr(illustrate_fal.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_ghibli_uses_lora_endpoint_with_lora_argument(fake_fal, tmp_path):
    generate_illustrations_fal(["a baby in a tub"], "ghibli", {}, str(tmp_path))

    endpoint, arguments = fake_fal[0]
    assert endpoint == illustrate_fal.FAL_LORA_ENDPOINT
    assert arguments["loras"][0]["path"] == "openfree/flux-chatgpt-ghibli-lora"


def test_crayon_uses_base_endpoint_without_lora(fake_fal, tmp_path):
    generate_illustrations_fal(["a baby in a tub"], "crayon", {}, str(tmp_path))

    endpoint, arguments = fake_fal[0]
    assert endpoint == illustrate_fal.FAL_BASE_ENDPOINT
    assert "loras" not in arguments


def test_prompt_uses_flux_fragment_and_full_scene(fake_fal, tmp_path):
    scene = "a baby wearing a neck float in a clear tub of water"
    generate_illustrations_fal([scene], "crayon", {}, str(tmp_path))

    prompt = fake_fal[0][1]["prompt"]
    assert "crayon" in prompt
    assert scene in prompt  # FLUX has a large token budget; scene is not trimmed


def test_disney_prompt_appends_trigger_suffix(fake_fal, tmp_path):
    generate_illustrations_fal(["a baby in a tub"], "disney", {}, str(tmp_path))

    prompt = fake_fal[0][1]["prompt"]
    assert prompt.startswith("This is a digital illustration from a Disney")
    # The trigger sentence is no longer necessarily the last text in the prompt:
    # the cross-spread character/setting continuity string is appended after it.
    assert "mid-20th century." in prompt


def test_one_image_saved_per_scene(fake_fal, tmp_path):
    paths = generate_illustrations_fal(["one", "two"], "crayon", {}, str(tmp_path))

    assert [Path(p).name for p in paths] == ["page_01.png", "page_02.png"]
    assert all(Path(p).exists() for p in paths)


def test_missing_fal_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(IllustrationError, match="FAL_KEY"):
        generate_illustrations_fal(["scene"], "pixar", {}, str(tmp_path))


def test_api_failure_is_wrapped(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "test-key")

    def boom(endpoint, arguments):
        raise RuntimeError("fal exploded")

    monkeypatch.setattr("fal_client.run", boom)
    with pytest.raises(IllustrationError, match="page 1"):
        generate_illustrations_fal(["scene"], "pixar", {}, str(tmp_path))


def test_blank_image_is_retried_then_succeeds(monkeypatch, tmp_path):
    """First attempt returns black, second returns a real image -> success."""
    monkeypatch.setenv("FAL_KEY", "test-key")
    urls = ["https://fal.example/black.png", "https://fal.example/color.png"]

    def fake_run(endpoint, arguments):
        return {"images": [{"url": urls.pop(0)}]}

    def fake_urlopen(url):
        return _FakeResponse(BLACK_PNG if "black" in url else COLOR_PNG)

    monkeypatch.setattr("fal_client.run", fake_run)
    monkeypatch.setattr(illustrate_fal.urllib.request, "urlopen", fake_urlopen)

    paths = generate_illustrations_fal(["scene"], "pixar", {}, str(tmp_path))

    assert not urls  # both attempts consumed
    saved = Image.open(paths[0]).convert("RGB").getextrema()
    assert max(m for _, m in saved) > illustrate_fal.BLANK_MAX_LEVEL


def test_persistent_blank_raises_after_max_attempts(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_KEY", "test-key")

    def fake_run(endpoint, arguments):
        return {"images": [{"url": "https://fal.example/black.png"}]}

    monkeypatch.setattr("fal_client.run", fake_run)
    monkeypatch.setattr(
        illustrate_fal.urllib.request, "urlopen", lambda url: _FakeResponse(BLACK_PNG)
    )

    with pytest.raises(IllustrationError, match="blank image"):
        generate_illustrations_fal(["scene"], "pixar", {}, str(tmp_path))


def test_landscape_spread_aspect_ratio_used(fake_fal, tmp_path):
    generate_illustrations_fal(["a baby in a tub"], "crayon", {}, str(tmp_path))

    _, arguments = fake_fal[0]
    assert arguments["image_size"] == illustrate_fal.SPREAD_IMAGE_SIZE
    assert illustrate_fal.SPREAD_IMAGE_SIZE["width"] > illustrate_fal.SPREAD_IMAGE_SIZE["height"]


def test_build_character_reference_dedupes_and_caps():
    from src.stages.illustrate_fal import CHARACTER_REF_WORD_CAP, _build_character_reference

    scenes = ["a red bike"] * 3 + [" ".join(f"word{i}" for i in range(100))]
    ref = _build_character_reference(scenes)

    assert ref.count("a red bike") == 1
    assert len(ref.split()) <= CHARACTER_REF_WORD_CAP


def test_prompt_includes_character_reference_from_other_pages(fake_fal, tmp_path):
    scenes = [
        "a girl in a red hat playing on a swing",
        "a girl in a red hat eating ice cream",
    ]
    generate_illustrations_fal(scenes, "crayon", {}, str(tmp_path))

    page_2_prompt = fake_fal[1][1]["prompt"]
    assert "eating ice cream" in page_2_prompt
    assert "playing on a swing" in page_2_prompt
