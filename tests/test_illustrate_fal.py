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

    def fake_upload_file(path):
        return f"https://fal.example/uploaded/{Path(path).name}"

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr("fal_client.run", fake_run)
    monkeypatch.setattr("fal_client.upload_file", fake_upload_file)
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


def test_build_character_description_dedupes_and_caps():
    from src.stages.illustrate_fal import CHARACTER_REF_WORD_CAP, _build_character_description

    scenes = ["a red bike"] * 3 + [" ".join(f"word{i}" for i in range(100))]
    ref = _build_character_description(scenes)

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


def test_prompt_includes_anatomy_quality_hint(fake_fal, tmp_path):
    """FLUX has no negative-prompt support, so anatomy correctness (e.g. the
    common 'extra hand' artifact in busy multi-child scenes) has to be steered
    with positive reinforcement in every prompt."""
    generate_illustrations_fal(
        ["two children reaching toward a shark tank display"], "ghibli", {}, str(tmp_path)
    )

    prompt = fake_fal[0][1]["prompt"]
    assert illustrate_fal.ANATOMY_QUALITY_HINT in prompt


def test_prompt_includes_text_quality_hint(fake_fal, tmp_path):
    """FLUX has no negative-prompt support, so illegible signage text and
    hallucinated watermark/signature marks also have to be steered with
    positive reinforcement in every prompt."""
    generate_illustrations_fal(
        ["a family walking past shop signs on a busy street"], "ghibli", {}, str(tmp_path)
    )

    prompt = fake_fal[0][1]["prompt"]
    assert illustrate_fal.TEXT_QUALITY_HINT in prompt


PREVIOUS_SPREAD = {"backend": "fal", "character_reference": "previous_spread"}


def test_previous_spread_conditions_each_later_spread_on_the_one_before(fake_fal, tmp_path):
    generate_illustrations_fal(
        ["a girl on a swing", "a girl eating ice cream", "a girl asleep"],
        "ghibli",
        PREVIOUS_SPREAD,
        str(tmp_path),
    )

    first_endpoint, first_arguments = fake_fal[0]
    assert first_endpoint == illustrate_fal.FAL_LORA_ENDPOINT
    assert "image_url" not in first_arguments
    for page, (endpoint, arguments) in enumerate(fake_fal[1:], start=2):
        assert endpoint == illustrate_fal.FAL_KONTEXT_ENDPOINT
        assert arguments["image_url"].endswith(f"page_{page - 1:02d}.png")
        assert arguments["loras"][0]["path"] == "openfree/flux-chatgpt-ghibli-lora"


def test_previous_spread_asks_for_a_new_scene_with_the_same_characters(fake_fal, tmp_path):
    generate_illustrations_fal(
        ["a girl on a swing", "a girl eating ice cream"], "ghibli", PREVIOUS_SPREAD, str(tmp_path)
    )

    first_prompt, second_prompt = (arguments["prompt"] for _, arguments in fake_fal)
    assert not first_prompt.startswith(illustrate_fal.PREVIOUS_SPREAD_INSTRUCTION)
    assert second_prompt.startswith(illustrate_fal.PREVIOUS_SPREAD_INSTRUCTION)
    assert "a girl eating ice cream" in second_prompt


SOURCE_PHOTO = {"backend": "fal", "character_reference": "source_photo"}


def _photo(path: Path, size: tuple[int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 140, 200)).save(path)
    return str(path)


def test_source_photo_conditions_every_spread_on_its_own_photo_cropped_to_spread(
    fake_fal, monkeypatch, tmp_path
):
    uploaded_sizes: dict[str, tuple[int, int]] = {}

    def record_upload(path):
        with Image.open(path) as img:
            uploaded_sizes[Path(path).name] = img.size
        return f"https://fal.example/uploaded/{Path(path).name}"

    monkeypatch.setattr("fal_client.upload_file", record_upload)
    photos = [
        _photo(tmp_path / "album" / "landscape.jpg", (500, 375)),
        _photo(tmp_path / "album" / "portrait.jpg", (375, 500)),
    ]

    generate_illustrations_fal(
        ["a family at a castle", "a child on a ride"],
        "ghibli",
        SOURCE_PHOTO,
        str(tmp_path / "out"),
        source_photo_paths=photos,
    )

    spread = (illustrate_fal.SPREAD_IMAGE_SIZE["width"], illustrate_fal.SPREAD_IMAGE_SIZE["height"])
    assert len(uploaded_sizes) == 2
    assert set(uploaded_sizes.values()) == {spread}
    for page, (endpoint, arguments) in enumerate(fake_fal, start=1):
        assert endpoint == illustrate_fal.FAL_KONTEXT_ENDPOINT
        assert f"page_{page:02d}" in arguments["image_url"]
        assert arguments["prompt"].startswith(illustrate_fal.SOURCE_PHOTO_INSTRUCTION)


@pytest.mark.parametrize("photos", [None, ["only_one.jpg"]])
def test_source_photo_without_one_photo_per_scene_raises(fake_fal, tmp_path, photos):
    with pytest.raises(IllustrationError, match="source_photo_paths"):
        generate_illustrations_fal(
            ["scene one", "scene two"],
            "ghibli",
            SOURCE_PHOTO,
            str(tmp_path),
            source_photo_paths=photos,
        )
    assert fake_fal == []


def test_seed_depends_on_album_and_page_not_on_condition(fake_fal, tmp_path):
    photos = [
        _photo(tmp_path / "cufed_ThemePark" / f"{i}.jpg", (500, 375)) for i in range(2)
    ]
    scenes = ["a family at a castle", "a child on a ride"]

    seeds_by_condition = []
    for mode in ("none", "previous_spread", "source_photo"):
        fake_fal.clear()
        generate_illustrations_fal(
            scenes,
            "ghibli",
            {"character_reference": mode},
            str(tmp_path / mode),
            source_photo_paths=photos,
        )
        seeds_by_condition.append([arguments["seed"] for _, arguments in fake_fal])

    assert seeds_by_condition[0] == seeds_by_condition[1] == seeds_by_condition[2]
    page_1_seed, page_2_seed = seeds_by_condition[0]
    assert page_1_seed != page_2_seed


def test_seed_differs_between_albums(fake_fal, tmp_path):
    for album in ("disneyland", "legoland"):
        generate_illustrations_fal(
            ["a child on a ride"],
            "ghibli",
            {},
            str(tmp_path / album / "out"),
            source_photo_paths=[_photo(tmp_path / album / "photo.jpg", (500, 375))],
        )

    disneyland_seed, legoland_seed = (arguments["seed"] for _, arguments in fake_fal)
    assert disneyland_seed != legoland_seed


def test_blank_retry_moves_to_next_seed_and_records_the_seed_used(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("FAL_KEY", "test-key")
    urls = ["https://fal.example/black.png", "https://fal.example/color.png"]
    sent_seeds: list[int] = []

    def fake_run(endpoint, arguments):
        sent_seeds.append(arguments["seed"])
        return {"images": [{"url": urls.pop(0)}]}

    monkeypatch.setattr("fal_client.run", fake_run)
    monkeypatch.setattr(
        illustrate_fal.urllib.request,
        "urlopen",
        lambda url: _FakeResponse(BLACK_PNG if "black" in url else COLOR_PNG),
    )

    out_dir = tmp_path / "out"
    generate_illustrations_fal(
        ["a child on a ride"],
        "crayon",
        {},
        str(out_dir),
        source_photo_paths=[_photo(tmp_path / "album" / "photo.jpg", (500, 375))],
    )

    first_seed, retry_seed = sent_seeds
    assert retry_seed == first_seed + 1
    assert json.loads((out_dir / "seeds.json").read_text()) == {"page_01.png": retry_seed}


@pytest.mark.parametrize("mode", ["none", "previous_spread", "source_photo"])
def test_every_condition_samples_with_28_steps_and_guidance_3_5(fake_fal, tmp_path, mode):
    generate_illustrations_fal(
        ["a girl on a swing", "a girl eating ice cream"],
        "ghibli",
        {"character_reference": mode},
        str(tmp_path / "out"),
        source_photo_paths=[
            _photo(tmp_path / "album" / f"{i}.jpg", (500, 375)) for i in range(2)
        ],
    )

    assert {(a["num_inference_steps"], a["guidance_scale"]) for _, a in fake_fal} == {(28, 3.5)}


def test_previous_spread_forbids_carrying_over_the_reference_scene(fake_fal, tmp_path):
    """A real CUFED run with a softer instruction kept the previous spread's
    castle and fireworks on every later spread, stacking the new scene on top."""
    generate_illustrations_fal(
        ["fireworks over a castle", "a rocket ride"], "ghibli", PREVIOUS_SPREAD, str(tmp_path)
    )

    second_prompt = fake_fal[1][1]["prompt"]
    assert "only for the characters' appearance" in second_prompt
    assert "Do not keep its background, setting, objects or composition" in second_prompt
