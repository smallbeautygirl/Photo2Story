# tests/test_stage0_select.py
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import piexif

torch = pytest.importorskip("torch")  # skip entire file if torch not available

from src.stages.stage0_select import clip_cluster_select  # noqa: E402


def _make_open_clip_mock(features_np: np.ndarray):
    """Return a fake open_clip module whose model.encode_image returns a real tensor."""
    mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.tensor(features_np)

    mock_oc = MagicMock()
    mock_oc.create_model_and_transforms.return_value = (mock_model, MagicMock(), mock_preprocess)
    return mock_oc


def test_clip_cluster_select_returns_k_paths(tmp_images):
    features = np.random.rand(5, 512).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(features)}):
        result = clip_cluster_select(tmp_images, k=3, model_name="ViT-B-32", pretrained="openai")

    assert len(result) == 3
    for path in result:
        assert path in tmp_images


def test_clip_cluster_select_no_duplicates(tmp_images):
    # Use identity-like features so each image is in its own cluster
    features = np.eye(5, 512).astype(np.float32)

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(features)}):
        result = clip_cluster_select(tmp_images, k=3, model_name="ViT-B-32", pretrained="openai")

    assert len(result) == len(set(result))


def test_clip_cluster_select_handles_heic_input(tmp_path):
    """iPhones export HEIC by default; PIL must be able to open it for CLIP embedding."""
    import pillow_heif
    from PIL import Image as PILImage

    heic_path = tmp_path / "photo.heic"
    pillow_heif.from_pillow(PILImage.new("RGB", (64, 64), color=(255, 0, 0))).save(heic_path)

    features = np.random.rand(1, 512).astype(np.float32)
    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(features)}):
        result = clip_cluster_select(
            [str(heic_path)], k=1, model_name="ViT-B-32", pretrained="openai"
        )

    assert result == [str(heic_path)]


from src.stages.stage0_select import sort_by_exif, run_stage0


def test_sort_by_exif_no_exif_returns_all(tmp_images):
    """Images without EXIF should be returned unchanged (all at end)."""
    result = sort_by_exif(tmp_images)
    assert len(result) == len(tmp_images)
    assert set(result) == set(tmp_images)


def test_sort_by_exif_with_timestamps(tmp_path):
    """Images with EXIF timestamps should be sorted chronologically."""
    from PIL import Image as PILImage

    paths = []
    timestamps = ["2024:07:01 10:00:00", "2024:07:01 08:00:00", "2024:07:01 12:00:00"]
    for i, ts in enumerate(timestamps):
        p = tmp_path / f"img_{i}.jpg"
        PILImage.new("RGB", (64, 64), color=(i*80, i*80, i*80)).save(p)
        exif_dict = {"0th": {}, "Exif": {piexif.ExifIFD.DateTimeOriginal: ts.encode()}, "1st": {}, "thumbnail": None, "GPS": {}}
        piexif.insert(piexif.dump(exif_dict), str(p))
        paths.append(str(p))

    result = sort_by_exif(paths)
    # Should be ordered: img_1 (08:00) < img_0 (10:00) < img_2 (12:00)
    assert result[0] == paths[1]
    assert result[1] == paths[0]
    assert result[2] == paths[2]


def test_run_stage0_random_mode(tmp_images, demo_config, mocker):
    """random mode selects k images without calling VLM."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "random"

    mock_describe = mocker.patch("src.stages.stage0_select.describe_photos_with_vlm",
                                  return_value={p: f"desc {i}" for i, p in enumerate(tmp_images)})

    result = run_stage0(tmp_images, context="test trip", config=cfg)

    assert len(result["selected_paths"]) == cfg["k"]
    assert "descriptions" in result
    assert "ordered_paths" in result
    assert len(result["ordered_paths"]) == cfg["k"]
    mock_describe.assert_called_once()  # still called to get descriptions


def test_run_stage0_keeps_undated_photos_in_input_order_not_selection_order(
    tmp_images, demo_config, mocker
):
    """Photos without EXIF (e.g. the CUFED mirror) keep the order they were
    given in, whatever order the selection strategy picked them in."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "random"
    cfg["k"] = len(tmp_images)
    mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        side_effect=lambda paths, *args, **kwargs: {p: "desc" for p in paths},
    )

    result = run_stage0(tmp_images, context="test trip", config=cfg)

    assert result["selected_paths"] != tmp_images  # random picks them shuffled
    assert result["ordered_paths"] == tmp_images
    assert list(result["descriptions"]) == tmp_images


def test_run_stage0_hybrid_empty_context_infers_theme(tmp_images, demo_config, mocker):
    """When context is empty, hybrid mode should infer a theme and still run scoring."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "hybrid"

    features = np.eye(5, 512).astype(np.float32)
    mock_oc = _make_open_clip_mock(features)

    mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        return_value={p: f"desc {i}" for i, p in enumerate(tmp_images[:4])},
    )
    mock_infer = mocker.patch(
        "src.stages.stage0_select.infer_theme_from_descriptions",
        return_value="a child's swimming lesson",
    )
    mock_score = mocker.patch(
        "src.stages.stage0_select.score_relevance_with_llm",
        return_value={p: 0.8 for p in tmp_images[:4]},
    )

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        result = run_stage0(tmp_images, context="", config=cfg)

    mock_infer.assert_called_once()
    mock_score.assert_called_once()
    assert result["effective_context"] == "a child's swimming lesson"


def test_parse_score_handles_decorations():
    """Score parser tolerates markdown / trailing commentary."""
    from src.stages.stage0_select import _parse_score

    assert _parse_score("0.85") == 0.85
    assert _parse_score("**0.85**") == 0.85
    assert _parse_score("0.85 (high relevance)") == 0.85
    assert _parse_score("Score: 0.42") == 0.42
    assert _parse_score("1.5") == 1.0  # clamped
    assert _parse_score("nope") is None


def test_run_stage0_hybrid_mode_calls_vlm_and_scores(tmp_images, demo_config, mocker):
    """hybrid mode calls VLM describe and LLM scoring."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "hybrid"

    features = np.eye(5, 512).astype(np.float32)
    mock_oc = _make_open_clip_mock(features)

    mock_describe = mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        return_value={p: f"desc {i}" for i, p in enumerate(tmp_images[:4])},
    )
    mock_score = mocker.patch(
        "src.stages.stage0_select.score_relevance_with_llm",
        return_value={p: 0.8 for p in tmp_images[:4]},
    )

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        result = run_stage0(tmp_images, context="beach holiday", config=cfg)

    assert len(result["ordered_paths"]) == cfg["k"]
    mock_describe.assert_called_once()
    mock_score.assert_called_once()


def test_run_stage0_hybrid_reports_scores_clusters_and_replacements(
    tmp_images, demo_config, mocker
):
    """hybrid mode exposes why it chose what it chose: every photo's cluster, every
    scored photo's relevance, and each low-score candidate swapped for a pool photo."""
    cfg = dict(demo_config["stage0"])
    cfg["mode"] = "hybrid"
    mock_oc = _make_open_clip_mock(np.eye(5, 512).astype(np.float32))

    mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        side_effect=lambda paths, model, **kw: {p: f"desc {p}" for p in paths},
    )
    scoring_calls: list[list[str]] = []

    def fake_scores(descriptions, context, model):
        paths = list(descriptions)
        scoring_calls.append(paths)
        if len(scoring_calls) == 1:  # candidates: first one is off-topic
            return {p: (0.1 if i == 0 else 0.9) for i, p in enumerate(paths)}
        return {p: 0.95 for p in paths}  # replacement pool

    mocker.patch("src.stages.stage0_select.score_relevance_with_llm", side_effect=fake_scores)

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        result = run_stage0(tmp_images, context="beach holiday", config=cfg)

    initial_candidates, pool = scoring_calls
    removed, added = initial_candidates[0], pool[0]
    diagnostics = result["diagnostics"]

    assert set(diagnostics["cluster_labels"]) == set(tmp_images)
    assert diagnostics["scores"][removed] == 0.1
    assert diagnostics["scores"][added] == 0.95
    assert diagnostics["replacements"] == [
        {"removed": removed, "removed_score": 0.1, "added": added, "added_score": 0.95}
    ]
    assert added in result["selected_paths"]
    assert removed not in result["selected_paths"]


def test_sort_by_exif_orders_heic_by_its_timestamp(tmp_path):
    """iPhone HEIC photos carry EXIF too; an earlier HEIC must sort before a later JPG
    instead of being dumped at the end as 'no EXIF'."""
    from PIL import Image as PILImage

    def save_with_timestamp(path, timestamp):
        exif = PILImage.Exif()
        exif[306] = timestamp  # DateTime
        exif.get_ifd(0x8769)[36867] = timestamp  # DateTimeOriginal
        PILImage.new("RGB", (64, 64), color=(10, 20, 30)).save(path, exif=exif.tobytes())
        return str(path)

    later_jpg = save_with_timestamp(tmp_path / "later.jpg", "2026:06:07 13:58:22")
    earlier_heic = save_with_timestamp(tmp_path / "earlier.heic", "2026:06:07 11:31:28")

    assert sort_by_exif([later_jpg, earlier_heic]) == [earlier_heic, later_jpg]


def test_select_top_k_with_llm_returns_the_photos_the_llm_named(tmp_images, mocker):
    """llm_only is listwise: the LLM sees every description and names k photos by number."""
    from src.stages.stage0_select import select_top_k_with_llm

    descriptions = {p: f"desc {i}" for i, p in enumerate(tmp_images)}
    mocker.patch("src.stages.stage0_llm.generate_text", return_value="[3, 1, 5]")

    chosen = select_top_k_with_llm(descriptions, "beach holiday", k=3, model="m", seed=0)

    assert chosen == [tmp_images[2], tmp_images[0], tmp_images[4]]


def test_select_top_k_with_llm_fills_invalid_or_missing_picks_deterministically(tmp_images, mocker):
    """Out-of-range or duplicate numbers are dropped and the gap is filled by a seeded
    random draw, so a flaky reply never yields fewer than k photos and reruns agree."""
    from src.stages.stage0_select import select_top_k_with_llm

    descriptions = {p: f"desc {i}" for i, p in enumerate(tmp_images)}
    mocker.patch("src.stages.stage0_llm.generate_text", return_value="[2, 2, 99]")

    first = select_top_k_with_llm(descriptions, "ctx", k=3, model="m", seed=7)
    second = select_top_k_with_llm(descriptions, "ctx", k=3, model="m", seed=7)

    assert first == second
    assert len(first) == 3 and len(set(first)) == 3
    assert first[0] == tmp_images[1]
    assert all(p in tmp_images for p in first)


def test_run_stage0_llm_only_does_not_fall_back_to_filename_order(tmp_images, demo_config, mocker):
    """With every photo scoring 1.0, the old top-k sort picked the first filenames;
    the listwise pick must be what the LLM chose, and all descriptions are reported."""
    cfg = dict(demo_config["stage0"], mode="llm_only", k=4)
    mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        side_effect=lambda paths, model, **kw: {p: f"desc {p}" for p in paths},
    )
    mocker.patch(
        "src.stages.stage0_select.score_relevance_with_llm",
        side_effect=lambda descs, ctx, model: {p: 1.0 for p in descs},
    )
    mocker.patch("src.stages.stage0_llm.generate_text", return_value="[5, 2, 4, 3]")

    result = run_stage0(tmp_images, context="beach holiday", config=cfg)

    assert set(result["selected_paths"]) == {tmp_images[4], tmp_images[1], tmp_images[3], tmp_images[2]}
    assert set(result["diagnostics"]["descriptions"]) == set(tmp_images)


def test_describe_photos_reuses_cached_descriptions(tmp_images, tmp_path, mocker):
    """VLM descriptions are deterministic and context-free, so a second run over the same
    photos must come from the on-disk cache instead of calling the model again."""
    from src.stages.stage0_select import describe_photos_with_vlm

    describe = mocker.patch(
        "src.stages.stage0_llm.describe_image", side_effect=lambda p, prompt, m: f"desc of {p}"
    )
    cache_dir = tmp_path / "cache"

    first = describe_photos_with_vlm(tmp_images[:3], "m", cache_dir=cache_dir)
    second = describe_photos_with_vlm(tmp_images[:4], "m", cache_dir=cache_dir)

    assert first == {p: f"desc of {p}" for p in tmp_images[:3]}
    assert second == {p: f"desc of {p}" for p in tmp_images[:4]}
    assert describe.call_count == 4  # 3 on the first run, only the new photo on the second


def test_run_stage0_hybrid_v2_lets_the_llm_pick_one_photo_per_cluster(tmp_images, demo_config, mocker):
    """hybrid_v2: CLIP clusters define the scenes to cover; within each cluster the LLM
    picks the photo, listwise, from that cluster's descriptions only. No centroid, no
    per-photo scores, no threshold."""
    cfg = dict(demo_config["stage0"], mode="hybrid_v2", k=3)
    # Photos 0 and 1 are near-identical, so k-means (k=3) must put them in one cluster.
    features = np.array([[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0], [0, 0, 1], [0, 0.99, 0.01]], dtype=np.float32)
    mock_oc = _make_open_clip_mock(features / np.linalg.norm(features, axis=1, keepdims=True))

    mocker.patch(
        "src.stages.stage0_select.describe_photos_with_vlm",
        side_effect=lambda paths, model, **kw: {p: f"desc {p}" for p in paths},
    )
    pick_calls: list[list[str]] = []

    def fake_pick(descriptions, context, *, k, model, seed):
        pick_calls.append(list(descriptions))
        return [sorted(descriptions)[-1]]  # deterministic: last path in the cluster

    mocker.patch("src.stages.stage0_select.select_top_k_with_llm", side_effect=fake_pick)
    mocker.patch("src.stages.stage0_select.score_relevance_with_llm", side_effect=AssertionError("no per-photo scoring in v2"))

    with patch.dict(sys.modules, {"open_clip": mock_oc}):
        result = run_stage0(tmp_images, context="beach holiday", config=cfg)

    labels = result["diagnostics"]["cluster_labels"]
    assert set(labels) == set(tmp_images)
    assert len(result["selected_paths"]) == 3
    assert len({labels[p] for p in result["selected_paths"]}) == 3  # one per cluster
    for call in pick_calls:  # each listwise call saw exactly one cluster
        assert len({labels[p] for p in call}) == 1 and len(call) > 1
    assert set(result["diagnostics"]["descriptions"]) == set(tmp_images)
    assert result["diagnostics"]["scores"] == {} and result["diagnostics"]["replacements"] == []
