from __future__ import annotations
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("tests/demo_photos/01.jpg").exists(),
    reason="Demo photos not present",
)


def test_full_pipeline_smoke(tmp_path):
    """Full end-to-end smoke test — requires GPU and demo photos."""
    from src.config import load_config
    from src.pipeline import StoryPipeline

    photo_dir = Path("tests/demo_photos")
    image_paths = sorted(str(p) for p in photo_dir.glob("*.jpg"))
    assert len(image_paths) >= 3

    config = load_config("configs/demo.yaml")
    pipeline = StoryPipeline(config)
    result = pipeline.run(
        image_paths=image_paths,
        context="家庭出遊，爸媽帶兩個小孩去海邊",
        style="watercolor",
        output_dir=str(tmp_path),
    )

    assert Path(result["pdf_path"]).exists()
    assert len(result["pages"]) == config["stage0"]["k"]
    assert all(len(p) > 10 for p in result["pages"])
