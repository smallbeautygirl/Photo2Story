"""Run the character_reference comparison (RQ4) on a fixed set of albums.

Per album, Stage 0 (selection) and Stage 1 (story) run once and are saved to
`story.json`; Stage 2 then runs once per condition on that same saved input, so
the conditions differ only in their reference image (see
docs/adr/0004-flux-kontext-character-reference.md). A condition whose
illustrations already exist is skipped, so a rerun never pays fal twice.

    python -m scripts.run_character_reference_experiment --dry-run
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
from pathlib import Path

from src.config import load_config
from src.core.logging import setup_logging
from src.eval.character_reference_sheet import render_character_reference_sheet
from src.stages.illustrate_fal import describe_recurring_characters
from src.stages.stage0_select import run_stage0
from src.stages.stage1_story import run_stage1
from src.stages.stage2_illustrate import run_stage2

logger = logging.getLogger(__name__)

ALL_CONDITIONS = ("none", "previous_spread", "character_sheet", "source_photo")
# Source photos go to the hosted image service, so the source_photo condition
# runs only on publicly licensed albums, never on private family albums.
PRIVATE_CONDITIONS = ("none", "previous_spread", "character_sheet")
PHOTO_GLOBS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.heic", "*.HEIC")

# CUFED: the two albums screened for a recurring family that are CC BY / CC BY-SA
# (the CC BY-ND beach album is excluded: its illustrations would be derivatives).
ALBUMS: dict[str, tuple[str, ...]] = {
    "cufed_ThemePark_11_89804841@N00": ALL_CONDITIONS,
    "cufed_ThemePark_14_23807199@N00": ALL_CONDITIONS,
    "清境農場(舊照片)": PRIVATE_CONDITIONS,
    "Lemon-Japan-aquarium": PRIVATE_CONDITIONS,
}


def main() -> None:
    args = _parse_args()
    out_root = Path(args.output)
    setup_logging(out_root / "run.log")
    config = load_config(args.config)
    albums = {name: ALBUMS[name] for name in args.albums} if args.albums else ALBUMS

    pending = sum(
        config["stage0"]["k"]
        for album, conditions in albums.items()
        for condition in conditions
        if not _is_complete(out_root / album / condition, config["stage0"]["k"])
    )
    print(f"Images still to generate: at most {pending}")
    if args.dry_run:
        return

    for album, conditions in albums.items():
        _run_album(album, conditions, config, args, out_root / album)


def _run_album(
    album: str, conditions: tuple[str, ...], config: dict, args, album_dir: Path
) -> None:
    story = _load_or_create_story(album, config, args, album_dir)
    rows: dict[str, list[str]] = {}
    for condition in conditions:
        illustrations_dir = album_dir / condition / "illustrations"
        if _is_complete(album_dir / condition, len(story["descriptions"])):
            logger.info(
                "Condition already generated; skipping",
                extra={"album": album, "condition": condition},
            )
            rows[condition] = _page_paths(illustrations_dir)
            continue
        stage2_config = copy.deepcopy(config["stage2"])
        stage2_config["character_reference"] = condition
        result = run_stage2(
            story["descriptions"],
            args.style,
            stage2_config,
            str(illustrations_dir),
            source_photo_paths=story["ordered_paths"],
            character_description=story.get("character_description"),
        )
        rows[condition] = result["illustration_paths"]
    sheet = render_character_reference_sheet(rows=rows, out_path=album_dir / "comparison.png")
    print(f"{album}: {sheet}")


def _load_or_create_story(album: str, config: dict, args, album_dir: Path) -> dict:
    story_path = album_dir / "story.json"
    if story_path.exists():
        return _with_character_description(json.loads(story_path.read_text()), config, story_path)

    image_paths = sorted(p for pattern in PHOTO_GLOBS for p in glob.glob(f"data/{album}/{pattern}"))
    if not image_paths:
        raise ValueError(f"No photos found in data/{album}")
    stage0 = run_stage0(image_paths, args.context, config["stage0"])
    context = stage0.get("effective_context") or args.context
    stage1 = run_stage1(stage0, context, args.style, args.language, config["stage1"])
    story = {
        "ordered_paths": stage0["ordered_paths"],
        "descriptions": [stage0["descriptions"].get(p, "") for p in stage0["ordered_paths"]],
        "effective_context": context,
        "pages": stage1["pages"],
        "narrative": stage1["narrative"],
    }
    album_dir.mkdir(parents=True, exist_ok=True)
    return _with_character_description(story, config, story_path)


def _with_character_description(story: dict, config: dict, story_path: Path) -> dict:
    """Describe the recurring characters once per album and store it, so every
    condition gets word-for-word the same text."""
    model = config["stage2"].get("character_description_model")
    if model and "character_description" not in story:
        story["character_description"] = describe_recurring_characters(
            story["descriptions"], model
        )
    story_path.write_text(json.dumps(story, ensure_ascii=False, indent=2))
    return story


def _page_paths(illustrations_dir: Path) -> list[str]:
    return sorted(str(p) for p in illustrations_dir.glob("page_*.png"))


def _is_complete(condition_dir: Path, pages: int) -> bool:
    """A partly generated condition (e.g. a run that failed mid-book) is rerun."""
    return len(_page_paths(condition_dir / "illustrations")) == pages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/demo_gemma.yaml")
    parser.add_argument("--output", default="output/character_reference")
    parser.add_argument("--albums", nargs="*", choices=sorted(ALBUMS), help="Default: all four")
    parser.add_argument("--style", default="ghibli")
    parser.add_argument("--language", default="zh-tw")
    parser.add_argument("--context", default="", help="Empty: Stage 0 infers the theme")
    parser.add_argument("--dry-run", action="store_true", help="Only report how many images remain")
    return parser.parse_args()


if __name__ == "__main__":
    main()
