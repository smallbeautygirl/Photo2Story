#!/usr/bin/env python3
"""Run a pipeline config against a photo set and save results for evaluation."""
from __future__ import annotations
import argparse
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.core.logging import setup_logging
from src.pipeline import StoryPipeline
from src.utils.styles import available_styles

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Photo2Story pipeline for ablation studies")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--photos", required=True, help="Glob pattern for photos, e.g. 'data/*.jpg'")
    parser.add_argument("--context", default="", help="Trip context description")
    parser.add_argument(
        "--style",
        default="watercolor",
        help=(
            "Illustration style preset. Registered: "
            f"{', '.join(available_styles())}. Unknown values are used as a free-form style."
        ),
    )
    parser.add_argument(
        "--language",
        default="en",
        help="BCP 47 locale code for the story output language (e.g. en, zh-tw, zh-cn, ja, fr)",
    )
    parser.add_argument("--output", required=True, help="Parent output directory (a per-run subdir is created inside)")
    args = parser.parse_args()

    run_dir = _make_run_dir(args.output, args.config)
    log_file = setup_logging(run_dir / "run.log")
    print(f"Run dir: {run_dir}")
    print(f"Log file: {log_file}")

    image_paths = sorted(glob.glob(args.photos))
    if not image_paths:
        raise ValueError(f"No images found for: {args.photos}")

    logger.info(
        "Starting pipeline run",
        extra={
            "config": args.config,
            "photos_glob": args.photos,
            "photo_count": len(image_paths),
            "context": args.context,
            "style": args.style,
            "language": args.language,
            "run_dir": str(run_dir),
        },
    )

    cfg = load_config(args.config)
    pipeline = StoryPipeline(cfg)
    result = pipeline.run(
        image_paths, args.context, args.style, str(run_dir), language=args.language
    )

    page_records = _build_page_records(result, run_dir)

    meta = {
        "config": args.config,
        "context": args.context,
        "effective_context": result.get("effective_context", ""),
        "style": args.style,
        "language": args.language,
        "narrative": result["narrative"],
        "selected_paths": result["selected_paths"],
        "pages": page_records,
    }
    meta_path = run_dir / "result.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    pages_md_path = run_dir / "pages.md"
    pages_md_path.write_text(_render_pages_markdown(args, meta, page_records, run_dir))

    print(f"Done. PDF: {result['pdf_path']}")
    print(f"Per-page report: {pages_md_path}")
    print(f"Metadata: {meta_path}")


def _make_run_dir(parent: str, config_path: str) -> Path:
    """Create <parent>/<config-stem>_<YYYYmmdd_HHMMSS>/ and return it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{Path(config_path).stem}_{stamp}"
    run_dir = Path(parent) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_page_records(result: dict, run_dir: Path) -> list[dict]:
    """Assemble per-page records: source photo, VLM description, story text, illustration."""
    ordered = result["ordered_paths"]
    descriptions = result["descriptions"]
    pages_text = result["pages"]
    illustrations = result["illustration_paths"]
    n = min(len(ordered), len(pages_text), len(illustrations))

    records: list[dict] = []
    for i in range(n):
        src = ordered[i]
        illustration_abs = Path(illustrations[i])
        try:
            illustration_rel = str(illustration_abs.relative_to(run_dir))
        except ValueError:
            illustration_rel = str(illustration_abs)
        records.append({
            "page": i + 1,
            "source_photo": src,
            "vlm_description": descriptions.get(src, ""),
            "page_text": pages_text[i],
            "illustration": illustration_rel,
        })
    return records


def _render_pages_markdown(
    args: argparse.Namespace,
    meta: dict,
    records: list[dict],
    run_dir: Path,
) -> str:
    """Render a human-scannable per-page report. Image paths are relative to run_dir."""
    lines = [
        f"# Storybook output — `{run_dir}`",
        "",
        f"- **Config:** `{args.config}`",
        f"- **Style:** {args.style}",
        f"- **Language:** {args.language}",
        f"- **User context:** {args.context or '(empty)'}",
        f"- **Effective context:** {meta.get('effective_context') or '(none)'}",
        f"- **Narrative arc:** {meta.get('narrative') or '(none)'}",
        "",
    ]
    for rec in records:
        source_rel = os.path.relpath(rec["source_photo"], start=run_dir)
        lines.extend([
            f"## Page {rec['page']}",
            "",
            f"- **Source photo:** `{rec['source_photo']}`",
            f"- **VLM description:** {rec['vlm_description'] or '(empty)'}",
            f"- **Page text:** {rec['page_text'] or '(empty)'}",
            f"- **Illustration:** `{rec['illustration']}`",
            "",
            "| Source photo | Illustration |",
            "| --- | --- |",
            f"| ![source]({source_rel}) | ![illustration]({rec['illustration']}) |",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
