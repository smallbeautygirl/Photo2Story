#!/usr/bin/env python3
"""Run a pipeline config against a photo set and save results for evaluation."""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

from src.config import load_config
from src.pipeline import StoryPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Photo2Story pipeline for ablation studies")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--photos", required=True, help="Glob pattern for photos, e.g. 'data/*.jpg'")
    parser.add_argument("--context", default="", help="Trip context description")
    parser.add_argument("--style", default="watercolor")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    image_paths = sorted(glob.glob(args.photos))
    if not image_paths:
        raise ValueError(f"No images found for: {args.photos}")

    cfg = load_config(args.config)
    pipeline = StoryPipeline(cfg)
    result = pipeline.run(image_paths, args.context, args.style, args.output)

    meta = {
        "config": args.config,
        "context": args.context,
        "style": args.style,
        "pages": result["pages"],
        "narrative": result["narrative"],
        "selected_paths": result["selected_paths"],
    }
    meta_path = Path(args.output) / "result.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Done. PDF: {result['pdf_path']}, metadata: {meta_path}")


if __name__ == "__main__":
    main()
