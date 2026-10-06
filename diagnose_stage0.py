#!/usr/bin/env python3
"""Run Stage 0 alone, per photo set and per selection mode, and show what it chose.

For each (photo set, mode) this writes a contact sheet (all photos, selected ones
framed, captions with CLIP cluster and relevance score) and a selection.json with
the scores, cluster labels, replacements and selection metrics; then prints a
summary table across all runs. clip_only and random run with no VLM/LLM call
unless --with-vlm is given, so they cost nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from src.config import load_config
from src.core.logging import setup_logging
from src.eval.stage0_contact_sheet import render_contact_sheet
from src.eval.stage0_metrics import (
    cluster_coverage,
    mean_pairwise_cosine_distance,
    mean_relevance,
)
from src.stages.stage0_select import (
    encode_clip_features,
    kmeans_select,
    run_stage0,
    sort_by_exif,
)

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
ALL_MODES = ("random", "clip_only", "llm_only", "hybrid")
VLM_FREE_MODES = ("random", "clip_only")


def list_photos(set_dir: Path) -> list[str]:
    return sorted(
        str(p) for p in set_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def select_without_vlm(
    mode: str, paths: list[str], features: np.ndarray, k: int, seed: int
) -> dict:
    """clip_only / random selection with the same result shape as run_stage0, minus descriptions."""
    if mode == "clip_only":
        selected, labels = kmeans_select(features, paths, k)
    else:
        selected = random.Random(seed).sample(paths, min(k, len(paths)))
        labels = {}
    return {
        "selected_paths": selected,
        "ordered_paths": sort_by_exif(selected),
        "effective_context": "",
        "diagnostics": {"cluster_labels": labels, "scores": {}, "replacements": []},
    }


def run_mode(
    mode: str,
    paths: list[str],
    features: np.ndarray,
    stage0_config: dict,
    context: str,
    *,
    seed: int,
    with_vlm: bool,
) -> dict:
    config = dict(stage0_config, mode=mode)
    if mode in VLM_FREE_MODES and not with_vlm:
        return select_without_vlm(mode, paths, features, config["k"], seed)
    random.seed(seed)
    return run_stage0(paths, context, config)


def compute_metrics(
    result: dict, paths: list[str], features: np.ndarray, reference_labels: dict[str, int]
) -> dict:
    selected = result["selected_paths"]
    selected_rows = [paths.index(p) for p in selected]
    return {
        "mean_pairwise_clip_distance": round(mean_pairwise_cosine_distance(features[selected_rows]), 4),
        "cluster_coverage": cluster_coverage(reference_labels, selected),
        "mean_relevance": mean_relevance(result["diagnostics"]["scores"], selected),
        "replacement_count": len(result["diagnostics"]["replacements"]),
    }


def write_outputs(
    run_dir: Path,
    paths: list[str],
    result: dict,
    reference_labels: dict[str, int],
    metrics: dict,
    context: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(
        image_paths=paths,
        selected_paths=result["selected_paths"],
        cluster_labels=reference_labels,
        scores=result["diagnostics"]["scores"],
        out_path=run_dir / "contact_sheet.jpg",
    )
    record = {
        "context": context,
        "effective_context": result["effective_context"],
        "selected_paths": result["selected_paths"],
        "ordered_paths": result["ordered_paths"],
        "descriptions": result.get("descriptions", {}),
        **result["diagnostics"],
        "cluster_labels": reference_labels,
        "metrics": metrics,
    }
    (run_dir / "selection.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))


def diagnose_set(
    set_dir: Path,
    modes: list[str],
    stage0_config: dict,
    context: str,
    out_root: Path,
    *,
    seed: int,
    with_vlm: bool,
) -> list[dict]:
    paths = list_photos(set_dir)
    k = stage0_config["k"]
    if len(paths) < k:
        logger.warning("Skipping set with too few photos", extra={"set": set_dir.name, "count": len(paths)})
        return []
    features = encode_clip_features(paths, stage0_config["clip_model"], stage0_config["clip_pretrained"])
    # One reference clustering per set so cluster coverage is comparable across modes.
    _, reference_labels = kmeans_select(features, paths, k)

    rows = []
    for mode in modes:
        logger.info("Running Stage 0", extra={"set": set_dir.name, "mode": mode, "photo_count": len(paths)})
        result = run_mode(mode, paths, features, stage0_config, context, seed=seed, with_vlm=with_vlm)
        metrics = compute_metrics(result, paths, features, reference_labels)
        write_outputs(out_root / set_dir.name / mode, paths, result, reference_labels, metrics, context)
        rows.append({"set": set_dir.name, "photo_count": len(paths), "mode": mode, **metrics})
    return rows


def merge_summary_rows(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """Fresh rows replace existing rows for the same (set, mode); everything else is kept.

    Lets a partial re-run (one set, one mode) update the summary without erasing
    the rows it did not recompute.
    """
    by_key = {(row["set"], row["mode"]): row for row in existing}
    by_key.update({(row["set"], row["mode"]): row for row in fresh})
    return [by_key[key] for key in sorted(by_key)]


def write_summary(summary_path: Path, fresh_rows: list[dict]) -> list[dict]:
    existing = json.loads(summary_path.read_text()) if summary_path.is_file() else []
    merged = merge_summary_rows(existing, fresh_rows)
    summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    return merged


def print_summary(rows: list[dict]) -> None:
    header = f"{'set':30s} {'n':>3s} {'mode':10s} {'clip_dist':>9s} {'clusters':>8s} {'relev':>6s} {'swaps':>5s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        relevance = "-" if row["mean_relevance"] is None else f"{row['mean_relevance']:.2f}"
        print(
            f"{row['set'][:30]:30s} {row['photo_count']:3d} {row['mode']:10s} "
            f"{row['mean_pairwise_clip_distance']:9.3f} {row['cluster_coverage']:8d} "
            f"{relevance:>6s} {row['replacement_count']:5d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 selection diagnostics")
    parser.add_argument("--data", default="data", help="Directory holding one sub-directory per photo set")
    parser.add_argument("--sets", default="", help="Comma-separated set names to run (default: all)")
    parser.add_argument("--modes", default=",".join(VLM_FREE_MODES), help=f"Comma-separated subset of {ALL_MODES}")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--k", type=int, default=None, help="Override stage0.k from the config")
    parser.add_argument("--contexts", default="", help="JSON file mapping set name -> user context")
    parser.add_argument("--output", default="output/stage0_diag")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random mode")
    parser.add_argument("--with-vlm", action="store_true", help="Describe photos even in random/clip_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "diagnose.log")

    stage0_config = dict(load_config(args.config)["stage0"])
    if args.k is not None:
        stage0_config["k"] = args.k
    modes = [m for m in args.modes.split(",") if m]
    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}; choose from {ALL_MODES}")
    contexts = json.loads(Path(args.contexts).read_text()) if args.contexts else {}
    wanted = {s for s in args.sets.split(",") if s}
    set_dirs = [d for d in sorted(Path(args.data).iterdir()) if d.is_dir() and (not wanted or d.name in wanted)]

    rows: list[dict] = []
    for set_dir in set_dirs:
        rows += diagnose_set(
            set_dir, modes, stage0_config, contexts.get(set_dir.name, ""), out_root,
            seed=args.seed, with_vlm=args.with_vlm,
        )
    merged = write_summary(out_root / "summary.json", rows)
    print_summary(merged)
    print(f"\n{len(rows)} run(s) this time; summary.json now holds {len(merged)} row(s). Outputs: {out_root}")


if __name__ == "__main__":
    main()
