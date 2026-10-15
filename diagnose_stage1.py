#!/usr/bin/env python3
"""Run Stage 1 (story generation) alone on saved Stage 0 selections and judge the result.

For each (photo set, Stage 0 selection mode, causal-inference condition) this feeds the
selected photos' descriptions into Stage 1, saves the narrative arc and pages, scores
page-to-page continuity with the LLM judge, and records shape checks (language,
page length, empty pages). Selections come from diagnose_stage0.py's output tree
(<stage0-dir>/<set>/<context_mode>/<mode>/selection.json), so Stage 1 is compared on
exactly the photos Stage 0 chose, without re-running selection.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.core.logging import setup_logging
from src.eval.narrative_continuity_judge import score_book_continuity
from src.eval.stage1_metrics import cjk_ratio, page_length_stats, stage0_result_from_selection
from src.stages.stage1_story import run_stage1
from src.utils.prompt_templates import READING_LEVELS

logger = logging.getLogger(__name__)

CONDITIONS = {"causal": True, "no_causal": False}
# Page-length limits by reading level, in characters (the prompt's length_hint for CJK).
MAX_CHARS = {"simple": 24, "standard": 45}
CJK_LANGUAGES = ("zh", "ja")


def find_selections(stage0_dir: Path, context_mode: str, modes: list[str], wanted_sets: set[str]) -> list[tuple[str, str, Path]]:
    found = []
    for set_dir in sorted(p for p in stage0_dir.iterdir() if p.is_dir()):
        if wanted_sets and set_dir.name not in wanted_sets:
            continue
        for mode in modes:
            selection = set_dir / context_mode / mode / "selection.json"
            if selection.is_file():
                found.append((set_dir.name, mode, selection))
            else:
                logger.warning("No selection for set/mode", extra={"set": set_dir.name, "mode": mode})
    return found


def run_one(selection_path: Path, stage1_config: dict, *, use_causal: bool, style: str, language: str) -> dict:
    selection = json.loads(selection_path.read_text())
    stage0_result = stage0_result_from_selection(selection)
    config = dict(stage1_config, use_causal_inference=use_causal)
    stage1 = run_stage1(stage0_result, stage0_result["effective_context"], style, language, config)
    return {
        "context": stage0_result["effective_context"],
        "ordered_paths": stage0_result["ordered_paths"],
        "descriptions": stage0_result["descriptions"],
        "narrative": stage1["narrative"],
        "pages": stage1["pages"],
    }


def judge_and_measure(story: dict, *, judge_model: str, language: str, reading_level: str) -> dict:
    continuity = score_book_continuity(story["pages"], judge_model)
    lengths = page_length_stats(story["pages"], max_chars=MAX_CHARS.get(reading_level, MAX_CHARS["standard"]))
    text = "".join(story["pages"])
    return {
        "continuity_ratio": continuity["continuous_count"] / max(continuity["total_transitions"], 1),
        "transitions": continuity["transitions"],
        "cjk_ratio": round(cjk_ratio(text), 3),
        "language_ok": (cjk_ratio(text) > 0.9) if language.startswith(CJK_LANGUAGES) else (cjk_ratio(text) < 0.1),
        **lengths,
    }


def merge_summary_rows(existing: list[dict], fresh: list[dict]) -> list[dict]:
    def key(row: dict) -> tuple[str, str, str]:
        return (row["set"], row["selection_mode"], row["condition"])

    by_key = {key(row): row for row in existing}
    by_key.update({key(row): row for row in fresh})
    return [by_key[k] for k in sorted(by_key)]


def print_summary(rows: list[dict]) -> None:
    header = f"{'set':30s} {'sel':9s} {'cond':9s} {'contin':>6s} {'cjk':>5s} {'maxch':>5s} {'over':>4s} {'empty':>5s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['set'][:30]:30s} {row['selection_mode']:9s} {row['condition']:9s} "
            f"{row['continuity_ratio']:6.2f} {row['cjk_ratio']:5.2f} {row['max_chars']:5d} "
            f"{row['over_limit']:4d} {row['empty']:5d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 story-generation diagnostics")
    parser.add_argument("--stage0-dir", default="output/stage0_diag_gemma")
    parser.add_argument("--context-mode", default="none", help="Which Stage 0 context run to read selections from")
    parser.add_argument("--selection-modes", default="llm_only,hybrid", help="Stage 0 modes whose selections to use")
    parser.add_argument("--sets", default="", help="Comma-separated set names (default: all with a selection)")
    parser.add_argument("--conditions", default="causal,no_causal", help=f"Subset of {list(CONDITIONS)}")
    parser.add_argument("--config", default="configs/demo_gemma.yaml")
    parser.add_argument("--style", default="ghibli")
    parser.add_argument("--language", default="zh-tw")
    parser.add_argument("--reading-level", default="simple", choices=list(READING_LEVELS))
    parser.add_argument("--judge-model", default="", help="Judge model (default: the Stage 1 model)")
    parser.add_argument("--output", default="output/stage1_diag_gemma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "diagnose.log")

    stage1_config = dict(load_config(args.config)["stage1"], reading_level=args.reading_level)
    judge_model = args.judge_model or stage1_config["model"]
    modes = [m for m in args.selection_modes.split(",") if m]
    conditions = [c for c in args.conditions.split(",") if c]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}; choose from {list(CONDITIONS)}")
    wanted = {s for s in args.sets.split(",") if s}

    rows: list[dict] = []
    for set_name, mode, selection_path in find_selections(Path(args.stage0_dir), args.context_mode, modes, wanted):
        for condition in conditions:
            logger.info("Running Stage 1", extra={"set": set_name, "selection_mode": mode, "condition": condition})
            story = run_one(selection_path, stage1_config, use_causal=CONDITIONS[condition],
                            style=args.style, language=args.language)
            metrics = judge_and_measure(story, judge_model=judge_model, language=args.language,
                                        reading_level=args.reading_level)
            run_dir = out_root / set_name / mode / condition
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "story.json").write_text(json.dumps({**story, "metrics": metrics}, ensure_ascii=False, indent=2))
            rows.append({"set": set_name, "selection_mode": mode, "condition": condition, **metrics})

    summary_path = out_root / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.is_file() else []
    merged = merge_summary_rows(existing, rows)
    summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print_summary(merged)
    print(f"\n{len(rows)} run(s) this time; summary.json now holds {len(merged)} row(s). Outputs: {out_root}")


if __name__ == "__main__":
    main()
