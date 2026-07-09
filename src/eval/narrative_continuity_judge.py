"""LLM-as-judge scoring of page-to-page narrative continuity.

Classifies each adjacent caption pair against the taxonomy hand-calibrated
in docs/eval/reference_continuity_calibration.md.
"""

from __future__ import annotations

import logging

from src.utils.gemini_client import generate_text

logger = logging.getLogger(__name__)

TAXONOMY = [
    "recurring_character_object",
    "consequence_of_prior_action",
    "setting_persistence",
    "emotional_arc_progression",
    "none",
]

JUDGE_PROMPT = """\
You are scoring narrative continuity between two consecutive pages of a children's picture book.

Page N: "{page_n}"
Page N+1: "{page_n_plus_1}"

Classify the relationship between these two pages using EXACTLY ONE of these categories:
- recurring_character_object: the same character or object from page N reappears in page N+1
- consequence_of_prior_action: page N+1's event follows causally from page N's event
- setting_persistence: the same location or event context carries across both pages
- emotional_arc_progression: there is a clear emotional shift between the two pages
- none: there is no detectable continuity between the two pages

Reply with ONLY the category name, nothing else."""


def classify_transition(page_n: str, page_n_plus_1: str, model_name: str) -> str:
    """Classify one adjacent-page transition against the continuity taxonomy.

    Falls back to "none" if the judge returns anything outside the taxonomy,
    rather than silently accepting an unscored value.
    """
    prompt = JUDGE_PROMPT.format(page_n=page_n, page_n_plus_1=page_n_plus_1)
    raw = generate_text(prompt, model_name, max_output_tokens=16).strip().lower()
    if raw not in TAXONOMY:
        logger.warning(
            "Judge returned an unrecognized category; defaulting to 'none'",
            extra={"raw": raw[:64]},
        )
        return "none"
    return raw


def score_book_continuity(pages: list[str], model_name: str) -> dict:
    """Classify every adjacent transition in a book.

    Returns:
        {"transitions": list[str], "continuous_count": int, "total_transitions": int}
    """
    transitions = [
        classify_transition(pages[i], pages[i + 1], model_name) for i in range(len(pages) - 1)
    ]
    continuous_count = sum(1 for t in transitions if t != "none")
    logger.info(
        "Scored book continuity",
        extra={"continuous_count": continuous_count, "total_transitions": len(transitions)},
    )
    return {
        "transitions": transitions,
        "continuous_count": continuous_count,
        "total_transitions": len(transitions),
    }
