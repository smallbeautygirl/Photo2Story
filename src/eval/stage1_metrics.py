"""Stage 1 output checks and the adapter from a saved Stage 0 selection to Stage 1 input.

These measure whether a generated story came back in the requested shape: the
right language, page lengths within the reading-level limit, no empty pages
from the line-split fallback. Narrative quality itself is judged separately by
`narrative_continuity_judge`.
"""

from __future__ import annotations

import unicodedata


def cjk_ratio(text: str) -> float:
    """Share of letters that are CJK ideographs; 0.0 for text with no letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    han = sum(1 for c in letters if unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH"))
    return han / len(letters)


def page_length_stats(pages: list[str], *, max_chars: int) -> dict[str, float | int]:
    lengths = [len(page.strip()) for page in pages]
    return {
        "max_chars": max(lengths, default=0),
        "mean_chars": sum(lengths) / len(lengths) if lengths else 0.0,
        "over_limit": sum(length > max_chars for length in lengths),
        "empty": sum(length == 0 for length in lengths),
    }


def stage0_result_from_selection(selection: dict) -> dict:
    """Rebuild the dict `run_stage1` expects from a diagnostic selection.json: the
    descriptions of the chosen photos, in page (time) order."""
    ordered = selection["ordered_paths"]
    descriptions = selection.get("descriptions", {})
    return {
        "ordered_paths": ordered,
        "selected_paths": selection.get("selected_paths", ordered),
        "descriptions": {path: descriptions.get(path, "") for path in ordered},
        "effective_context": selection.get("effective_context", ""),
    }
