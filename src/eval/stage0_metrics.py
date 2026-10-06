"""Stage 0 selection metrics.

Quantify what a photo-selection strategy achieved, independent of the
downstream story: how different the selected photos are from each other
(diversity), how many CLIP clusters they span (coverage), and how relevant
the LLM judged them to the user's theme (relevance).
"""

from __future__ import annotations

import numpy as np


def mean_pairwise_cosine_distance(features: np.ndarray) -> float:
    """Mean of (1 - cosine similarity) over all photo pairs; 0.0 for fewer than two photos."""
    if features.shape[0] < 2:
        return 0.0
    normalised = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = normalised @ normalised.T
    upper_triangle = np.triu_indices(features.shape[0], k=1)
    return float(np.mean(1.0 - similarity[upper_triangle]))


def cluster_coverage(cluster_labels: dict[str, int], selected_paths: list[str]) -> int:
    """Number of distinct CLIP clusters represented among the selected photos."""
    return len({cluster_labels[path] for path in selected_paths if path in cluster_labels})


def mean_relevance(scores: dict[str, float], selected_paths: list[str]) -> float | None:
    """Mean LLM relevance score of the selected photos; None if none of them were scored."""
    selected_scores = [scores[path] for path in selected_paths if path in scores]
    if not selected_scores:
        return None
    return float(np.mean(selected_scores))
