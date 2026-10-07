"""Stage 0 selection metrics: diversity, cluster coverage, relevance."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.stage0_metrics import (
    cluster_coverage,
    mean_pairwise_cosine_distance,
    mean_relevance,
)


def test_orthogonal_selection_has_maximal_diversity():
    features = np.eye(3, 8, dtype=np.float32)

    assert mean_pairwise_cosine_distance(features) == pytest.approx(1.0)


def test_duplicate_photos_lower_diversity():
    # Two identical vectors and one orthogonal: pair distances 0, 1, 1 -> mean 2/3.
    features = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)

    assert mean_pairwise_cosine_distance(features) == pytest.approx(2 / 3)


def test_single_photo_has_zero_diversity():
    features = np.array([[1, 0]], dtype=np.float32)

    assert mean_pairwise_cosine_distance(features) == 0.0


def test_cluster_coverage_counts_distinct_clusters_among_selected():
    labels = {"a": 0, "b": 0, "c": 1, "d": 2}

    assert cluster_coverage(labels, ["a", "b", "c"]) == 2
    assert cluster_coverage(labels, ["a", "c", "d"]) == 3


def test_mean_relevance_averages_selected_scores_only():
    scores = {"a": 0.8, "b": 0.4, "c": 0.1}

    assert mean_relevance(scores, ["a", "b"]) == pytest.approx(0.6)


def test_mean_relevance_is_none_when_nothing_was_scored():
    assert mean_relevance({}, ["a", "b"]) is None


def test_mean_importance_averages_human_scores_of_selected_photos():
    from src.eval.stage0_metrics import mean_importance

    importance = {"a": 1.0, "b": 0.5, "c": -0.5}

    assert mean_importance(importance, ["a", "c"]) == pytest.approx(0.25)
    assert mean_importance(importance, ["zzz"]) is None


def test_precision_at_k_counts_ties_with_the_kth_best_as_hits():
    from src.eval.stage0_metrics import precision_at_k

    importance = {"a": 1.5, "b": 1.0, "c": 1.0, "d": 0.5, "e": -0.5}

    # k=2: the 2nd-best score is 1.0, so a is a hit and d is not.
    assert precision_at_k(importance, ["a", "d"]) == pytest.approx(0.5)
    # b and c both tie the 2nd-best score: both count.
    assert precision_at_k(importance, ["b", "c"]) == pytest.approx(1.0)
    assert precision_at_k(importance, []) is None
