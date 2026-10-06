"""Summary merging for the Stage 0 diagnostic: a partial re-run must not erase
rows from photo sets or modes it did not touch."""

from __future__ import annotations

from diagnose_stage0 import merge_summary_rows


def test_partial_rerun_replaces_only_its_own_rows():
    existing = [
        {"set": "A", "mode": "clip_only", "cluster_coverage": 4},
        {"set": "A", "mode": "random", "cluster_coverage": 2},
        {"set": "B", "mode": "clip_only", "cluster_coverage": 4},
    ]
    fresh = [{"set": "A", "mode": "random", "cluster_coverage": 3}]

    merged = merge_summary_rows(existing, fresh)

    assert merged == [
        {"set": "A", "mode": "clip_only", "cluster_coverage": 4},
        {"set": "A", "mode": "random", "cluster_coverage": 3},
        {"set": "B", "mode": "clip_only", "cluster_coverage": 4},
    ]


def test_merge_with_no_existing_summary_keeps_fresh_rows_sorted():
    fresh = [
        {"set": "B", "mode": "random"},
        {"set": "A", "mode": "random"},
        {"set": "A", "mode": "clip_only"},
    ]

    assert merge_summary_rows([], fresh) == [
        {"set": "A", "mode": "clip_only"},
        {"set": "A", "mode": "random"},
        {"set": "B", "mode": "random"},
    ]
