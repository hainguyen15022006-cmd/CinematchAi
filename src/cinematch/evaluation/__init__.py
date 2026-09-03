"""Evaluation tools for the CineMatch recommendation system."""

from cinematch.evaluation.candidates import (
    DEFAULT_NEGATIVE_SAMPLE_SIZE,
    DEFAULT_RANDOM_SEED,
    build_candidate_set,
    build_positive_items,
    build_seen_items,
    sample_negative_items,
)
from cinematch.evaluation.ranking import (
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)
from cinematch.evaluation.topk import TopKEvaluationResult, evaluate_topk
from cinematch.evaluation.cold_start import (
    ColdStartEvaluationResult,
    evaluate_cold_start,
)


__all__ = [
    "DEFAULT_NEGATIVE_SAMPLE_SIZE",
    "DEFAULT_RANDOM_SEED",
    "build_candidate_set",
    "build_positive_items",
    "build_seen_items",
    "coverage_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "sample_negative_items",
    "TopKEvaluationResult",
    "evaluate_topk",
    "ColdStartEvaluationResult",
    "evaluate_cold_start",
]
