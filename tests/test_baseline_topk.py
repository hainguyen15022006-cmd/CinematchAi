"""Tests for the model-independent Top-K evaluator."""

import numpy as np
import pandas as pd
import pytest

from cinematch.evaluation.topk import evaluate_topk


class DescendingMoviePredictor:
    model_name = "test_model"

    def score(self, user_index: int, candidates: tuple[int, ...]) -> np.ndarray:
        return np.asarray(candidates, dtype=np.float64)

    def predict_for_new_user(self, ratings, candidates):
        return self.score(-1, candidates)


def _partition(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_index", "movie_index", "rating"])


def test_evaluate_topk_reports_metrics_and_skip_counts() -> None:
    train = _partition([(0, 0, 5.0), (1, 1, 4.0)])
    validation = _partition([(0, 1, 3.0), (1, 2, 4.0)])
    test = _partition([(0, 9, 5.0), (1, 8, 2.0)])

    result = evaluate_topk(
        DescendingMoviePredictor(),
        train,
        validation,
        test,
        range(10),
        k=1,
        negative_sample_size=2,
        seed=42,
    )

    assert result.evaluated_users == 1
    assert result.skipped_users == 1
    assert result.skipped_no_positive == 1
    assert result.recall_at_k == 1.0
    assert result.ndcg_at_k == 1.0
    assert result.hit_rate_at_k == 1.0
    assert result.coverage_at_k == pytest.approx(0.1)
    assert result.score_seconds >= 0.0


def test_evaluate_topk_is_reproducible() -> None:
    train = _partition([(0, 0, 5.0)])
    validation = _partition([(0, 1, 3.0)])
    test = _partition([(0, 9, 5.0)])
    first = evaluate_topk(
        DescendingMoviePredictor(), train, validation, test, range(10),
        k=3, negative_sample_size=4, seed=42,
    )
    second = evaluate_topk(
        DescendingMoviePredictor(), train, validation, test, range(10),
        k=3, negative_sample_size=4, seed=42,
    )
    assert first.recall_at_k == second.recall_at_k
    assert first.ndcg_at_k == second.ndcg_at_k
    assert first.hit_rate_at_k == second.hit_rate_at_k
    assert first.coverage_at_k == second.coverage_at_k


def test_evaluate_topk_rejects_a_bad_predictor_shape() -> None:
    class BadPredictor(DescendingMoviePredictor):
        def score(self, user_index: int, candidates: tuple[int, ...]) -> np.ndarray:
            return np.asarray([1.0])

    frame = _partition([(0, 0, 5.0)])
    test = _partition([(0, 9, 5.0)])
    with pytest.raises(ValueError, match="one score per candidate"):
        evaluate_topk(
            BadPredictor(), frame, frame, test, range(10),
            negative_sample_size=2,
        )
