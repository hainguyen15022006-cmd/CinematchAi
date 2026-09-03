"""End-to-end Top-K evaluation over the shared CineMatch protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from cinematch.evaluation.candidates import (
    build_candidate_set,
    build_positive_items,
    build_seen_items,
)
from cinematch.evaluation.ranking import (
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)
from cinematch.serving.predictor import Predictor


REQUIRED_COLUMNS = {"user_index", "movie_index", "rating"}


@dataclass(frozen=True)
class TopKEvaluationResult:
    """Aggregate metrics and audit counts for one model run."""

    model_name: str
    k: int
    positive_threshold: float
    negative_sample_size: int
    seed: int
    recall_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    coverage_at_k: float
    evaluated_users: int
    skipped_users: int
    skipped_no_positive: int
    score_seconds: float

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def _validate_partition(frame: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} cannot be empty")


def evaluate_topk(
    predictor: Predictor,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    catalog_movie_indices: range | list[int] | tuple[int, ...],
    *,
    k: int = 10,
    positive_threshold: float = 4.0,
    negative_sample_size: int = 100,
    seed: int = 42,
    max_users: int | None = None,
) -> TopKEvaluationResult:
    """Evaluate one predictor with shared candidates and ranking metrics.

    Train and validation define seen items.  Test is used only to identify
    held-out positives and compute final ranking metrics.
    """
    for frame, name in ((train, "train"), (validation, "validation"), (test, "test")):
        _validate_partition(frame, name)
    if k <= 0:
        raise ValueError("k must be positive")
    if negative_sample_size < 0:
        raise ValueError("negative_sample_size must be non-negative")
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users must be positive when provided")
    catalog = tuple(int(item) for item in catalog_movie_indices)
    if not catalog:
        raise ValueError("catalog_movie_indices cannot be empty")

    train_by_user = train.groupby("user_index")["movie_index"].apply(list).to_dict()
    validation_by_user = (
        validation.groupby("user_index")["movie_index"].apply(list).to_dict()
    )
    recalls: list[float] = []
    ndcgs: list[float] = []
    hit_rates: list[float] = []
    rankings: list[list[int]] = []
    skipped_no_positive = 0
    score_seconds = 0.0

    user_groups = list(test.groupby("user_index", sort=True))
    if max_users is not None:
        user_groups = user_groups[:max_users]
    for user_index, user_test in user_groups:
        seen = build_seen_items(
            train_by_user.get(user_index, ()),
            validation_by_user.get(user_index, ()),
        )
        positives = build_positive_items(
            user_test["movie_index"].tolist(),
            user_test["rating"].tolist(),
            positive_threshold=positive_threshold,
        )
        if not positives:
            skipped_no_positive += 1
            continue
        candidates = build_candidate_set(
            catalog_movie_indices=catalog,
            seen_items=seen,
            positive_items=positives,
            number_of_negatives=negative_sample_size,
            seed=seed,
        )
        started = perf_counter()
        scores = np.asarray(predictor.score(int(user_index), candidates), dtype=np.float64)
        score_seconds += perf_counter() - started
        if scores.shape != (len(candidates),):
            raise ValueError("predictor must return one score per candidate")
        if not np.isfinite(scores).all():
            raise ValueError("predictor returned a non-finite score")
        ranking = [
            int(movie)
            for movie, _score in sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: (-float(pair[1]), int(pair[0])),
            )
        ]
        rankings.append(ranking[:k])
        recalls.append(recall_at_k(ranking, positives, k))
        ndcgs.append(ndcg_at_k(ranking, positives, k))
        hit_rates.append(hit_rate_at_k(ranking, positives, k))

    if not rankings:
        raise ValueError("no users are eligible for Top-K evaluation")
    evaluated = len(rankings)
    return TopKEvaluationResult(
        model_name=predictor.model_name,
        k=k,
        positive_threshold=float(positive_threshold),
        negative_sample_size=negative_sample_size,
        seed=seed,
        recall_at_k=float(np.mean(recalls)),
        ndcg_at_k=float(np.mean(ndcgs)),
        hit_rate_at_k=float(np.mean(hit_rates)),
        coverage_at_k=coverage_at_k(rankings, catalog, k),
        evaluated_users=evaluated,
        skipped_users=skipped_no_positive,
        skipped_no_positive=skipped_no_positive,
        score_seconds=score_seconds,
    )
