"""Offline evaluation for train-only nearest-neighbour user fold-in."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from cinematch.evaluation.candidates import (
    build_candidate_set,
    build_positive_items,
)
from cinematch.evaluation.ranking import coverage_at_k, hit_rate_at_k, ndcg_at_k, recall_at_k
from cinematch.serving.predictor import TorchPredictor


@dataclass(frozen=True)
class ColdStartEvaluationResult:
    profile_size: int
    k: int
    seed: int
    recall_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    coverage_at_k: float
    evaluated_users: int
    skipped_no_positive: int
    skipped_insufficient_profile: int
    mean_neighbor_count: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_cold_start(
    predictor: TorchPredictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog_movie_indices: range | list[int] | tuple[int, ...],
    *,
    profile_size: int,
    k: int = 10,
    positive_threshold: float = 4.0,
    negative_sample_size: int = 100,
    seed: int = 42,
    max_users: int | None = None,
) -> ColdStartEvaluationResult:
    """Simulate an unseen user from N earliest train ratings.

    The simulated target user is excluded from neighbour matching. Test data
    supplies held-out positives only and never participates in fold-in.
    """
    if profile_size <= 0:
        raise ValueError("profile_size must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    catalog = tuple(int(item) for item in catalog_movie_indices)
    sort_columns = ["timestamp", "movie_index"] if "timestamp" in train else ["movie_index"]
    train_by_user = {
        int(user): rows.sort_values(sort_columns, kind="mergesort")
        for user, rows in train.groupby("user_index", sort=True)
    }
    recalls: list[float] = []
    ndcgs: list[float] = []
    hits: list[float] = []
    rankings: list[list[int]] = []
    neighbor_counts: list[int] = []
    skipped_no_positive = 0
    skipped_insufficient = 0
    user_groups = list(test.groupby("user_index", sort=True))
    if max_users is not None:
        user_groups = user_groups[:max_users]

    for user_index, user_test in user_groups:
        rows = train_by_user.get(int(user_index))
        if rows is None or len(rows) < profile_size:
            skipped_insufficient += 1
            continue
        positives = build_positive_items(
            user_test["movie_index"].tolist(),
            user_test["rating"].tolist(),
            positive_threshold=positive_threshold,
        )
        if not positives:
            skipped_no_positive += 1
            continue
        profile_rows = rows.head(profile_size)
        profile = {
            int(row.movie_index): float(row.rating)
            for row in profile_rows.itertuples(index=False)
        }
        candidates = build_candidate_set(
            catalog_movie_indices=catalog,
            seen_items=profile,
            positive_items=positives,
            number_of_negatives=negative_sample_size,
            seed=seed,
        )
        match = predictor.match_new_user(
            profile, exclude_user_indices=frozenset({int(user_index)})
        )
        scores = predictor.predict_for_new_user(
            profile,
            candidates,
            exclude_user_indices=frozenset({int(user_index)}),
        )
        ranking = [
            int(movie)
            for movie, _score in sorted(
                zip(candidates, scores, strict=True),
                key=lambda pair: (-float(pair[1]), int(pair[0])),
            )
        ]
        rankings.append(ranking[:k])
        neighbor_counts.append(len(match.neighbors))
        recalls.append(recall_at_k(ranking, positives, k))
        ndcgs.append(ndcg_at_k(ranking, positives, k))
        hits.append(hit_rate_at_k(ranking, positives, k))

    if not rankings:
        raise ValueError("no users are eligible for cold-start evaluation")
    return ColdStartEvaluationResult(
        profile_size=profile_size,
        k=k,
        seed=seed,
        recall_at_k=float(np.mean(recalls)),
        ndcg_at_k=float(np.mean(ndcgs)),
        hit_rate_at_k=float(np.mean(hits)),
        coverage_at_k=coverage_at_k(rankings, catalog, k),
        evaluated_users=len(rankings),
        skipped_no_positive=skipped_no_positive,
        skipped_insufficient_profile=skipped_insufficient,
        mean_neighbor_count=float(np.mean(neighbor_counts)),
    )
