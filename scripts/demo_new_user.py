"""Generate a personalized MF Top 10 from new-user MovieLens ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cinematch.data.configuration import load_data_config
from cinematch.data.io import load_processed_movies, load_processed_ratings
from cinematch.data.mapping import load_id_mappings
from cinematch.serving.predictor import load_torch_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rating",
        action="append",
        required=True,
        metavar="MOVIELENS_ID=SCORE",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/baselines/seed_42/mf_checkpoint.pt",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/cinematch.yaml")
    return parser.parse_args()


def _ratings(values: list[str]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--rating must use MOVIELENS_ID=SCORE")
        movie_id, score = value.split("=", 1)
        result[int(movie_id)] = float(score)
    return result


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    config = load_data_config(args.config, project_root=PROJECT_ROOT)
    train = load_processed_ratings(config.paths.train)
    movies = load_processed_movies(config.paths.movies_processed)
    _user_mapping, movie_mapping = load_id_mappings(config.paths.mappings)
    external_ratings = _ratings(args.rating)
    unknown = set(external_ratings) - set(movie_mapping.external_to_index)
    if unknown:
        raise ValueError(f"unknown MovieLens movie IDs: {sorted(unknown)}")
    indexed_ratings = {
        movie_mapping.external_to_index[movie_id]: rating
        for movie_id, rating in external_ratings.items()
    }
    candidates = [
        index for index in range(movie_mapping.size) if index not in indexed_ratings
    ]
    predictor = load_torch_predictor(args.checkpoint, train, neighbor_count=20)
    started = perf_counter()
    scores = predictor.predict_for_new_user(indexed_ratings, candidates)
    score_seconds = perf_counter() - started
    order = sorted(
        zip(candidates, np.asarray(scores), strict=True),
        key=lambda pair: (-float(pair[1]), int(pair[0])),
    )[: args.top_k]
    metadata = movies.set_index("movie_id")
    recommendations = []
    for rank, (movie_index, score) in enumerate(order, start=1):
        movie_id = movie_mapping.external_ids[movie_index]
        recommendations.append(
            {
                "rank": rank,
                "movie_id": movie_id,
                "movie_index": movie_index,
                "title": str(metadata.loc[movie_id, "title"]),
                "predicted_score": round(float(score), 6),
            }
        )
    profile = predictor.match_new_user(indexed_ratings)
    print(
        json.dumps(
            {
                "model": predictor.model_name,
                "ratings": external_ratings,
                "neighbor_count": len(profile.neighbors),
                "candidate_count": len(candidates),
                "score_seconds": score_seconds,
                "recommendations": recommendations,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
