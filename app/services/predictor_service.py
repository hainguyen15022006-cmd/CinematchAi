"""Load the trained recommendation model once and score CineMatch users.

The AI library speaks in model indices; the API speaks in original
MovieLens IDs. This module owns that translation, keeps one loaded
predictor per process and turns library errors into messages the API can
return to the Frontend.

New CineMatch accounts are absent from MovieLens, so every member is
scored through the cold-start path: their onboarding ratings are matched
against training users and the resulting fold-in vector scores the
candidate movies. The behaviour and limits of that path are described in
``docs/COLD_START.md``.
"""

from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from app.core.config import settings
from cinematch.recommendation.model_scores import (
    prepare_group_candidate_scores,
)
from cinematch.serving import TorchPredictor, load_torch_predictor


class RecommenderUnavailable(RuntimeError):
    """The serving artifacts are missing, unreadable or incompatible."""


class MemberProfileTooThin(ValueError):
    """A member cannot be scored because their rating profile is unusable."""


@dataclass(frozen=True)
class MovieIndexMapping:
    """Translation between MovieLens IDs and model movie indices."""

    index_by_movielens_id: dict[int, int]
    movielens_id_by_index: dict[int, int]

    @classmethod
    def from_mapping_file(cls, path: Path) -> "MovieIndexMapping":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            external_ids = payload["movies"]["external_ids"]
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RecommenderUnavailable(
                f"Could not read movie ID mappings from {path}"
            ) from error

        index_by_id = {
            int(movielens_id): index
            for index, movielens_id in enumerate(external_ids)
        }
        return cls(
            index_by_movielens_id=index_by_id,
            movielens_id_by_index={
                index: movielens_id
                for movielens_id, index in index_by_id.items()
            },
        )


def _read_training_ratings(path: Path) -> pd.DataFrame:
    """Read the frozen train-only ratings used for cold-start fold-in."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                frame = pd.read_csv(handle)
        else:
            frame = pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise RecommenderUnavailable(
            f"Could not read training ratings from {path}"
        ) from error

    missing = {"user_index", "movie_index", "rating"} - set(frame.columns)
    if missing:
        raise RecommenderUnavailable(
            f"Training ratings {path} is missing columns: {sorted(missing)}"
        )
    return frame


class GroupRecommender:
    """Score a group of CineMatch members against candidate movies."""

    def __init__(
        self,
        predictor: TorchPredictor,
        mapping: MovieIndexMapping,
    ) -> None:
        self.predictor = predictor
        self.mapping = mapping

    @property
    def model_name(self) -> str:
        return self.predictor.model_name

    def known_movielens_ids(
        self,
        movielens_ids: Sequence[int],
    ) -> list[int]:
        """Keep only the IDs the trained model can score, in input order."""
        return [
            int(movielens_id)
            for movielens_id in movielens_ids
            if int(movielens_id) in self.mapping.index_by_movielens_id
        ]

    def score_members(
        self,
        member_ratings: Sequence[Mapping[int, float]],
        candidate_movielens_ids: Sequence[int],
    ) -> dict[int, tuple[float, ...]]:
        """Return candidate scores per MovieLens ID, ordered by member.

        Args:
            member_ratings:
                One mapping per member from MovieLens movie ID to the
                rating that member gave during onboarding.
            candidate_movielens_ids:
                Movies to score, given as original MovieLens IDs.

        Returns:
            Mapping from MovieLens movie ID to one score per member, in
            the same member order as ``member_ratings``.

        Raises:
            MemberProfileTooThin:
                If a member has no rating the model can use.
            RecommenderUnavailable:
                If the model rejects the prepared request.
        """
        candidates = self.known_movielens_ids(candidate_movielens_ids)
        if not candidates:
            raise RecommenderUnavailable(
                "No candidate movie is present in the trained model catalog"
            )

        candidate_indices = [
            self.mapping.index_by_movielens_id[movielens_id]
            for movielens_id in candidates
        ]

        score_rows: list[list[float]] = []
        for position, ratings in enumerate(member_ratings, start=1):
            indexed_ratings = {
                self.mapping.index_by_movielens_id[int(movielens_id)]: float(
                    rating
                )
                for movielens_id, rating in ratings.items()
                if int(movielens_id) in self.mapping.index_by_movielens_id
            }
            if not indexed_ratings:
                raise MemberProfileTooThin(
                    f"Member {position} has no rating the model can use; "
                    "rate a few movies from the onboarding list first"
                )
            try:
                scores = self.predictor.predict_for_new_user(
                    indexed_ratings,
                    candidate_indices,
                )
            except ValueError as error:
                raise MemberProfileTooThin(
                    f"Member {position} could not be matched to the "
                    f"training users: {error}"
                ) from error
            score_rows.append([float(score) for score in scores])

        return prepare_group_candidate_scores(
            movie_ids=candidates,
            member_score_rows=score_rows,
        )


_LOCK = threading.Lock()
_RECOMMENDER: GroupRecommender | None = None


def _serving_path(filename: str) -> Path:
    base = Path(settings.SERVING_ARTIFACTS_DIR)
    if not base.is_absolute():
        base = settings.project_root / base
    return base / filename


def load_recommender() -> GroupRecommender:
    """Return the process-wide recommender, loading it on first use."""
    global _RECOMMENDER
    with _LOCK:
        if _RECOMMENDER is not None:
            return _RECOMMENDER

        checkpoint = _serving_path(settings.SERVING_CHECKPOINT_FILE)
        ratings_path = _serving_path(settings.SERVING_TRAIN_RATINGS_FILE)
        mappings_path = _serving_path(settings.SERVING_ID_MAPPINGS_FILE)

        for path in (checkpoint, ratings_path, mappings_path):
            if not path.is_file():
                raise RecommenderUnavailable(
                    f"Serving artifact is missing: {path}. See "
                    "docs/DEPLOYMENT.md for how to create the bundle."
                )

        mapping = MovieIndexMapping.from_mapping_file(mappings_path)
        training_ratings = _read_training_ratings(ratings_path)

        try:
            predictor = load_torch_predictor(
                checkpoint,
                training_ratings,
                device="cpu",
                neighbor_count=settings.RECOMMENDER_NEIGHBOR_COUNT,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise RecommenderUnavailable(
                f"Could not load the model checkpoint {checkpoint}: {error}"
            ) from error

        _RECOMMENDER = GroupRecommender(predictor, mapping)
        return _RECOMMENDER


def reset_recommender() -> None:
    """Drop the cached recommender. Used by tests and after a model swap."""
    global _RECOMMENDER
    with _LOCK:
        _RECOMMENDER = None


def recommender_status() -> dict[str, object]:
    """Describe whether real model scoring is available right now."""
    try:
        recommender = load_recommender()
    except RecommenderUnavailable as error:
        return {"available": False, "model": None, "detail": str(error)}
    return {
        "available": True,
        "model": recommender.model_name,
        "detail": "Cold-start fold-in over train-only ratings",
    }
