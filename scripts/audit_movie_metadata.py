"""Audit suspicious MovieLens movie metadata records."""

import logging
from pathlib import Path

import pandas as pd

from cinematch.data.io import (
    load_ml100k_movies,
    load_ml100k_ratings,
)
from cinematch.data.schema import MOVIE_DATE_FORMAT
from cinematch.data.validation import (
    validate_movies,
    validate_ratings,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-100k"
    / "u.data"
)

MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-100k"
    / "u.item"
)


def main() -> int:
    """Inspect missing and temporally suspicious metadata."""
    ratings = load_ml100k_ratings(RATINGS_PATH)
    movies = load_ml100k_movies(MOVIES_PATH)

    validate_ratings(ratings)
    validate_movies(movies)

    release_dates = pd.to_datetime(
        movies["release_date"],
        format=MOVIE_DATE_FORMAT,
        errors="coerce",
    )

    latest_rating_date = pd.to_datetime(
        ratings["timestamp"].max(),
        unit="s",
        utc=True,
    ).tz_localize(None)

    missing_release_mask = (
        movies["release_date"].isna()
        | movies["release_date"].str.strip().eq("")
    )

    missing_imdb_mask = (
        movies["imdb_url"].isna()
        | movies["imdb_url"].str.strip().eq("")
    )

    unknown_genre_mask = (
        movies["unknown"] == 1
    )

    future_release_mask = (
        release_dates > latest_rating_date
    )

    audited_movies = movies.assign(
        release_date_parsed=release_dates,
    )

    rating_dates = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True,
    ).dt.tz_localize(None)

    temporal_audit = (
        ratings.assign(rating_date=rating_dates)
        .merge(
            audited_movies[
                [
                    "movie_id",
                    "title",
                    "release_date_parsed",
                ]
            ],
            on="movie_id",
            how="left",
            validate="many_to_one",
        )
    )

    pre_release_ratings = temporal_audit.loc[
        temporal_audit["rating_date"]
        < temporal_audit["release_date_parsed"]
    ]

    pre_release_summary = (
        pre_release_ratings.groupby(
            [
                "movie_id",
                "title",
                "release_date_parsed",
            ],
            as_index=False,
        )
        .agg(
            rating_count=("rating", "size"),
            earliest_rating=("rating_date", "min"),
            latest_rating=("rating_date", "max"),
        )
        .sort_values(
            ["rating_count", "movie_id"],
            ascending=[False, True],
        )
    )

    columns_to_show = [
        "movie_id",
        "title",
        "release_date",
        "imdb_url",
    ]

    LOGGER.info(
        "Latest rating date: %s",
        latest_rating_date.date(),
    )

    LOGGER.info(
        "Movies missing release date:\n%s",
        movies.loc[
            missing_release_mask,
            columns_to_show,
        ].to_string(index=False),
    )

    LOGGER.info(
        "Movies missing IMDb URL:\n%s",
        movies.loc[
            missing_imdb_mask,
            columns_to_show,
        ].to_string(index=False),
    )

    LOGGER.info(
        "Movies with unknown genre:\n%s",
        movies.loc[
            unknown_genre_mask,
            columns_to_show,
        ].to_string(index=False),
    )

    LOGGER.info(
        "Movies released after latest rating:\n%s",
        audited_movies.loc[
            future_release_mask,
        ]
        .sort_values("release_date_parsed")
        .loc[:, columns_to_show]
        .to_string(index=False),
    )

    LOGGER.info(
        "Future release movie count: %d",
        int(future_release_mask.sum()),
    )

    LOGGER.info(
        "Ratings recorded before catalog release date: %d",
        len(pre_release_ratings),
    )

    LOGGER.info(
        "Movies affected by temporal inconsistency: %d",
        pre_release_summary["movie_id"].nunique(),
    )

    LOGGER.info(
        "Temporal inconsistencies by movie:\n%s",
        pre_release_summary.to_string(index=False),
    )

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())
