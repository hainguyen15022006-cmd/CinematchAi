"""Transform validated raw data into model-ready datasets."""

import pandas as pd

from cinematch.data.schema import (
    MOVIE_DATE_FORMAT,
    PROCESSED_MOVIE_COLUMNS,
)
from cinematch.data.validation import validate_movies


def prepare_movie_metadata(
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Create the processed movie catalog without dropping records.

    Release dates in MovieLens 100K are auxiliary metadata and are not
    reliable enough to determine whether a rating is valid. This
    transformation therefore parses the date and records missingness,
    but never filters a movie or a rating by release date.

    Args:
        movies:
            Raw ``u.item`` data that follows the MovieLens schema.

    Returns:
        A copy with a parsed ``release_date``, nullable
        ``release_year``, an explicit missing-value indicator and the
        original genre indicators.

    Raises:
        DataValidationError:
            If the raw movie catalog violates its expected schema.
    """
    validate_movies(movies)

    processed = movies.copy()
    processed["release_date"] = pd.to_datetime(
        processed["release_date"],
        format=MOVIE_DATE_FORMAT,
        errors="coerce",
    )
    processed["release_date_missing"] = (
        processed["release_date"].isna().astype("int8")
    )
    processed["release_year"] = (
        processed["release_date"].dt.year.astype("Int16")
    )

    # A blank IMDb URL is optional display metadata, not a training
    # error. Normalize blank strings so downstream code sees one clear
    # missing-value representation.
    processed["imdb_url"] = processed["imdb_url"].replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

    return processed.loc[:, list(PROCESSED_MOVIE_COLUMNS)].copy()
