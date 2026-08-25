"""Input/output functions for MovieLens data."""

from pathlib import Path

import pandas as pd

from cinematch.data.schema import (
    MOVIE_COLUMNS,
    MOVIE_DTYPES,
    MAPPED_RATING_COLUMNS,
    MAPPED_RATING_DTYPES,
    PROCESSED_MOVIE_COLUMNS,
    PROCESSED_MOVIE_DTYPES,
    RATING_COLUMNS,
    RATING_DTYPES,
)


def load_ml100k_ratings(path: Path) -> pd.DataFrame:
    """Load MovieLens 100K ratings from ``u.data``.

    Args:
        path:
            Path to the MovieLens ``u.data`` file.

    Returns:
        A DataFrame containing user ID, movie ID, rating
        and timestamp.

    Raises:
        FileNotFoundError:
            If the ratings file does not exist.
        ValueError:
            If pandas cannot parse the ratings file.
    """
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "MovieLens ratings file was not found: "
            f"{resolved_path}"
        )

    try:
        ratings = pd.read_csv(
            resolved_path,
            sep="\t",
            header=None,
            names=list(RATING_COLUMNS),
            dtype=RATING_DTYPES,
            on_bad_lines="error",
        )
    except (pd.errors.ParserError, TypeError, ValueError) as error:
        raise ValueError(
            "Failed to parse MovieLens ratings file: "
            f"{resolved_path}"
        ) from error

    return ratings


def load_ml100k_movies(path: Path) -> pd.DataFrame:
    """Load MovieLens 100K movie metadata from ``u.item``.

    Args:
        path:
            Path to the MovieLens ``u.item`` file.

    Returns:
        A DataFrame containing movie titles, release
        information and 19 genre indicator columns.

    Raises:
        FileNotFoundError:
            If the metadata file does not exist.
        ValueError:
            If pandas cannot parse the metadata file.
    """
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "MovieLens movie file was not found: "
            f"{resolved_path}"
        )

    try:
        movies = pd.read_csv(
            resolved_path,
            sep="|",
            header=None,
            names=list(MOVIE_COLUMNS),
            dtype=MOVIE_DTYPES,
            encoding="latin-1",
            on_bad_lines="error",
        )
    except (
        pd.errors.ParserError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Failed to parse MovieLens movie file: "
            f"{resolved_path}"
        ) from error

    return movies


def load_processed_movies(path: Path) -> pd.DataFrame:
    """Load a processed CineMatch movie catalog from CSV.

    The explicit schema prevents nullable release years from being
    inferred as floating-point values when the CSV is read again.
    """
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Processed movie file was not found: "
            f"{resolved_path}"
        )

    try:
        movies = pd.read_csv(
            resolved_path,
            dtype=PROCESSED_MOVIE_DTYPES,
            parse_dates=["release_date"],
        )
    except (pd.errors.ParserError, TypeError, ValueError) as error:
        raise ValueError(
            "Failed to parse processed movie file: "
            f"{resolved_path}"
        ) from error

    actual_columns = tuple(movies.columns)
    if actual_columns != PROCESSED_MOVIE_COLUMNS:
        raise ValueError(
            "Unexpected processed movie columns: "
            f"expected {PROCESSED_MOVIE_COLUMNS}, "
            f"received {actual_columns}"
        )

    return movies


def load_processed_ratings(path: Path) -> pd.DataFrame:
    """Load a processed train, validation or test interaction CSV."""
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Processed rating file was not found: "
            f"{resolved_path}"
        )

    try:
        ratings = pd.read_csv(
            resolved_path,
            dtype=MAPPED_RATING_DTYPES,
        )
    except (pd.errors.ParserError, TypeError, ValueError) as error:
        raise ValueError(
            "Failed to parse processed rating file: "
            f"{resolved_path}"
        ) from error

    actual_columns = tuple(ratings.columns)
    if actual_columns != MAPPED_RATING_COLUMNS:
        raise ValueError(
            "Unexpected processed rating columns: "
            f"expected {MAPPED_RATING_COLUMNS}, "
            f"received {actual_columns}"
        )

    return ratings
