"""Schema definitions for MovieLens 100K data."""

from typing import Final


RATING_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "movie_id",
    "rating",
    "timestamp",
)

RATING_DTYPES: Final[dict[str, str]] = {
    "user_id": "int64",
    "movie_id": "int64",
    "rating": "float32",
    "timestamp": "int64",
}

RATING_MIN: Final[float] = 1.0
RATING_MAX: Final[float] = 5.0

EXPECTED_RATING_COUNT: Final[int] = 100_000
EXPECTED_USER_COUNT: Final[int] = 943
EXPECTED_MOVIE_COUNT: Final[int] = 1_682