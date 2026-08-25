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

GENRE_COLUMNS: Final[tuple[str, ...]] = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

MOVIE_COLUMNS: Final[tuple[str, ...]] = (
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *GENRE_COLUMNS,
)

MOVIE_DTYPES: Final[dict[str, str]] = {
    "movie_id": "int64",
    "title": "string",
    "release_date": "string",
    "video_release_date": "string",
    "imdb_url": "string",
    **{
        genre: "int8"
        for genre in GENRE_COLUMNS
    },
}

MOVIE_DATE_FORMAT: Final[str] = "%d-%b-%Y"

PROCESSED_MOVIE_COLUMNS: Final[tuple[str, ...]] = (
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "release_date_missing",
    "imdb_url",
    *GENRE_COLUMNS,
)

PROCESSED_MOVIE_DTYPES: Final[dict[str, str]] = {
    "movie_id": "int64",
    "title": "string",
    "release_year": "Int16",
    "release_date_missing": "int8",
    "imdb_url": "string",
    **{
        genre: "int8"
        for genre in GENRE_COLUMNS
    },
}


MAPPED_RATING_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "movie_id",
    "user_index",
    "movie_index",
    "rating",
    "timestamp",
)

MAPPED_RATING_DTYPES: Final[dict[str, str]] = {
    "user_id": "int64",
    "movie_id": "int64",
    "user_index": "int64",
    "movie_index": "int64",
    "rating": "float32",
    "timestamp": "int64",
}
