import pandas as pd

from cinematch.data.dataset import prepare_movie_metadata
from cinematch.data.io import load_processed_movies
from cinematch.data.schema import (
    GENRE_COLUMNS,
    MOVIE_COLUMNS,
    MOVIE_DTYPES,
    PROCESSED_MOVIE_COLUMNS,
)


def make_movies_with_missing_metadata() -> pd.DataFrame:
    """Create metadata containing known MovieLens edge cases."""
    data: dict[str, object] = {
        "movie_id": [267, 315],
        "title": ["unknown", "Apt Pupil (1998)"],
        "release_date": [None, "23-Oct-1998"],
        "video_release_date": [None, None],
        "imdb_url": [None, "https://example.com/315"],
    }

    for genre in GENRE_COLUMNS:
        data[genre] = [0, 0]

    data["unknown"] = [1, 0]
    data["Drama"] = [0, 1]

    return pd.DataFrame(
        data,
        columns=list(MOVIE_COLUMNS),
    ).astype(MOVIE_DTYPES)


def test_prepare_movie_metadata_parses_release_date() -> None:
    processed = prepare_movie_metadata(
        make_movies_with_missing_metadata(),
    )

    assert tuple(processed.columns) == PROCESSED_MOVIE_COLUMNS
    assert processed.loc[1, "release_date"] == pd.Timestamp(
        "1998-10-23"
    )
    assert processed.loc[1, "release_year"] == 1998
    assert processed.loc[1, "release_date_missing"] == 0


def test_prepare_movie_metadata_preserves_missing_date() -> None:
    raw_movies = make_movies_with_missing_metadata()

    processed = prepare_movie_metadata(raw_movies)

    assert len(processed) == len(raw_movies)
    assert processed["movie_id"].tolist() == [267, 315]
    assert pd.isna(processed.loc[0, "release_date"])
    assert pd.isna(processed.loc[0, "release_year"])
    assert processed.loc[0, "release_date_missing"] == 1
    assert processed.loc[0, "unknown"] == 1


def test_prepare_movie_metadata_does_not_mutate_raw_data() -> None:
    raw_movies = make_movies_with_missing_metadata()
    original = raw_movies.copy(deep=True)

    prepare_movie_metadata(raw_movies)

    pd.testing.assert_frame_equal(raw_movies, original)


def test_processed_movie_csv_round_trip(tmp_path) -> None:
    processed = prepare_movie_metadata(
        make_movies_with_missing_metadata(),
    )
    output_path = tmp_path / "movies.csv"
    processed.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    loaded = load_processed_movies(output_path)

    assert str(loaded["release_year"].dtype) == "Int16"
    assert str(loaded["release_date"].dtype) == "datetime64[ns]"
    assert loaded.loc[1, "release_year"] == 1998
    assert pd.isna(loaded.loc[0, "release_year"])
