"""Input/output functions for MovieLens data."""

from pathlib import Path

import pandas as pd

from cinematch.data.schema import (
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