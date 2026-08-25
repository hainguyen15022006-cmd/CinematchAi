"""Deterministic ID mapping for recommendation models."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cinematch.data.schema import MAPPED_RATING_COLUMNS


MAPPING_FORMAT_VERSION = 1


class UnknownIdentifierError(ValueError):
    """Raised when an ID does not exist in a mapping."""


@dataclass(frozen=True)
class IdMapping:
    """Map external IDs to zero-based model indices.

    Attributes:
        entity_name:
            Human-readable entity name, such as ``user`` or ``movie``.
        external_ids:
            Original IDs ordered by their corresponding model index.
            The position of an ID in this tuple is its internal index.
    """

    entity_name: str
    external_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the mapping after initialization."""
        if not self.external_ids:
            raise ValueError(
                f"{self.entity_name} mapping cannot be empty"
            )

        if len(set(self.external_ids)) != len(
            self.external_ids
        ):
            raise ValueError(
                f"{self.entity_name} mapping contains duplicate IDs"
            )

        if tuple(sorted(self.external_ids)) != self.external_ids:
            raise ValueError(
                f"{self.entity_name} IDs must be sorted"
            )

    @property
    def size(self) -> int:
        """Return the number of known entities."""
        return len(self.external_ids)

    @property
    def external_to_index(self) -> dict[int, int]:
        """Return the external-ID to model-index lookup."""
        return {
            external_id: index
            for index, external_id
            in enumerate(self.external_ids)
        }

    def encode(self, values: pd.Series) -> pd.Series:
        """Convert external IDs into zero-based model indices."""
        encoded = values.map(self.external_to_index)

        unknown_mask = encoded.isna()

        if unknown_mask.any():
            unknown_values = (
                values.loc[unknown_mask]
                .dropna()
                .unique()
                .tolist()
            )

            raise UnknownIdentifierError(
                f"Unknown {self.entity_name} IDs: "
                f"{unknown_values[:5]}"
            )

        return encoded.astype("int64")

    def decode(self, indices: pd.Series) -> pd.Series:
        """Convert model indices back to external IDs."""
        invalid_mask = (
            (indices < 0)
            | (indices >= self.size)
        )

        if invalid_mask.any():
            invalid_values = (
                indices.loc[invalid_mask]
                .unique()
                .tolist()
            )

            raise UnknownIdentifierError(
                f"Invalid {self.entity_name} indices: "
                f"{invalid_values[:5]}"
            )

        index_to_external = dict(
            enumerate(self.external_ids)
        )

        return indices.map(
            index_to_external
        ).astype("int64")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping representation."""
        return {
            "entity_name": self.entity_name,
            "external_ids": list(self.external_ids),
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
    ) -> "IdMapping":
        """Create and validate a mapping from decoded JSON data."""
        if not isinstance(payload, dict):
            raise ValueError(
                "ID mapping entry must be a JSON object"
            )

        entity_name = payload.get("entity_name")
        external_ids = payload.get("external_ids")

        if not isinstance(entity_name, str):
            raise ValueError(
                "ID mapping entity_name must be a string"
            )

        if not isinstance(external_ids, list):
            raise ValueError(
                "ID mapping external_ids must be a list"
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            for value in external_ids
        ):
            raise ValueError(
                "ID mapping external_ids must contain integers"
            )

        return cls(
            entity_name=entity_name,
            external_ids=tuple(external_ids),
        )


def build_id_mapping(
    values: Iterable[int],
    entity_name: str,
) -> IdMapping:
    """Build a deterministic mapping from unique positive IDs."""
    external_ids = tuple(
        sorted({
            int(value)
            for value in values
        })
    )

    if any(value <= 0 for value in external_ids):
        raise ValueError(
            f"{entity_name} IDs must be positive"
        )

    return IdMapping(
        entity_name=entity_name,
        external_ids=external_ids,
    )


def build_cinematch_id_mappings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[IdMapping, IdMapping]:
    """Build user and movie mappings for CineMatch.

    User IDs come from rating interactions. Movie IDs come from the
    complete catalog so every known movie can be represented.
    """
    user_mapping = build_id_mapping(
        ratings["user_id"],
        entity_name="user",
    )

    movie_mapping = build_id_mapping(
        movies["movie_id"],
        entity_name="movie",
    )

    return user_mapping, movie_mapping


def apply_id_mappings(
    ratings: pd.DataFrame,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
) -> pd.DataFrame:
    """Add model-ready user and movie indices to ratings."""
    mapped = ratings.copy()

    mapped["user_index"] = user_mapping.encode(
        mapped["user_id"]
    )
    mapped["movie_index"] = movie_mapping.encode(
        mapped["movie_id"]
    )

    return mapped.loc[
        :,
        list(MAPPED_RATING_COLUMNS),
    ].copy()


def save_id_mappings(
    path: Path,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
) -> None:
    """Save user and movie mappings as versioned UTF-8 JSON."""
    if user_mapping.entity_name != "user":
        raise ValueError(
            "user_mapping must have entity_name='user'"
        )

    if movie_mapping.entity_name != "movie":
        raise ValueError(
            "movie_mapping must have entity_name='movie'"
        )

    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": MAPPING_FORMAT_VERSION,
        "users": user_mapping.as_dict(),
        "movies": movie_mapping.as_dict(),
    }

    with resolved_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")


def load_id_mappings(
    path: Path,
) -> tuple[IdMapping, IdMapping]:
    """Load and validate versioned user and movie mappings."""
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"ID mapping file was not found: {resolved_path}"
        )

    try:
        with resolved_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid ID mapping JSON: {resolved_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "ID mapping document must be a JSON object"
        )

    version = payload.get("version")
    if version != MAPPING_FORMAT_VERSION:
        raise ValueError(
            "Unsupported ID mapping version: "
            f"expected {MAPPING_FORMAT_VERSION}, "
            f"received {version}"
        )

    user_mapping = IdMapping.from_dict(
        payload.get("users")
    )
    movie_mapping = IdMapping.from_dict(
        payload.get("movies")
    )

    if user_mapping.entity_name != "user":
        raise ValueError(
            "Loaded user mapping has an invalid entity name"
        )

    if movie_mapping.entity_name != "movie":
        raise ValueError(
            "Loaded movie mapping has an invalid entity name"
        )

    return user_mapping, movie_mapping
