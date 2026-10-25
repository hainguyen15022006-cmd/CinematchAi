"""Create the small serving bundle the Backend deploys with.

The Backend needs four things to score a group with the trained model:

- a trained MF/GMF/NCF checkpoint,
- the train-only ratings used for cold-start nearest-neighbour fold-in,
- the MovieLens ID to model-index mapping,
- the movie catalog, so a fresh deployment can seed its database.

The full ``data/processed`` and ``outputs`` folders are derived data and
stay out of Git. This script copies just the serving inputs into
``artifacts/serving`` and compresses the two tables, which keeps the
bundle well under one megabyte so a deployment needs no data pipeline.

Usage:

    python scripts/build_serving_bundle.py
    python scripts/build_serving_bundle.py --checkpoint outputs/baselines/gmf_checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "baselines" / "mf_checkpoint.pt"
)
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_MAPPINGS = PROJECT_ROOT / "data" / "processed" / "id_mappings.json"
DEFAULT_MOVIES = PROJECT_ROOT / "data" / "processed" / "movies.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "serving"

TRAIN_COLUMNS = ("user_index", "movie_index", "rating")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require(path: Path, hint: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} is missing. {hint}")
    return resolved


def write_train_ratings(source: Path, target: Path) -> int:
    """Copy only the three columns the cold-start fold-in needs."""
    with source.open(newline="", encoding="utf-8") as src, gzip.open(
        target, "wt", newline="", encoding="utf-8", compresslevel=9
    ) as dst:
        reader = csv.DictReader(src)
        missing = set(TRAIN_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{source} is missing columns: {sorted(missing)}"
            )
        writer = csv.writer(dst)
        writer.writerow(TRAIN_COLUMNS)
        rows = 0
        for row in reader:
            writer.writerow([row[column] for column in TRAIN_COLUMNS])
            rows += 1
    return rows


def write_movies(source: Path, target: Path) -> int:
    """Compress the movie catalog used to seed a fresh database."""
    with source.open("rb") as src, gzip.open(
        target, "wb", compresslevel=9
    ) as dst:
        shutil.copyfileobj(src, dst)
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def main() -> int:
    args = parse_args()

    checkpoint = _require(
        args.checkpoint,
        "Run python scripts/train_baseline.py first.",
    )
    train = _require(
        args.train,
        "Run python scripts/prepare_data.py first.",
    )
    mappings = _require(
        args.mappings,
        "Run python scripts/prepare_data.py first.",
    )
    movies = _require(
        args.movies,
        "Run python scripts/prepare_data.py first.",
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(checkpoint, output / "mf_checkpoint.pt")
    rating_rows = write_train_ratings(train, output / "train_ratings.csv.gz")
    shutil.copyfile(mappings, output / "id_mappings.json")
    movie_rows = write_movies(movies, output / "movies.csv.gz")

    mapping_payload = json.loads(mappings.read_text(encoding="utf-8"))
    movie_count = len(mapping_payload["movies"]["external_ids"])
    user_count = len(mapping_payload["users"]["external_ids"])

    total_bytes = sum(
        path.stat().st_size for path in output.iterdir() if path.is_file()
    )

    print(f"Bundle written to {output}")
    print(f"  checkpoint: {checkpoint.name}")
    print(f"  train ratings: {rating_rows} rows")
    print(f"  catalog: {movie_rows} movies")
    print(f"  model mapping: {user_count} users, {movie_count} movies")
    print(f"  total size: {total_bytes / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
