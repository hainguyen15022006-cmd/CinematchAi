"""Download and verify the canonical MovieLens 100K archive."""

import hashlib
import logging
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


LOGGER = logging.getLogger(__name__)

DATASET_URL = (
    "https://files.grouplens.org/"
    "datasets/movielens/ml-100k.zip"
)
EXPECTED_ARCHIVE_MD5 = "0e33842e24a9c977be4e0107933c0723"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIRECTORY / "ml-100k.zip"
DATASET_DIRECTORY = RAW_DIRECTORY / "ml-100k"

EXPECTED_FILES = (
    "u.data",
    "u.item",
    "u.user",
)

EXPECTED_LINE_COUNTS = {
    "u.data": 100_000,
    "u.item": 1_682,
    "u.user": 943,
}


def dataset_exists() -> bool:
    """Return whether all required extracted files are present."""
    return all(
        (DATASET_DIRECTORY / filename).is_file()
        for filename in EXPECTED_FILES
    )


def calculate_md5(path: Path) -> str:
    """Calculate an MD5 checksum without loading a file into memory."""
    digest = hashlib.md5()

    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    """Reject a corrupted or unexpected MovieLens archive."""
    actual_md5 = calculate_md5(path)
    if actual_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            "MovieLens archive checksum mismatch: "
            f"expected {EXPECTED_ARCHIVE_MD5}, "
            f"received {actual_md5}"
        )

    LOGGER.info("Archive MD5 verified: %s", actual_md5)


def download_archive() -> None:
    """Download the MovieLens archive and verify its checksum."""
    LOGGER.info("Downloading: %s", DATASET_URL)

    request = Request(
        DATASET_URL,
        headers={
            "User-Agent": "CineMatch-Data-Pipeline/1.0"
        },
    )

    with urlopen(request, timeout=120) as response:
        with ARCHIVE_PATH.open("wb") as output_file:
            shutil.copyfileobj(
                response,
                output_file,
            )

    verify_archive(ARCHIVE_PATH)
    LOGGER.info("Saved archive: %s", ARCHIVE_PATH)


def is_safe_zip_member(filename: str) -> bool:
    """Return whether a ZIP member remains under the destination."""
    member_path = Path(filename)

    if member_path.is_absolute():
        return False

    if ".." in member_path.parts:
        return False

    return True


def extract_archive() -> None:
    """Extract the verified archive after path traversal checks."""
    LOGGER.info("Extracting: %s", ARCHIVE_PATH)

    with ZipFile(ARCHIVE_PATH) as archive:
        for member in archive.infolist():
            if not is_safe_zip_member(member.filename):
                raise RuntimeError(
                    f"Unsafe path in ZIP: {member.filename}"
                )

        archive.extractall(RAW_DIRECTORY)

    LOGGER.info("Extracted into: %s", RAW_DIRECTORY)


def count_lines(path: Path) -> int:
    """Count text records in one MovieLens source file."""
    with path.open(
        "r",
        encoding="latin-1",
    ) as input_file:
        return sum(1 for _ in input_file)


def validate_dataset() -> None:
    """Validate required files and canonical MovieLens row counts."""
    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (
            DATASET_DIRECTORY / filename
        ).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing dataset files: {missing_files}"
        )

    for filename, expected_count in EXPECTED_LINE_COUNTS.items():
        source_path = DATASET_DIRECTORY / filename
        actual_count = count_lines(source_path)

        if actual_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} rows in {filename}, "
                f"found {actual_count}"
            )

        LOGGER.info(
            "Validated %s: %d rows",
            source_path,
            actual_count,
        )


def main() -> int:
    """Download MovieLens only when absent, then validate it."""
    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dataset_exists():
        LOGGER.info(
            "Dataset already exists: %s",
            DATASET_DIRECTORY,
        )
        validate_dataset()
        return 0

    download_archive()
    extract_archive()
    validate_dataset()

    ARCHIVE_PATH.unlink(missing_ok=True)

    LOGGER.info("MovieLens 100K is ready")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
