# CineMatch AI

CineMatch is a group movie recommendation system that combines
Deep Learning models with Group Recommendation strategies.

The repository is being developed within a four-week MVP scope.
The Data part provides a reproducible pipeline on
MovieLens 100K for MF, GMF, NCF and Hybrid NCF.

## Team members

- Hải Anh: Data.
- Thành: AI Baseline, MF, GMF.
- Công Thành: NCF, Hybrid NCF, Text.
- Chúc: Backend.
- Dương: Frontend.
- Hoàng Anh: Testing, Integration, DevOps.
- Sơn: Group Recommendation and Evaluation.

## Environment requirements

- Python 3.12.
- Git.
- Network access for the first MovieLens download.

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` is the primary source of dependencies. The
`requirements.txt` file only re-references the development dependency
group to avoid declaring versions in two different places.

## Running the Data pipeline

Download MovieLens 100K:

```bash
python scripts/download_data.py
```

Explore and inspect the raw data:

```bash
python scripts/inspect_data.py \
  --output outputs/eda/ratings_profile.json
python scripts/inspect_movies.py \
  --output outputs/eda/movies_profile.json
python scripts/audit_movie_metadata.py
```

Generate model-ready data and run the post-split audit:

```bash
python scripts/prepare_data.py
python scripts/audit_splits.py
python scripts/prepare_evaluation_data.py
python scripts/prepare_numeric_features.py
python scripts/prepare_text_features.py
python scripts/report_feature_coverage.py
python scripts/create_data_manifest.py
```

Paths, split ratios, expected counts and the positive
threshold are managed centrally in
`configs/cinematch.yaml`.

## Generated outputs

```text
data/processed/
├── train.csv
├── validation.csv
├── test.csv
├── movies.csv
└── id_mappings.json
```

```text
outputs/eda/
├── ratings_profile.json
├── movies_profile.json
└── split_audit.json
```

```text
outputs/
├── data_manifest.json
├── evaluation/
    ├── catalog.json
    ├── seen_items.json
    ├── positive_test_items.json
    └── evaluation_data_summary.json
└── features/
    ├── movie_numeric_features.csv
    ├── user_genre_profiles.csv
    ├── numeric_feature_preprocessor.json
    ├── user_pseudo_text.csv
    ├── movie_text.csv
    ├── user_text_vectors.npz
    ├── movie_text_vectors.npz
    ├── text_feature_preprocessor.json
    └── feature_coverage_report.json
```

`data_manifest.json` is created last, after every other artifact exists. It
records the dataset and feature-contract versions, the feature-generation
parameters,
counts calculated from the generated files, split policy, quality checks and
SHA-256 checksums. Its contract is documented by
`schemas/data_manifest.schema.json`.

The evaluation handoff freezes the catalog, seen-item policy, positive test
items, eligible/skipped users and cold-start counts used by every model. Its
contract and ownership rules are in `docs/EVALUATION_DATA_HANDOFF.md`.

The numeric-feature handoff contains the 39 real numeric inputs for Hybrid
NCF: 19 movie genres, one train-normalized release year and 19 train-only user
genre-history values. Its formulas, fixed column order and ownership rules are
in `docs/NUMERIC_FEATURES.md`.

The text-feature handoff deterministically derives English user pseudo-text
from train ratings and movie text from catalog titles and genres. It encodes
both sides to 128 dimensions and uses a Hadamard product to preserve the
167-dimensional Hybrid contract. See `docs/PSEUDO_TEXT_FEATURES.md`.

The coverage report summarizes fallback and imputation behaviour: pseudo-text
fallback users (235/943), missing-year imputations, movies outside train,
normalized-year outliers and text-vector norms. Preference genres require a
train mean rating of at least 4.0 over at least 3 rated movies of that genre.

The files above are derived data and are not committed to
GitHub. Team members regenerate them using the scripts.

## Data contract for the models

All three partitions share the same schema:

```text
user_id, movie_id, user_index, movie_index, rating, timestamp
```

- MF, GMF and NCF use `user_index` and `movie_index`.
- `user_id` and `movie_id` are kept for Backend integration.
- `rating` is the target for explicit-feedback training.
- `timestamp` is used to demonstrate the temporal split, not as a
  default embedding feature.
- Hybrid NCF can join the 19 genre columns from `movies.csv` via
  `movie_id`.
- Hybrid NCF must load the generated numeric artifacts instead of fitting
  release-year or history statistics again inside the training script.

Only `train.csv` is used to fit model parameters.
`validation.csv` is used for hyperparameter selection or early
stopping. `test.csv` is used only for the final evaluation.

## Evaluation protocol

- Per-user temporal split of approximately 80/10/10.
- Ratings of 4 or higher are treated as positive interactions.
- Candidate construction and negative sampling must be identical
  across the models being compared.
- Cold-start items are not removed from the original data.
- Metrics are reported on the full test set, and warm-start metrics
  may additionally be reported.

Details are in `docs/DATA_REPORT.md` and
`docs/DATA_DICTIONARY.md`.

## Testing

Run the full test suite:

```bash
python -m pytest -v
```

## Running the Backend API

After setting up the environment and the Data pipeline, create the local
configuration file and seed the movie catalog:

```bash
cp .env.example .env
python scripts/seed_movies.py
uvicorn app.main:app --reload
```

Check the API at:

- Health check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

The Backend uses FastAPI, SQLAlchemy and SQLite for the MVP. Authentication
uses JWT; passwords are hashed with Argon2. `movie_id` in the API is always
the original MovieLens ID, not the internal `movies.id` key.

The Frontend can integrate early via `POST /recommend/mock`. This endpoint
returns a deterministic Top-K according to contract v1 without requiring a
model or database. Endpoint details, the room/vote flow and ID conventions
are in `docs/BACKEND_API.md`; the Group Recommendation schema is in
`docs/RECOMMENDATION_CONTRACT.md`.

How the branches were merged and the results of the week 1 joint
Frontend–Backend testing are recorded in `docs/FE_BE_INTEGRATION_WEEK1.md`.

## Running the AI baselines

After running the Data pipeline, train Most Popular, MF and GMF on the
same temporal split:

```bash
python scripts/train_baseline.py
python scripts/evaluate_baselines.py
```

Hyperparameters are under `baselines` in `configs/cinematch.yaml`.
Checkpoints and the MSE/RMSE/MAE table are written to `outputs/baselines/`
and are not committed to GitHub. The theory and evaluation contract are
described in `docs/BASELINE_THEORY.md`.

## Running NCF, Hybrid NCF and the text demo

The week 1 smoke training runs use small data to confirm forward and
backpropagation. They are not yet final evaluation results on MovieLens.

```bash
python scripts/train_ncf.py
python scripts/train_hybrid_ncf.py
python experiments/text_encoder_demo.py
python -m pytest \
  tests/test_ncf.py \
  tests/test_hybrid_ncf.py \
  tests/test_text_encoder.py -v
```

- NCF theory: `docs/NCF_THEORY.md`.
- Design and feature dimensions: `docs/HYBRID_DESIGN.md`.
- Text artifact schema: `docs/TEXT_ARTIFACT_CONTRACT.md`.

Check the formatting of changes before committing:

```bash
git diff --check
git status --short
```

## Git policy for data

Do not commit:

- `data/raw/**`
- `data/processed/**`
- `outputs/**`
- virtual environments, caches or model checkpoints.

`.gitkeep` files are kept to preserve the directory structure.
