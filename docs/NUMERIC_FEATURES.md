# Hybrid Numeric Feature Handoff

## 1. Purpose

This handoff gives training and evaluation code one reproducible source for
the numeric side of Hybrid NCF. It prevents each model owner from inventing a
different release-year normalization or user-history formula.

The numeric vector has exactly 39 dimensions in this order:

```text
[19 movie genres, 1 normalized release year, 19 user genre-history values]
```

The text owner appends a separate 128-dimensional preference vector. Together
they form the fixed 167-dimensional Hybrid side-feature contract.

## 2. Inputs and data flow

```text
train.csv + movies.csv + id_mappings.json
                     |
                     v
       prepare_numeric_features.py
                     |
       +-------------+----------------+
       |             |                |
       v             v                v
movie numeric   user genre       frozen preprocessing
features        profiles         contract
```

Only `train.csv` fits preprocessing. The complete movie catalog is transformed
after the statistics have been frozen. Validation and test are consumers only.

## 3. Release-year feature

1. Select the unique movies occurring in `train.csv`.
2. Calculate the median over their non-missing release years.
3. Fill missing years with that train median.
4. Calculate the train-only mean and population standard deviation.
5. Transform every catalog movie using `(year - mean) / std`.

Release year is an auxiliary feature. It is never a hard rule for deleting a
rating because MovieLens timestamps and release metadata are not fully
consistent.

## 4. User genre-history feature

Each rating is mapped from the 1-5 scale to `[-1, 1]`:

```text
centered_rating = (rating - 3) / 2
```

For each user and genre, the feature is the mean centered rating over the
user's training movies carrying that genre. Multi-genre movies contribute to
each of their genres. A genre the user has not rated receives `0`, meaning
unknown/neutral rather than dislike.

This is a static train-only profile suitable for the current offline
experiment. A future online system may update profiles from newly submitted
ratings, but must version that serving policy separately.

## 5. Generate and load artifacts

Generate all three files:

```bash
python scripts/prepare_data.py
python scripts/prepare_numeric_features.py
```

Load and join them in Python:

```python
from cinematch.data.configuration import load_data_config
from cinematch.data.io import load_processed_ratings
from cinematch.features.numeric_features import (
    build_interaction_numeric_features,
    load_numeric_feature_artifacts,
)
from cinematch.features.hybrid_features import build_hybrid_side_features
import torch

config = load_data_config("configs/cinematch.yaml")
artifacts = load_numeric_feature_artifacts(
    config.paths.movie_numeric_features,
    config.paths.user_genre_profiles,
    config.paths.numeric_feature_preprocessor,
)
train = load_processed_ratings(config.paths.train)
numeric_features = build_interaction_numeric_features(
    train,
    artifacts.movie_features,
    artifacts.user_profiles,
)
assert numeric_features.shape == (len(train), 39)

text_vectors = torch.zeros((len(train), 128))  # replace with text output
side_features = build_hybrid_side_features(
    numeric_features[:, :19],
    numeric_features[:, 19:20],
    numeric_features[:, 20:39],
    text_vectors,
)
assert side_features.shape == (len(train), 167)
```

The generated files under `outputs/` are ignored by Git. Every team member
regenerates them from the committed code and MovieLens input.

## 6. Ownership and handoff

### Công Thành: Hybrid NCF and text

- Load these artifacts; do not refit their statistics in the training script.
- Join the 39 numeric values to each interaction.
- Generate the separate 128-dimensional text vector.
- Pass the four groups to `build_hybrid_side_features` in the documented
  order, producing exactly 167 side-feature dimensions.
- Save the numeric preprocessor beside the model checkpoint for serving.

### Sơn: group recommendation and evaluation

- Use `user_genre_profiles.csv` when constructing documented homogeneous,
  conflicting or mixed-preference test groups.
- Record the numeric preprocessor schema/data versions in experiment results.
- Continue to use the frozen files under `outputs/evaluation/` for candidate
  sets and ranking eligibility; numeric features do not redefine positives or
  negatives.
- Compare all models with the same test protocol and report evaluated and
  skipped users.

## 7. Verification and limitations

Run:

```bash
python -m pytest \
  tests/test_numeric_features.py \
  tests/test_feature_leakage.py -v
```

Tests cover dimensions, order, missing-year imputation, rating conversion,
finite values, artifact round-trip loading and train-only leakage protection.

MovieLens contains no free-text preference sentence. The 128 text dimensions
are therefore outside this Data artifact and must come from onboarding input
or a clearly documented pseudo-text experiment. No measured Hybrid quality is
claimed by this feature-generation step.
