# Pseudo-Text and Movie-Text Feature Handoff

## 1. Scope and honesty statement

MovieLens 100K contains ratings, titles and genres, but no free-form user
preference sentence and no movie overview. CineMatch therefore generates
controlled English pseudo-text from training ratings and movie documents from
catalog fields. These strings are synthetic features, not user-authored data.

No validation/test rating participates in user text generation. This step
creates model inputs only; it does not claim a recommendation-quality result.

## 2. User pseudo-text rule

For every user and each of the 19 genres:

1. Join `train.csv` to `movies.csv` by `movie_id`.
2. Calculate the genre mean rating and observation count.
3. Exclude the MovieLens `unknown` indicator because it is not a semantic
   preference genre. It remains present in movie metadata.
4. Keep genres whose mean rating is at least the configured positive threshold
   (`4.0`) AND that the user rated at least `minimum_genre_observations`
   times (`3` in `configs/cinematch.yaml`). Without the count condition a
   single 5-star rating of a rare genre (Film-Noir has only 24 catalog
   movies) outranks well-observed genres; with it, the selected genres are
   backed by repeated evidence.
5. Sort by mean rating descending, count descending and fixed MovieLens genre
   order.
6. Keep at most three genres.
7. If none qualifies, use the highest-rated observed genres (no count
   condition) and record `used_fallback=true`. On the real split this
   affects 235 of 943 users (24.92%); the flag is stored per user and
   reported in `feature_coverage_report.json`.
8. Choose an English template deterministically from `seed + user_id`.

Example:

```text
I enjoy Action, Sci-Fi and Thriller movies.
```

The seed changes wording only; it never changes selected genres.

## 3. Movie-text rule

Every catalog movie receives exactly one document:

```text
{title}. Genres: {genre names}.
```

Example:

```text
Toy Story (1995). Genres: Animation, Children and Comedy.
```

No plot, runtime, poster or external metadata is fabricated.

## 4. Encoding and the 167-dimensional contract

The same versioned `PreferenceTextEncoder` encodes user and movie strings:

```text
user vectors:  [943, 128]
movie vectors: [1682, 128]
```

For a user-movie interaction:

```python
text_interaction = user_text_vector * movie_text_vector
```

This Hadamard product remains 128-dimensional. Hybrid then receives:

```text
19 genres + 1 year + 19 history + 128 text interaction = 167
```

The current hashing encoder mainly captures matching words/ngrams and is not a
semantic language model. A replacement encoder must bump its artifact version,
produce 128 dimensions and preserve the fusion contract.

## 5. Generate and load

```bash
python scripts/prepare_data.py
python scripts/prepare_numeric_features.py
python scripts/prepare_text_features.py
python scripts/report_feature_coverage.py
```

Generated files under `outputs/features/`:

```text
user_pseudo_text.csv
movie_text.csv
user_text_vectors.npz
movie_text_vectors.npz
text_feature_preprocessor.json
```

Load them with `load_text_feature_artifacts` and build batch features with
`build_interaction_text_features`. The function validates ID/index pairs before
selecting and multiplying vectors.

## 6. Handoff to Công Thành

- Load all five text artifacts; do not regenerate pseudo-text inside training.
- Use `build_interaction_text_features` to obtain `[batch, 128]`.
- Use the 39-column numeric artifact from Data.
- Pass numeric slices and the text interaction to
  `build_hybrid_side_features`, which returns `[batch, 167]`.
- Use a feature mask for E3 (text off) and E4 (text on), without changing the
  model class or the data contract.
- Save both numeric/text preprocessor versions with every checkpoint.

## 7. Handoff to Sơn

- Record seed, positive threshold, fallback count, encoder and data version.
- Use the same E3/E4 split, candidates and seeds before comparing text.
- Report that pseudo-text is derived from train ratings and is not real user
  language.
- Do not interpret hashing similarity as semantic understanding.
- Report unfavorable or statistically unclear results as measured.

## 8. Verification

```bash
python -m pytest tests/test_pseudo_text.py -v
python -m pytest \
  tests/test_numeric_features.py \
  tests/test_feature_leakage.py \
  tests/test_pseudo_text.py \
  tests/test_text_encoder.py \
  tests/test_hybrid_ncf.py -v
```

Tests cover selection, fallback, deterministic templates, shapes, finite
float32 vectors, safe save/load, ID mapping, train-only provenance and the
final 167-dimensional Hybrid contract.
