# Hybrid NCF Design, Week 1

## 1. Objective

Hybrid NCF keeps the user/movie embeddings of NCF and adds metadata so that the
model does not depend on IDs alone. The feature concatenation order is a fixed
contract between Data, AI training and Backend serving.

```text
[user embedding, movie embedding, genres, year, history, text]
```

## 2. Feature dimensions

| Component | Dimensions | Source |
|---|---:|---|
| User embedding | 32 | Learned from `user_index` |
| Movie embedding | 32 | Learned from `movie_index` |
| Movie genres | 19 | Genre columns in `movies.csv` |
| Release year | 1 | Normalized `release_year` |
| User genre history | 19 | Genre profile aggregated from past ratings |
| User-movie text interaction | 128 | User pseudo-text x movie text |
| **Side features** | **167** | 19 + 1 + 19 + 128 |
| **Total MLP input** | **231** | 32 + 32 + 167 |

The constants and concatenation order are in
`cinematch.features.hybrid_features`. Do not change the order on the Backend side.

## 3. Meaning of each group

### ID embeddings

`user_index` and `movie_index` represent the collaborative signal. The original
MovieLens IDs are still kept outside the model to join data and return responses.

### Genres

A 19-dimensional multi-hot vector taken directly from `movies.csv`. A movie can
have multiple values of 1.

### Release year

Only an auxiliary feature, not used as a condition for removing ratings. Before training,
the year must be scaled using statistics taken from the train set only.

### User history

A 19-dimensional vector representing genre preferences from ratings before the
prediction time. Future ratings from validation/test must not be used, as this would cause data
leakage.

### Text

MovieLens has no user-written preference text or movie overview. The Data
pipeline therefore creates explicitly labelled pseudo-text from train-only
genre preferences and creates movie documents from title plus genre names.
Both are encoded into 128 dimensions by the same encoder. Their element-wise
(Hadamard) product is the single 128-dimensional text block passed to Hybrid,
so the side-feature contract remains 167 rather than growing to 295.

The current encoder uses signed feature hashing. It is deterministic and
offline-friendly but does not understand semantic similarity. If a frozen
Sentence Transformer replaces it, the encoder artifact version must change;
the 128-dimensional output and Hadamard fusion contract remain fixed.

## 4. Forward contract

```python
side_features = build_hybrid_side_features(
    genres,
    normalized_year,
    history_profile,
    text_vector,
)
predictions = model(user_indices, movie_indices, side_features)
```

`side_features` must have shape `[batch_size, 167]`. Hybrid returns a tensor of shape
`[batch_size]` and each prediction lies in the range 1–5.

## 5. Current implementation status

Week 1 smoke training used numeric tensors of the correct shape to demonstrate
that the forward/backward pass runs. The Data pipeline now generates the real
39 numeric dimensions from MovieLens:

```bash
python scripts/prepare_numeric_features.py
python scripts/prepare_text_features.py
```

Training code loads the resulting numeric and text artifacts and frozen
preprocessor contract, then joins them to interactions with
`build_interaction_numeric_features`. It must append the separate 128-value
text interaction through `build_hybrid_side_features`; it must not recalculate
release-year or user-history statistics from validation or test.

This feature preparation is not itself a trained Hybrid result. Official
training, ablations and evaluation still require the shared evaluation
protocol and must report measured values rather than smoke-test values.

Run:

```bash
python scripts/train_hybrid_ncf.py
python -m pytest \
  tests/test_numeric_features.py \
  tests/test_feature_leakage.py \
  tests/test_hybrid_ncf.py \
  tests/test_text_encoder.py -v
```
