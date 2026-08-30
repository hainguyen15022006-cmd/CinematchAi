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
| Preference text | 128 | `PreferenceTextEncoder` |
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

The onboarding preference sentence (English) is converted into a 128-dimensional vector. The week 1
baseline uses signed feature hashing to verify the contract. This method
does not capture deep semantics and may be replaced by a pretrained encoder in a
later phase if the team has time.

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

## 5. Week 1 scope

Smoke training uses fake metadata of the correct shape and real text vectors to demonstrate
that the entire forward/backward pass runs. It is not yet a MovieLens experimental
result. Joining real metadata and computing leakage-free history belong to the
next training pipeline.

Run:

```bash
python scripts/train_hybrid_ncf.py
python -m pytest tests/test_hybrid_ncf.py tests/test_text_encoder.py -v
```
