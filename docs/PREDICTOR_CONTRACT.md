# CineMatch Predictor Contract v1

## Purpose

`cinematch.serving.predictor.Predictor` is the only model-facing interface
used by offline evaluation and intended for the Backend model adapter. MF,
GMF and NCF must not require separate evaluator or API branches.

## Identifier boundary

The predictor accepts zero-based `user_index` and `movie_index`. The Backend
uses original MovieLens `movie_id`; it must convert IDs using
`data/processed/id_mappings.json` before scoring and decode the ranked indices
afterward. A database primary key is never a model index.

## Required calls

```python
scores = predictor.score(user_index, candidate_movie_indices)
scores = predictor.predict_for_new_user(ratings_by_movie_index, candidate_movie_indices)
```

Both methods return a one-dimensional NumPy array aligned exactly with the
candidate order. Scores are finite and clamped to [1, 5]. Candidate indices
must be unique, non-negative and inside the checkpoint catalog.

`ratings_by_movie_index` is a mapping such as `{0: 5.0, 49: 2.0}`. Ratings
must be finite values from 1 to 5. An empty profile, unknown movie or profile
with no overlap in train fails explicitly rather than returning a global
Top 10 silently.

## Checkpoint contract

`load_torch_predictor` requires:

```text
format_version
model_config.model          # mf, gmf or ncf
model_config.num_users
model_config.num_movies
model_state_dict
seed
```

Baseline checkpoints also include `data_version` and `config_hash`. NCF must
save `NCF.config()` under `model_config` to use the same loader.

## Cold-start behavior

The predictor selects up to 20 training users by overlap-adjusted rating
RMSE, normalizes their positive weights, averages their learned user
embeddings and scores the candidate batch with that folded-in vector. MF also
averages user bias. Offline simulation excludes the target MovieLens user
from its neighbour set.

## Failure behavior for Backend review

| Condition | Predictor behavior | Suggested API behavior |
|---|---|---|
| Missing/incompatible checkpoint | Raise load error | 503, use declared MF fallback if available |
| Unknown candidate movie | Raise `ValueError` | Filter only if mapping/catalog is stale and log it |
| Empty new-user ratings | Raise `ValueError` | 422, request onboarding ratings |
| No train overlap | Raise `ValueError` | 409, use disclosed Most Popular fallback |
| NaN/non-finite score | Evaluator rejects result | 503 and block persistence |

## Review required

Before Backend integration, Chúc must confirm the ID conversion and error
mapping. Công Thành must confirm NCF checkpoints use `NCF.config()`. Sơn must
confirm that the evaluator consumes scores without modifying the shared
candidate or ranking protocol.
