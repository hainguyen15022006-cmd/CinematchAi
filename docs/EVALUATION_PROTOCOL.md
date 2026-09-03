# CineMatch Offline Evaluation Protocol

## 1. Objective

This document defines the unified offline evaluation procedure for
Most Popular, MF, GMF, NCF and Hybrid NCF.

Every model must use the same:

- Temporal split.
- ID mapping.
- Positive threshold.
- Seen-item policy.
- Candidate set.
- Negative sampling.
- Top K.
- Random seed.
- Ranking metrics.

The executable implementation is `scripts/evaluate_topk.py`. It loads every
model through `cinematch.serving.predictor.Predictor`, so model-specific code
cannot silently change candidate construction or metric definitions.

The committed defaults are `K=10`, 100 sampled negatives, positive rating
threshold 4.0 and seeds 42/43/44. These values are read from
`configs/cinematch.yaml`; CLI overrides are for explicit experiments and must
be reported with their results.

The candidate set must not be changed per model, because that would
make the comparison unfair.

## 2. Data sources

The data is taken from the artifacts produced by the Data pipeline:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/movies.csv`
- `data/processed/id_mappings.json`

The three interaction files share the same schema:

| Column | Meaning |
|---|---|
| `user_id` | Original MovieLens user ID |
| `movie_id` | Original MovieLens movie ID |
| `user_index` | Contiguous index used by the model |
| `movie_index` | Contiguous index used by the model |
| `rating` | Rating from 1.0 to 5.0 |
| `timestamp` | Rating time |

Offline evaluation and the model use `user_index` and `movie_index`.
The API and the interface use `movie_id`. The conversion must use
`id_mappings.json`.

## 3. Temporal split

MovieLens 100K is split temporally per user:

- The oldest 80% of interactions: train.
- The next 10%: validation.
- The newest 10%: test.

Current sizes:

| Partition | Number of rows | Number of users |
|---|---:|---:|
| Train | 80,014 | 943 |
| Validation | 10,132 | 943 |
| Test | 9,854 | 943 |

Only train is used to update the model weights.

Validation is used to select checkpoints or hyperparameters.

Test is used only to report the final results. Test must not be
used for training, early stopping or configuration selection.

The current audit confirms:

- No interaction overlap between partitions.
- No train-validation temporal violations.
- No validation-test temporal violations.
- No user cold-start.

## 4. Seen items

For each user:

```text
seen_items = train_movie_indices ∪ validation_movie_indices
```
