# Cold Start for New CineMatch Users

## Problem

A newly registered CineMatch account is absent from MovieLens 100K and has no
row in the MF, GMF or NCF user embedding table. Passing the Backend user ID as
a model index would either fail or, worse, score the wrong MovieLens user.

## Week 2 solution

During onboarding, the user rates known MovieLens movies. The predictor:

1. Converts original `movie_id` values to `movie_index`.
2. Compares the profile only with `train.csv` interactions.
3. Ranks training users by overlap-adjusted rating RMSE.
4. Selects at most 20 nearest users.
5. Computes a normalized weighted average of their user embeddings.
6. Scores unseen candidate movies with the folded-in embedding.

Validation and test ratings are never used to find neighbours. Deterministic
tie-breaking uses the lower `user_index`.

## Offline evidence

MF seed 42 was evaluated over the 836 users with at least one positive test
item. The simulated target is excluded from its own neighbour pool.

| Profile | Recall@10 | NDCG@10 | HitRate@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Earliest 5 train ratings | 0.2241 | 0.1942 | 0.5574 | 0.1623 |
| Earliest 10 train ratings | 0.2122 | 0.1802 | 0.5395 | 0.1653 |

The 10-rating profile is not better in this experiment. This is reported
unchanged rather than hidden. More ratings are not automatically more useful
when the selected items are uninformative or the fixed KNN rule is weak.

## Limitations

- KNN can only use rated movies that overlap MovieLens train interactions.
- Averaging embeddings can erase unusual or multi-modal preferences.
- The model's item embeddings were still trained on the complete training
  population, including the simulated user's historical contribution; this
  is not a strict retrain-from-scratch user holdout.
- The current distance treats every supplied rating equally and does not
  model confidence or rating recency.
- It is a transparent MVP mechanism, not evidence that KNN is the optimal
  cold-start method.

## Reproduce

```bash
python scripts/train_baseline.py --device cpu
python scripts/evaluate_cold_start.py
```

The committed config supplies seeds 42/43/44, profile sizes 5/10 and K=20
neighbours. The generated JSON is `outputs/evaluation/cold_start.json`.

## Backend fallback

If fold-in cannot find overlap, the API should return a clear onboarding
error or use Most Popular only as an explicitly disclosed fallback. It must
not invent an embedding or silently map the account to an arbitrary
MovieLens user.
