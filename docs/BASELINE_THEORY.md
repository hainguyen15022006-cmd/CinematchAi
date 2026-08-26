# CineMatch AI baseline theory

## Purpose

Most Popular, Matrix Factorization (MF), and Generalized Matrix
Factorization (GMF) are controlled reference models. NCF and Hybrid NCF
should be judged against them under the same data split and metrics.

## User-item matrix and embeddings

The user-item matrix contains observed ratings at user/movie
intersections. It is sparse because users rate only a small part of the
catalog. Embeddings replace discrete user and movie indices with short,
trainable latent vectors. Backpropagation updates these vectors to
reduce prediction error.

Only `train.csv` updates model parameters. `validation.csv` controls
model selection and early stopping. `test.csv` is reserved for final
reporting. Baselines use the shared per-user temporal split and never
create a separate random split.

## Most Popular

Most Popular is non-personalized. It uses the Bayesian-smoothed score:

```text
score_i = (v_i * R_i + m * C) / (v_i + m)
```

`R_i` is the movie mean, `v_i` its training rating count, `C` the global
training mean, and `m` the configured prior count. Smoothing prevents a
movie with one lucky rating from automatically ranking first. Unknown
items fall back to `C`, and already-seen movies can be removed from
Top-K recommendations.

## Matrix Factorization

Biased MF predicts explicit ratings as follows:

```text
r_hat_ui = global_mean + user_bias_u + movie_bias_i + dot(p_u, q_i)
```

`p_u` and `q_i` are trainable embeddings. User and movie biases capture
systematic effects such as generous raters or broadly popular movies.

## Generalized Matrix Factorization

GMF computes an element-wise interaction and learns its output weights:

```text
z_ui = p_u element-wise-multiply q_i
r_hat_ui = global_mean + linear(z_ui)
```

MF sums latent interactions with fixed weights. GMF learns a different
weight for each latent dimension while keeping the shared interface
`model(user_indices, movie_indices)`.

## Loss and metrics

- MSE is the mean squared prediction error and training loss.
- RMSE is the square root of MSE and penalizes large errors strongly.
- MAE is the mean absolute error and is interpretable in rating units.

Predictions are clamped to MovieLens' 1-5 range only while reporting
metrics, not inside the training forward pass. Later Top-K evaluation
must use the same candidate construction and negative sampling for all
models.

## Reproducibility contract

- Use `user_index`, `movie_index`, and `rating` from processed splits.
- Fit on train, select on validation, and report final results on test.
- Record the seed and all hyperparameters.
- Save versioned checkpoints containing config and model state.
- Do not commit processed data, outputs, or checkpoints.

## References

1. Koren, Bell, and Volinsky. *Matrix Factorization Techniques for
   Recommender Systems*. IEEE Computer, 2009.
2. He et al. *Neural Collaborative Filtering*. WWW, 2017.
3. Harper and Konstan. *The MovieLens Datasets: History and Context*.
   ACM Transactions on Interactive Intelligent Systems, 2015.
