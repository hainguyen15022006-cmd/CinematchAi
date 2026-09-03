# CineMatch — Week 1 AI Baseline Results

## 1. Experiment objective

The week 1 experiment builds reference models against which the deeper
models of later stages will be evaluated. Three models are compared:

- **Most Popular:** a non-personalized baseline that ranks movies by their
  mean rating, smoothed by the number of ratings.
- **Matrix Factorization (MF):** learns a user embedding, a movie embedding,
  a user bias and a movie bias to predict the rating.
- **Generalized Matrix Factorization (GMF):** multiplies the two embeddings
  element-wise, then uses a linear layer to learn a weight for each latent
  dimension.

The results in this document are the reference mark. NCF and Hybrid NCF are
only considered an improvement when they are trained and evaluated on the
same data, the same candidate protocol and the same metrics.

## 2. Data and evaluation protocol

The team uses MovieLens 100K with:

| Attribute | Value |
|---|---:|
| Number of ratings | 100,000 |
| Number of users | 943 |
| Number of movies | 1,682 |
| Rating range | 1–5 |
| Sparsity of the user–movie matrix | 93.6953% |

The data is split temporally per user, not with a random split:

| Partition | Number of ratings | Role |
|---|---:|---|
| Train | 80,014 | Update model parameters |
| Validation | 10,132 | Select the best epoch and checkpoint |
| Test | 9,854 | Report the final results |

Mandatory rules:

1. Only `train.csv` is used to update the weights.
2. Early stopping and checkpoint selection are based on validation RMSE.
3. `test.csv` must not be used to tune hyperparameters.
4. All three baselines use the same `user_index`, `movie_index` and temporal
   split produced by the Data pipeline.
5. Predictions are clamped to the 1–5 range only when computing metrics, not
   in the forward pass during training.

## 3. Run configuration

The configuration is read from `configs/cinematch.yaml`:

| Hyperparameter | Value |
|---|---:|
| Random seed | 42 |
| Embedding dimension | 32 |
| Batch size | 1,024 |
| Maximum epochs | 20 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Early-stopping patience | 3 epochs |
| Optimizer | Adam |
| Training loss | MSE |
| Most Popular prior count | 20 |

The verification experiment was run with Python 3.12 on CPU. The specific
versions of Python, PyTorch, pandas and NumPy should be recorded for every
official run because they may differ between machines.

The environment can be printed with:

```bash
python --version
python -c "import torch, pandas, numpy; print('torch=', torch.__version__); print('pandas=', pandas.__version__); print('numpy=', numpy.__version__)"
```

## 4. Verified results

| Model | Validation RMSE | Validation MAE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Most Popular | 1.0780 | 0.8656 | 1.1115 | 0.8989 |
| MF | **0.9781** | 0.7694 | 1.0285 | **0.8148** |
| GMF | 0.9799 | **0.7692** | **1.0268** | 0.8149 |

Results are rounded to four decimal places. When run on a different PyTorch
version or hardware, results may fluctuate slightly even with the same seed.

Best checkpoints in the verification run:

| Model | Best epoch |
|---|---:|
| MF | 16 |
| GMF | 11 |

## 5. Discussion of results

### 5.1 Most Popular is a reasonable lower bound

Most Popular reaches a test RMSE of 1.1115. This model does not use the user's
identity, so every user receives the same predicted score for a given movie.
It is still necessary because it proves that a personalized model must beat a
simple strategy based only on popularity.

### 5.2 Both MF and GMF improve over Most Popular

MF reduces test RMSE from 1.1115 to 1.0285. GMF reduces it to 1.0268. This
shows that the user and movie embeddings have learned personal preference
signals that Most Popular cannot represent.

In terms of test RMSE, GMF improves by about 7.6% over Most Popular:

```text
(1.1115 - 1.0268) / 1.1115 × 100% ≈ 7.6%
```

### 5.3 It cannot yet be concluded that GMF outperforms MF

GMF's test RMSE is only about 0.0017 lower than MF's. This difference is very
small and may change between runs. A correct report should conclude that MF
and GMF perform roughly equivalently in the week 1 configuration, and should
not claim that GMF is definitively better in a statistical sense.

### 5.4 How GMF is used in CineMatch

The original GMF in Neural Collaborative Filtering is usually presented for
implicit feedback. CineMatch week 1 is solving an explicit 1–5 rating
prediction problem, so the model uses a linear output and MSE loss. This is a
GMF variant for regression and must be stated clearly when presenting.

## 6. What do the metrics mean?

- **MSE:** the mean squared error, which is also the loss used for training.
- **RMSE:** the square root of MSE; it penalizes large errors heavily and has
  the same unit as the rating.
- **MAE:** the mean absolute error. A test MAE of about 0.815 means the
  predictions are off by about 0.815 rating points on average.

Lower RMSE is better. However, RMSE/MAE only measure rating prediction; they
do not directly measure the quality of the Top-K list.

## 7. Artifacts produced after training

The command:

```bash
python scripts/train_baseline.py --device cpu
```

produces:

```text
outputs/baselines/
├── mf_checkpoint.pt
├── gmf_checkpoint.pt
└── metrics.json
```

- A checkpoint contains the model config, state dictionary, best epoch, seed
  and format version.
- `metrics.json` contains validation metrics, test metrics and the training
  history.
- `outputs/` and checkpoints are not committed to GitHub; this results
  document is committed to preserve the experimental evidence.

To print the results table:

```bash
python scripts/evaluate_baselines.py
```

## 8. Testing

Run the baseline tests on their own:

```bash
python -m pytest tests/test_baselines.py -v
```

Verified result for week 1:

```text
10 passed
```

The tests cover Most Popular, excluding already-seen movies, fallback for
unknown movies, the forward and backward passes of MF/GMF, input shape checks,
regression metrics and a smoke test proving that MF can learn.

Before merging, additionally run:

```bash
python -m pytest -v
git diff --check
git status --short
```

## 9. Remaining limitations and next week's work

1. Add warm-start metrics because validation and test still contain some
   movies that do not appear in train.
2. Do not use test metrics to tune the learning rate, embedding size or
   number of epochs.
3. Build a shared candidate set and negative-sampling protocol.
4. Add Hit Rate@10, Recall@10 and NDCG@10 for Top-K recommendation.
5. Compare the baselines with NCF and Hybrid NCF using the same split and
   protocol.
6. Hand over predicted member scores to Group Recommendation to try Average,
   Least Misery and Average Without Misery.

## 10. Week 1 conclusion

Most Popular, MF and GMF have all been implemented, tested and run
successfully on the shared temporal split. MF and GMF improve clearly over the
non-personalized baseline, proving that the embeddings do learn user–movie
signals.

These results are sufficient as the week 1 baseline mark. They are not the
final CineMatch results, because the system still needs NCF/Hybrid NCF, Top-K
evaluation, cold-start handling and the score aggregation strategies for
groups.

## 11. Week 2 reproducibility update (seeds 42, 43 and 44)

The baseline trainer now creates one checkpoint and manifest per seed. The
manifest records the data hash, config hash, seed, model configuration,
checkpoint path and metrics. Values below are population mean ± standard
deviation across all three runs; no seed was discarded.

| Model | Test RMSE | Test MAE |
|---|---:|---:|
| Most Popular | 1.1115 ± 0.0000 | 0.8989 ± 0.0000 |
| MF | **1.0270 ± 0.0042** | **0.8145 ± 0.0035** |
| GMF | 1.0282 ± 0.0011 | 0.8156 ± 0.0016 |

MF and GMF remain effectively tied. Their mean RMSE difference is about
0.0012, which is smaller than MF's run-to-run standard deviation; therefore
the results do not support claiming that either model is definitively better.

## 12. Shared Top-K results

The evaluator uses the same per-user candidate protocol for every model:
train + validation items are seen, test ratings >= 4 are relevant, and each
eligible user receives all relevant items plus 100 sampled negatives. There
are 836 evaluated users and 107 skipped users without a positive test item.

| Model | Recall@10 | NDCG@10 | HitRate@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Most Popular | 0.2649 ± 0.0232 | 0.2357 ± 0.0162 | 0.5841 ± 0.0225 | 0.1793 ± 0.0087 |
| MF | 0.2829 ± 0.0143 | 0.2486 ± 0.0119 | 0.6423 ± 0.0188 | 0.5351 ± 0.0131 |
| GMF | **0.2927 ± 0.0150** | **0.2580 ± 0.0117** | **0.6447 ± 0.0176** | **0.5862 ± 0.0261** |

The three evaluation seeds change both the model seed and deterministic
negative sample. Thus the standard deviation represents the complete
repeated protocol, not training randomness alone. GMF has the highest means,
but MF and GMF overlap substantially across runs, so this is descriptive and
not a claim of statistical superiority.

The NCF row is intentionally absent until Công Thành supplies a real
MovieLens checkpoint. A smoke-trained or random-tensor checkpoint must not be
inserted into this table.

## 13. Cold-start baseline

MF seed 42 was evaluated with nearest-neighbour embedding fold-in. For each
simulated new user, only the earliest 5 or 10 training ratings form the
profile, and that user's original MovieLens identity is excluded from the 20
neighbours. Test ratings remain held out.

| Known ratings | Recall@10 | NDCG@10 | HitRate@10 | Coverage@10 |
|---:|---:|---:|---:|---:|
| 5 | 0.2241 | 0.1942 | 0.5574 | 0.1623 |
| 10 | 0.2122 | 0.1802 | 0.5395 | 0.1653 |

Ten ratings did not improve ranking metrics in this run. This unfavorable
result is retained: the selected earliest ratings may be less informative,
and KNN fold-in is a pragmatic demo mechanism rather than a learned optimal
new-user encoder. See `docs/COLD_START.md` for limitations.
