# Neural Collaborative Filtering Theory for CineMatch

## 1. Problem

NCF predicts a user's rating for a movie. The model's input is not the
original `user_id` and `movie_id` but the contiguous `user_index` and
`movie_index` produced by the Data pipeline.

```text
user_index ──> user embedding ──┐
                                ├─> concatenate ─> MLP ─> rating 1–5
movie_index ─> movie embedding ─┘
```

An embedding turns a discrete index into a learnable real-valued vector.
In the week 1 configuration, both the user embedding and the movie embedding have 32 dimensions.

## 2. How does NCF differ from MF and GMF?

- MF takes the dot product of the two embeddings. The user–movie relationship is
  modeled by a single fixed linear interaction.
- GMF multiplies the two embeddings element-wise and then uses a linear layer to
  learn weights for the latent dimensions.
- NCF concatenates the two embeddings and passes them through several fully connected layers. ReLU allows
  the MLP to represent more complex non-linear relationships.

NCF is not automatically better than MF/GMF. The models must be trained and
evaluated on the same temporal split before drawing conclusions.

## 3. MLP, activation and dropout

Each block currently consists of:

```text
Linear -> LayerNorm -> ReLU -> Dropout
```

- `Linear` learns how to combine the input features.
- `LayerNorm` keeps the activation distribution stable and still works when the last
  batch contains only a single sample.
- `ReLU` introduces non-linearity.
- `Dropout` randomly disables a fraction of neurons during training, making it harder for the model to
  depend on a few neurons and reducing the risk of overfitting.

When `model.eval()` is called, Dropout is automatically disabled.

## 4. Output rating

The final linear layer produces an unbounded logit. CineMatch converts the logit
to a MovieLens rating with:

```text
predicted_rating = 1 + 4 * sigmoid(logit)
```

Since the sigmoid lies in the range 0–1, the final prediction always lies in the range
1–5. Tests must check the shape, finite values, gradients and rating range.

## 5. Week 1 training smoke

The training smoke run only confirms that forward, loss and backpropagation work.
The random data in the smoke test must not be used to report model
quality. Next week NCF must use `train.csv`, select the model with
`validation.csv` and evaluate only once at the end on `test.csv`.

Run:

```bash
python scripts/train_ncf.py
python -m pytest tests/test_ncf.py -v
```

## 6. Concepts to distinguish

- A cold-start user already has an embedding slot but does not yet have enough ratings to learn a
  good vector.
- An unknown user has no index in the mapping and needs a fallback before the
  embedding is called.
- Overfitting occurs when the train loss keeps decreasing while the validation loss
  increases.
- A decreasing smoke loss only proves that the pipeline can optimize, not that the
  recommendations are good.
