# Text encoder artifact contract v1

## 1. Purpose

The artifact allows training and serving to recreate exactly the same text encoder.
The week 1 baseline is deterministic feature hashing, so there are no learned
weights; the artifact only needs to store the configuration.

## 2. JSON schema

Example `artifacts/text_encoder.json`:

```json
{
  "schema_version": "1.0",
  "encoder_type": "signed_feature_hashing",
  "dimension": 128,
  "lowercase": true,
  "ngram_range": [1, 2]
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Contract version, currently `1.0` |
| `encoder_type` | string | Encoder algorithm |
| `dimension` | integer | Number of dimensions of each text vector |
| `lowercase` | boolean | Whether the text is converted to lowercase |
| `ngram_range` | array[integer] | Sizes of the unigrams/bigrams to be hashed |

The loader must reject artifacts with missing fields, a wrong version or a wrong encoder
type. Artifacts and checkpoints are regenerated, so they are not committed to Git.

## 3. Vector contract

- Input: a non-empty preference-text string (English in the MVP; the encoder is Unicode-aware).
- Output: `torch.float32`, shape `[128]`.
- Batch output: shape `[batch_size, 128]`.
- All elements are finite.
- The vector is L2-normalized.
- The same text with the same artifact must produce the same vector.

Empty text is rejected with a `ValueError` so that the Backend can ask the user to
re-enter it or use a separately agreed fallback.

## 4. Demo

```bash
python experiments/text_encoder_demo.py
```

The demo encodes three English preference sentences and prints the shape, dtype, finite
status and norm. The save/load contract test is in
`tests/test_text_encoder.py`.

## 5. Limitations

Feature hashing recognizes identical tokens and phrases but does not understand that two
sentences are synonymous. If it is replaced by PhoBERT or a Sentence Transformer, the team must
bump `schema_version`, declare the model name/revision and keep the principles of a
fixed dimension, deterministic inference and artifact validation.

## 6. Dataset-level text artifacts

The encoder artifact above describes only the encoder. The Data pipeline uses
it to create user and movie text matrices with:

```bash
python scripts/prepare_text_features.py
```

The generation rules, train-only provenance, seed, source-table counts and
Hadamard fusion policy are saved separately in
`outputs/features/text_feature_preprocessor.json`. The complete handoff is
documented in `docs/PSEUDO_TEXT_FEATURES.md`.
