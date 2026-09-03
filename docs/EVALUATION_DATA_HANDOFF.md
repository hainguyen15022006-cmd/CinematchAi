# Data-to-Evaluation Handoff

## 1. Purpose

This handoff gives every recommendation model the same offline-evaluation
inputs. The files are derived from the versioned temporal split and must not
be edited manually.

On a clean clone, first install the project and obtain MovieLens as described
in the README. Then run the complete artifact pipeline in this order:

```bash
python scripts/prepare_data.py
python scripts/audit_splits.py
python scripts/prepare_evaluation_data.py
python scripts/prepare_numeric_features.py
python scripts/prepare_text_features.py
python scripts/report_feature_coverage.py
python scripts/create_data_manifest.py
```

`prepare_evaluation_data.py` writes four ignored artifacts under
`outputs/evaluation/`. The final manifest requires these files and the numeric
and text artifacts, so it must not be generated earlier. If only regenerating
the evaluation handoff, that script can run independently; regenerate the
manifest after all artifacts are ready.

For recipient confirmation, send the test result and the CLI's
`Reproducibility fingerprint (cinematch-content-v1)` line. Compare
`reproducibility.content_sha256`, not the first file checksum or the whole
manifest: JSON generation timestamps legitimately change between machines.

## 2. Shared protocol

| Setting | Value |
|---|---|
| Data version | `ml100k-temporal-v1` |
| Seen-item policy | Train items union validation items |
| Positive test threshold | Rating greater than or equal to 4.0 |
| ID space for model evaluation | Zero-based `user_index` and `movie_index` |
| Top K | 10 |
| Negative samples per evaluable user | 100 |
| Random seed | 42 |

Only train updates model weights. Validation interactions are included in
`seen_items` because they happened before test and must not be recommended as
unseen items. Test interactions are never included in `seen_items`.

## 3. Files

### `catalog.json`

Contains every mapped MovieLens movie. Each entry connects the original
`movie_id` used by the Backend with the zero-based `movie_index` used by the
models. `observed_in_train` supports separate full-catalog and warm-start
evaluation.

### `seen_items.json`

Contains one entry for every mapped user. `movie_indices` is the sorted union
of the user's train and validation movie indices. Candidate construction must
exclude all of these indices.

### `positive_test_items.json`

Contains one entry for every mapped user. `movie_indices` contains test movies
whose rating is at least 4.0. A user without a positive item remains in the
file with an empty list; the user is reported as skipped rather than silently
deleted.

### `evaluation_data_summary.json`

Records the protocol, split sizes, eligibility counts, positive count and
cold-start statistics. It also stores the exact user indices skipped because
their test partition has no positive item.

## 4. Audited MovieLens 100K result

| Statistic | Value |
|---|---:|
| Total users | 943 |
| Evaluable users | 836 |
| Users skipped without a positive test item | 107 |
| Catalog movies | 1,682 |
| Movies observed in train | 1,611 |
| Validation cold-start movies | 33 |
| Test cold-start movies | 45 |
| Validation cold-start rows | 36 |
| Test cold-start rows | 52 |

The identity `836 + 107 = 943` is checked by tests. These values are generated
from the artifacts and are not model results.

## 5. Responsibilities

The Data owner maintains the split, mappings and these generated handoff
inputs. The Evaluation owner uses them to construct one candidate set per
evaluable user and must reuse that same candidate set for Most Popular, MF,
GMF, NCF and Hybrid NCF.

The Evaluation owner must report the evaluated and skipped user counts next to
Recall@10, NDCG@10, Hit Rate@10 and Coverage. Cold-start movies are retained in
the full evaluation; a warm-start result may be reported separately.

Python consumers can load and validate the complete bundle with:

```python
from pathlib import Path

from cinematch.data.evaluation_handoff import load_evaluation_handoff

handoff = load_evaluation_handoff(Path("outputs/evaluation"))
catalog = handoff.catalog
seen_items = handoff.seen_items
positive_test_items = handoff.positive_test_items
summary = handoff.summary
```

## 6. Reproducibility rules

- Do not edit generated JSON files manually.
- Do not replace model indices with original MovieLens IDs during scoring.
- Do not use test interactions to update model weights or choose a checkpoint.
- Do not change the positive threshold or candidate seed for one model only.
- Regenerate all files whenever the data version or split changes.
- Treat the generated files as local artifacts; commit their generator, tests
  and documentation instead of committing the large JSON outputs.
