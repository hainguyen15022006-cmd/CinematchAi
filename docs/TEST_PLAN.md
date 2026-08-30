# Test Plan — Week 1

Scope: verify 6 areas following the existing code structure in the repo (Data, AI Baseline,
NCF/Hybrid, Group Recommendation & Evaluation, Backend API, Frontend).

**Note on CI:** GitHub Actions (`ci.yml`) only runs `pytest` on the synthetic data
available in the test code; it **does not download or process MovieLens 100K**. Therefore the
following 4 tests SKIP themselves on CI (not fail): `tests/test_candidates.py`,
`tests/test_data_pipeline.py`, `tests/test_mapping.py` and
`tests/test_splitting.py`. To run the full test suite without skips, the data must be
downloaded and processed first:
```
python scripts/download_data.py
python scripts/prepare_data.py
python -m pytest -v
```

## 1. Data (owner: Hải Anh — automated via pytest)
Existing tests: `tests/test_dataset.py`, `tests/test_data_pipeline.py`,
`tests/test_mapping.py`, `tests/test_splitting.py`, `tests/test_auditing.py`,
`tests/test_configuration.py`

| Item | How to verify | Expected |
|---|---|---|
| Load MovieLens raw | `load_ml100k_ratings`, `load_ml100k_movies` | Correct row/column counts per `docs/DATA_DICTIONARY.md` |
| ID mapping | `build_id_mapping`, `apply_id_mappings` | Forward/reverse mapping consistent, no IDs lost |
| Temporal split | `test_splitting.py` | Train/val/test do not overlap per user, ratio ~80/10/10 |
| Data leakage | `test_auditing.py::validate_split_integrity` | Detects simulated leakage if present |
| Config loader | `test_configuration.py` | Reports a clear error when the config is wrong/missing fields |

Additional manual check: run `scripts/prepare_data.py` then `scripts/audit_splits.py` on a machine
that has downloaded the raw data, and confirm `outputs/eda/split_audit.json` has no serious warnings.
(The 4 tests depending on data artifacts above are skipped by CI — this part is the substitute evidence
from a manual run.)

## 2. AI Baseline (owner: Thành — automated via pytest)
Existing tests: `tests/test_baselines.py`

| Item | How to verify | Expected |
|---|---|---|
| Most Popular | Unit test on a small set | Correct ranking by rating frequency |
| MF forward | Output shape, NaN check | Output has correct shape `(batch,)`, no NaN |
| MF train smoke | 1-2 epochs on a small set | Loss decreases or does not increase abnormally |
| GMF forward/gradient | `nn.Module` forward + backward | Gradients exist, no NaN/Inf |

## 3. NCF / Hybrid NCF / Text Encoder (owner: Công Thành — automated via pytest)
Existing tests: `tests/test_ncf.py`, `tests/test_hybrid_ncf.py`, `tests/test_text_encoder.py`

| Item | How to verify | Expected |
|---|---|---|
| NCF forward | `test_ncf.py::test_ncf_forward_and_shapes` | Output shape `(batch,)`, no NaN/Inf |
| Hybrid NCF forward | `test_hybrid_ncf.py::test_hybrid_ncf_forward_shape` | Correct output shape, values within the valid rating range (1.0–5.0) |
| Text encoder | `test_text_encoder.py` | Encoded vector is finite, with the configured number of dimensions |
| Hybrid side-features | `cinematch.features.hybrid_features.build_hybrid_side_features` | Feature size matches `HYBRID_SIDE_FEATURE_DIM` |

## 4. Group Recommendation & Evaluation (owner: Sơn — automated via pytest)
Existing tests: `tests/test_group.py`, `tests/test_ranking.py`,
`tests/test_candidates.py`, `tests/test_model_scores.py`,
`tests/test_group_response.py`

| Item | How to verify | Expected |
|---|---|---|
| Average / Least Misery / Average Without Misery | `test_group.py` | Matches hand-computed results per `docs/GROUP_RECOMMENDATION_THEORY.md` |
| Disagreement score | `test_group.py::disagreement_score` | Correct formula, non-negative |
| Ranking metrics (Recall@K, NDCG@K, Hit Rate@K, Coverage) | `test_ranking.py` | Matches hand-computed examples |
| Candidate protocol | `test_candidates.py` | Seen items are not used as negatives; seed is reproducible |
| Model-to-Group adapter | `test_model_scores.py` | Movie IDs and each member's scores are aligned in the correct order |
| Group response contract | `test_group_response.py` | Response contains all required fields (group_score, minimum_score, disagreement, member_scores, explanations, fairness fields) |

## 5. Backend API (owner: Chúc — automated via pytest + manual via Swagger)
Existing tests: `tests/test_api_basic.py`, `tests/test_recommendation_mock_api.py`,
`tests/test_recommendation_schema.py`

| Item | How to verify | Expected |
|---|---|---|
| Health check | `GET /health` | 200 OK |
| Auth register/login | `test_api_basic.py` | Valid token returned; wrong password → 401 |
| Movies | `GET /movies` | Returns a list with the correct schema |
| Ratings | `POST /ratings` | Out-of-range ratings are rejected |
| Mock recommendation | `POST /recommend/mock` | Matches the contract in `docs/RECOMMENDATION_CONTRACT.md` |

Additional manual check: open `http://127.0.0.1:8000/docs`, try each endpoint by hand, and try the
error cases (expired token, missing fields, non-existent movie_id).

## 6. Frontend (owner: Dương — mainly manual, build verified via CI)
No automated tests in week 1 (as planned — "no need for a polished UI yet"). CI only
confirms that `npm run build` succeeds. For manual tests see `MANUAL_TEST_CASES.md`.

## 7. Integration (owner: Hoàng Anh)
- Data → AI: `scripts/train_baseline.py` runs on the data generated by `prepare_data.py`
  with no shape/mapping errors.
- Frontend → Backend → Mock Top 10: register/login → rating → call
  `/recommend/mock` → display results in the UI, no console errors.

## How to run all automated tests
```
python -m pytest -v
cd frontend && npm run build
```
To run the raw Data test suite in full (without skips), see the CI note at the top of this file.
