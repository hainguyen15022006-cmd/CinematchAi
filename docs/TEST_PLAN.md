# Test Plan — Tuần 1

Phạm vi: kiểm tra 6 mảng theo đúng cấu trúc code hiện có trên repo (Data, AI Baseline,
NCF/Hybrid, Group Recommendation & Evaluation, Backend API, Frontend).

**Lưu ý về CI:** GitHub Actions (`ci.yml`) chỉ chạy `pytest` trên dữ liệu synthetic có
sẵn trong code test, **không tự tải MovieLens 100K**. Do đó 3 test sau sẽ tự SKIP trên
CI (không phải fail): `tests/test_data_pipeline.py`, `tests/test_mapping.py`,
`tests/test_splitting.py`. Để chạy đầy đủ bộ test Data (không skip), phải chạy thủ
công trên máy đã có sẵn raw data:
```
python scripts/download_data.py
python -m pytest -v tests/test_data_pipeline.py tests/test_mapping.py tests/test_splitting.py
```

## 1. Data (owner: Hải Anh — automated qua pytest)
Test hiện có: `tests/test_dataset.py`, `tests/test_data_pipeline.py`,
`tests/test_mapping.py`, `tests/test_splitting.py`, `tests/test_auditing.py`,
`tests/test_configuration.py`

| Hạng mục | Cách kiểm tra | Kỳ vọng |
|---|---|---|
| Load MovieLens raw | `load_ml100k_ratings`, `load_ml100k_movies` | Đúng số dòng/cột theo `docs/DATA_DICTIONARY.md` |
| ID mapping | `build_id_mapping`, `apply_id_mappings` | Mapping xuôi/ngược nhất quán, không mất ID |
| Temporal split | `test_splitting.py` | Train/val/test không chồng lấn theo user, tỷ lệ ~80/10/10 |
| Data leakage | `test_auditing.py::validate_split_integrity` | Phát hiện được nếu có leakage giả lập |
| Config loader | `test_configuration.py` | Báo lỗi rõ ràng khi config sai/thiếu field |

Manual bổ sung: chạy `scripts/prepare_data.py` rồi `scripts/audit_splits.py` trên máy
đã tải raw data, xác nhận `outputs/eda/split_audit.json` không có cảnh báo nghiêm trọng.
(3 test raw-data ở trên bị CI skip — phần này là bằng chứng thay thế chạy thủ công.)

## 2. AI Baseline (owner: Thành — automated qua pytest)
Test hiện có: `tests/test_baselines.py`

| Hạng mục | Cách kiểm tra | Kỳ vọng |
|---|---|---|
| Most Popular | Unit test trên tập nhỏ | Xếp hạng đúng theo tần suất rating |
| MF forward | Shape output, kiểm tra NaN | Output đúng shape `(batch,)`, không NaN |
| MF train smoke | 1-2 epoch trên tập nhỏ | Loss giảm hoặc không tăng bất thường |
| GMF forward/gradient | `nn.Module` forward + backward | Gradient tồn tại, không NaN/Inf |

## 3. NCF / Hybrid NCF / Text Encoder (owner: Công Thành — automated qua pytest)
Test hiện có: `tests/test_ncf.py`, `tests/test_hybrid_ncf.py`, `tests/test_text_encoder.py`

| Hạng mục | Cách kiểm tra | Kỳ vọng |
|---|---|---|
| NCF forward | `test_ncf.py::test_ncf_forward_and_shapes` | Output shape `(batch,)`, không NaN/Inf |
| Hybrid NCF forward | `test_hybrid_ncf.py::test_hybrid_ncf_forward_shape` | Output shape đúng, giá trị nằm trong khoảng rating hợp lệ (1.0–5.0) |
| Text encoder | `test_text_encoder.py` | Vector encode trả về hữu hạn (finite), đúng số chiều cấu hình |
| Hybrid side-features | `cinematch.features.hybrid_features.build_hybrid_side_features` | Kích thước feature khớp `HYBRID_SIDE_FEATURE_DIM` |

## 4. Group Recommendation & Evaluation (owner: Sơn — automated qua pytest)
Test hiện có: `tests/test_group.py`, `tests/test_ranking.py`, `tests/test_group_response.py`

| Hạng mục | Cách kiểm tra | Kỳ vọng |
|---|---|---|
| Average / Least Misery / Average Without Misery | `test_group.py` | Khớp kết quả tính tay theo `docs/GROUP_RECOMMENDATION_THEORY.md` |
| Disagreement score | `test_group.py::disagreement_score` | Đúng công thức, không âm |
| Ranking metrics (Recall@K, NDCG@K, Hit Rate@K, Coverage) | `test_ranking.py` | Khớp ví dụ tính tay |
| Group response contract | `test_group_response.py` | Response chứa đủ field bắt buộc (group_score, minimum_score, disagreement, member_scores, explanations, fairness fields) |

## 5. Backend API (owner: Chúc — automated qua pytest + manual qua Swagger)
Test hiện có: `tests/test_api_basic.py`, `tests/test_recommendation_mock_api.py`,
`tests/test_recommendation_schema.py`

| Hạng mục | Cách kiểm tra | Kỳ vọng |
|---|---|---|
| Health check | `GET /health` | 200 OK |
| Auth register/login | `test_api_basic.py` | Token trả về hợp lệ; sai mật khẩu → 401 |
| Movies | `GET /movies` | Trả danh sách đúng schema |
| Ratings | `POST /ratings` | Rating ngoài khoảng hợp lệ bị từ chối |
| Mock recommendation | `POST /recommend/mock` | Đúng contract trong `docs/RECOMMENDATION_CONTRACT.md` |

Manual bổ sung: mở `http://127.0.0.1:8000/docs`, thử từng endpoint bằng tay, thử các
case lỗi (token hết hạn, thiếu field, movie_id không tồn tại).

## 6. Frontend (owner: Dương — chủ yếu manual, build kiểm tra qua CI)
Chưa có test tự động ở tuần 1 (đúng theo kế hoạch — "chưa cần giao diện đẹp"). CI chỉ
xác nhận `npm run build` không lỗi. Manual test xem `MANUAL_TEST_CASES.md`.

## 7. Integration (owner: Hoàng Anh)
- Data → AI: `scripts/train_baseline.py` chạy được trên dữ liệu do `prepare_data.py`
  sinh ra, không lỗi shape/mapping.
- Frontend → Backend → Mock Top 10: đăng ký/đăng nhập → rating → gọi
  `/recommend/mock` → hiển thị kết quả trên UI, không lỗi console.

## Cách chạy toàn bộ automated test
```
python -m pytest -v
cd frontend && npm run build
```
Để chạy đầy đủ (không skip) bộ Data test raw, xem ghi chú CI ở đầu file này.
