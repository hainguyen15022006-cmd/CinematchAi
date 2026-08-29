# Test Plan — Tuần 1

Phạm vi: kiểm tra 4 mảng theo đúng cấu trúc code hiện có trên repo (Data, AI Baseline,
Backend API, Frontend). Mảng NCF/Hybrid và Group Recommendation chưa có code/test
tương ứng tại thời điểm viết plan này — cập nhật khi Công Thành/Hải Anh có PR đầu tiên.

## 1. Data (owner: Sơn — automated qua pytest)
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
sạch, xác nhận `outputs/eda/split_audit.json` không có cảnh báo nghiêm trọng.

## 2. AI Baseline (owner: Thành — automated qua pytest)
Test hiện có: `tests/test_baselines.py`

| Hạng mục | Cách kiểm tra | Kỳ vọng |
|---|---|---|
| Most Popular | Unit test trên tập nhỏ | Xếp hạng đúng theo tần suất rating |
| MF forward | Shape output, kiểm tra NaN | Output đúng shape `(batch,)`, không NaN |
| MF train smoke | 1-2 epoch trên tập nhỏ | Loss giảm hoặc không tăng bất thường |
| GMF forward/gradient | `nn.Module` forward + backward | Gradient tồn tại, không NaN/Inf |

## 3. Backend API (owner: Chúc — automated qua pytest + manual qua Swagger)
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

## 4. Frontend (owner: Dương — chủ yếu manual, build kiểm tra qua CI)
Chưa có test tự động ở tuần 1 (đúng theo kế hoạch — "chưa cần giao diện đẹp"). CI chỉ
xác nhận `npm run build` không lỗi. Manual test xem Mục 2 trong
`MANUAL_TEST_CASES.md`.

## 5. Integration (owner: Hoàng Anh)
- Data → AI: `scripts/train_baseline.py` chạy được trên dữ liệu do `prepare_data.py`
  sinh ra, không lỗi shape/mapping.
- Frontend → Backend → Mock Top 10: đăng ký/đăng nhập → rating → gọi
  `/recommend/mock` → hiển thị kết quả trên UI, không lỗi console.

## Cách chạy toàn bộ automated test
```
python -m pytest -v
cd frontend && npm run build
```

## Việc cần cập nhật khi có PR mới
- Khi Công Thành có `tests/test_ncf.py` → thêm mục "NCF" vào plan này.
- Khi Hải Anh có `tests/test_group.py`, `tests/test_ranking.py` → thêm mục
  "Group Recommendation & Evaluation".
