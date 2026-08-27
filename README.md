# CineMatch AI

CineMatch là hệ thống gợi ý phim cho nhóm, kết hợp mô hình
Deep Learning với các chiến lược Group Recommendation.

Repository đang được phát triển theo phạm vi MVP bốn tuần.
Phần Data cung cấp một pipeline có thể tái lập trên
MovieLens 100K cho MF, GMF, NCF và Hybrid NCF.

## Thành viên

- Hải Anh: Data.
- Thành: AI Baseline, MF, GMF.
- Công Thành: NCF, Hybrid NCF, Text.
- Chúc: Backend.
- Dương: Frontend.
- Hoàng Anh: Testing, Integration, DevOps.
- Thành viên còn lại: Group Recommendation và Evaluation.

## Yêu cầu môi trường

- Python 3.12.
- Git.
- Kết nối mạng trong lần tải MovieLens đầu tiên.

Trên macOS hoặc Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` là nguồn dependency chính. File
`requirements.txt` chỉ gọi lại nhóm dependency phát triển để
tránh khai báo phiên bản ở hai nơi khác nhau.

## Chạy Data pipeline

Tải MovieLens 100K:

```bash
python scripts/download_data.py
```

Khảo sát và kiểm tra dữ liệu raw:

```bash
python scripts/inspect_data.py \
  --output outputs/eda/ratings_profile.json
python scripts/inspect_movies.py \
  --output outputs/eda/movies_profile.json
python scripts/audit_movie_metadata.py
```

Tạo dữ liệu model-ready và chạy post-split audit:

```bash
python scripts/prepare_data.py
python scripts/audit_splits.py
```

Các đường dẫn, tỷ lệ split, số lượng kỳ vọng và positive
threshold được quản lý tập trung trong
`configs/cinematch.yaml`.

## Kết quả được tạo

```text
data/processed/
├── train.csv
├── validation.csv
├── test.csv
├── movies.csv
└── id_mappings.json
```

```text
outputs/eda/
├── ratings_profile.json
├── movies_profile.json
└── split_audit.json
```

Các file trên là dữ liệu dẫn xuất và không được commit lên
GitHub. Thành viên trong nhóm tái tạo chúng bằng các script.

## Data contract cho mô hình

Ba partition có cùng schema:

```text
user_id, movie_id, user_index, movie_index, rating, timestamp
```

- MF, GMF và NCF dùng `user_index` và `movie_index`.
- `user_id` và `movie_id` được giữ để kết nối Backend.
- `rating` là target cho explicit-feedback training.
- `timestamp` dùng để chứng minh temporal split, không phải
  feature embedding mặc định.
- Hybrid NCF có thể join 19 cột genre từ `movies.csv` bằng
  `movie_id`.

Chỉ `train.csv` được dùng để fit tham số mô hình.
`validation.csv` dùng để chọn hyperparameter hoặc early
stopping. `test.csv` chỉ dùng cho đánh giá cuối cùng.

## Evaluation protocol

- Per-user temporal split gần 80/10/10.
- Rating từ 4 trở lên được coi là positive interaction.
- Candidate construction và negative sampling phải giống
  nhau giữa các mô hình được so sánh.
- Không xóa item cold-start khỏi dữ liệu gốc.
- Báo cáo metric trên full test và có thể báo cáo thêm
  warm-start metric.

Chi tiết nằm trong `docs/DATA_REPORT.md` và
`docs/DATA_DICTIONARY.md`.

## Kiểm thử

Chạy toàn bộ test:

```bash
python -m pytest -v
```

## Chạy Backend API

Sau khi tạo môi trường và Data pipeline, tạo file cấu hình local rồi seed
movie catalog:

```bash
cp .env.example .env
python scripts/seed_movies.py
uvicorn app.main:app --reload
```

Kiểm tra API tại:

- Health check: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Backend dùng FastAPI, SQLAlchemy và SQLite cho MVP. Authentication dùng
JWT; mật khẩu được hash bằng Argon2. `movie_id` trong API luôn là ID gốc
MovieLens, không phải khóa nội bộ `movies.id`.

Frontend có thể tích hợp sớm bằng `POST /recommend/mock`. Endpoint này trả
Top-K xác định theo contract v1 mà chưa cần model hoặc database. Chi tiết
endpoint, luồng room/vote và quy ước ID nằm trong
`docs/BACKEND_API.md`; schema Group Recommendation nằm trong
`docs/RECOMMENDATION_CONTRACT.md`.

## Chạy AI baseline

Sau khi chạy Data pipeline, huấn luyện Most Popular, MF và GMF trên
cùng temporal split:

```bash
python scripts/train_baseline.py
python scripts/evaluate_baselines.py
```

Hyperparameter nằm trong `baselines` của `configs/cinematch.yaml`.
Checkpoint và bảng MSE/RMSE/MAE được ghi vào `outputs/baselines/` và
không commit lên GitHub. Lý thuyết và evaluation contract được mô tả
trong `docs/BASELINE_THEORY.md`.

Kiểm tra định dạng thay đổi trước khi commit:

```bash
git diff --check
git status --short
```

## Chính sách Git cho dữ liệu

Không commit:

- `data/raw/**`
- `data/processed/**`
- `outputs/**`
- virtual environment, cache hoặc model checkpoint.

Các file `.gitkeep` được giữ để duy trì cấu trúc thư mục.
