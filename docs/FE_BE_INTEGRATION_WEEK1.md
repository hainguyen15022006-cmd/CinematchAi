# Kiểm thử tích hợp Frontend–Backend tuần 1

Tài liệu này ghi lại cách ghép và kiểm chứng hai phần:

- Backend nền: `codex/backend-recommendation-schema`.
- Frontend: `feature/frontend-duong`.
- Nhánh kiểm thử: `integration/fe-be-week1`.

Nhánh integration chỉ dùng để kiểm thử chung; không thay thế nhánh cá nhân
của Chúc hoặc Dương và không tự động thay đổi `main`.

## 1. Quan hệ giữa schema Backend và type Frontend

Backend là nguồn định nghĩa contract:

- Pydantic schemas: `app/schemas/`.
- API documentation: `docs/BACKEND_API.md`.
- Group Recommendation contract: `docs/RECOMMENDATION_CONTRACT.md`.
- OpenAPI khi server chạy: `http://127.0.0.1:8000/openapi.json`.

Frontend ánh xạ contract sang TypeScript tại:

```text
frontend/src/types/index.ts
```

Frontend gọi endpoint qua:

```text
frontend/src/services/api.ts
```

Quy ước quan trọng: `movie_id` trong request/response là ID gốc
MovieLens. `movies.id` là khóa nội bộ database và không được Frontend sử
dụng thay cho MovieLens ID.

## 2. Cách tạo nhánh tích hợp

```bash
git fetch origin
git switch codex/backend-recommendation-schema
git switch -c integration/fe-be-week1
git merge --no-ff origin/feature/frontend-duong \
  -m "merge: integrate frontend and backend week 1"
```

Hai nhánh đã merge tự động, không có conflict.

## 3. Chạy Backend

Từ thư mục gốc repository:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/seed_movies.py
uvicorn app.main:app --reload
```

Backend chạy tại `http://127.0.0.1:8000`; Swagger nằm tại
`http://127.0.0.1:8000/docs`.

## 4. Chạy Frontend

Mở Terminal khác:

```bash
cd frontend
npm ci
npm run dev
```

Frontend chạy tại `http://localhost:5173` và đọc Backend URL từ
`VITE_API_BASE_URL`. Giá trị mặc định là `http://127.0.0.1:8000`.

## 5. Luồng đã kiểm chứng

| Bước trên Frontend | Request thật tới Backend | Kết quả |
|---|---|---|
| Đăng nhập | `POST /auth/login` | `200`, nhận JWT |
| Mở onboarding | `GET /movies?limit=50&skip=0` | `200`, nhận 50 phim |
| Chấm Toy Story 4 sao | `POST /ratings` | `201`, rating được lưu |
| Tạo phòng | `POST /rooms` | `201`, nhận room code |
| Bấm Ready | `POST /rooms/{id}/ready` | `200`, trạng thái `READY` |
| Làm mới lobby | `GET /rooms/{code}` | `200`, nhận member list |
| Tạo Top 10 | `POST /recommend/mock` | `200`, nhận 10 phim |

Kết quả Top 10 đã hiển thị thành công:

- `group_score`, `minimum_score`, `disagreement`;
- `member_scores` của ba thành viên minh họa;
- `misery_warning`;
- danh sách `explanations`.

`POST /recommend/mock` vẫn dùng điểm dự đoán giả của tuần 1. Auth, movie
catalog, rating, room và ready đã chạy với Backend/database thật.

## 6. Lệnh kiểm tra trước khi merge

Backend:

```bash
python -m pytest -q
```

Frontend:

```bash
cd frontend
npm run build
```

Kết quả kiểm tra của nhánh integration:

```text
Backend: 70 passed
Frontend: TypeScript và Vite production build thành công
Seed: 1.682 movie rows
```

## 7. Tiêu chí hoàn thành tích hợp tuần 1

- Không dùng token hard-code ở Frontend.
- Tắt Backend thì Frontend phải hiển thị lỗi kết nối.
- Request cần đăng nhập phải có `Authorization: Bearer <JWT>`.
- Rating gửi bằng MovieLens ID và được lưu vào database.
- TypeScript types phải khớp Pydantic/OpenAPI.
- Top 10 hiển thị đủ fairness fields và explanation.
- Cả Backend test và Frontend build đều vượt qua trước Pull Request.
