# CineMatch Frontend — Dương (Tuần 1)

Frontend React + TypeScript + Vite cho CineMatch, bám theo Backend API/Recommendation Contract hiện có.

## Phạm vi hoàn thành

- React + TypeScript + Vite, chạy mặc định tại `http://localhost:5173`.
- Register/Login bằng API thật; JWT được lưu ở `localStorage` và tự thêm vào header `Authorization`.
- Onboarding lấy phim từ `GET /movies`, chấm 1–5 sao và gửi `POST /ratings`.
- Trang Top 10 gọi `POST /recommend/mock` và hiển thị đủ `group_score`, `minimum_score`, `disagreement`, `member_scores`, `misery_warning`, `explanations`.
- Trang phòng nhóm dùng API thật để tạo/join room và toggle ready.
- Có loading state, error state, success feedback và route cần đăng nhập.

## Chạy frontend

```bash
cd frontend
npm install
npm run dev
```

Mở `http://localhost:5173`.

## Backend

Backend cần chạy ở `http://127.0.0.1:8000` (hoặc tạo `.env` từ `.env.example` và đổi `VITE_API_BASE_URL`).

Ví dụ chạy backend từ root repo:

```bash
uvicorn app.main:app --reload
```

Nếu database chưa có movie catalog, làm theo `docs/BACKEND_API.md` của Backend trước khi test `/movies` và `/ratings`.

## Luồng demo đề xuất

1. Mở `/register`, tạo tài khoản.
2. Sau khi đăng ký thành công, app tự login và chuyển tới `/onboarding`.
3. Chấm ít nhất 5 phim. Mỗi click sao gọi API rating thật.
4. Mở `/recommendations`, chọn một trong 3 strategy và bấm **Tạo Top 10**.
5. Kiểm tra các trường fairness/explanation của từng phim.
6. Có thể mở `/room` để tạo phòng, tham gia phòng hoặc đổi trạng thái ready.

## API contract frontend đang dùng

- `POST /auth/register`
- `POST /auth/login`
- `GET /users/me`
- `GET /movies`
- `POST /ratings`
- `POST /recommend/mock`
- `POST /rooms`
- `GET /rooms/{code}`
- `POST /rooms/{code}/join`
- `POST /rooms/{id}/ready`

`movie_id` khi gửi rating/recommendation là **MovieLens ID**, không phải khóa database nội bộ.

## Build kiểm tra trước Pull Request

```bash
npm run build
```

Trước khi commit:

```bash
git status
git add frontend
git commit -m "feat: implement frontend week 1 flow"
git push -u origin feature/frontend-duong
```
