# CineMatch Backend API (MVP)

Tài liệu này mô tả phần Backend tuần 1 để Chúc, Dương và các thành viên
tích hợp cùng một quy ước. Swagger tại `http://127.0.0.1:8000/docs` là
nguồn tham chiếu chính xác cho request/response đang chạy.

## Khởi động

```bash
cp .env.example .env
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/seed_movies.py
uvicorn app.main:app --reload
```

`scripts/seed_movies.py` đọc `data/processed/movies.csv`. Script có thể
chạy lại nhiều lần: phim mới được thêm, phim đã có được cập nhật theo
`movielens_id`.

MVP chưa dùng migration tool. Nếu đã tạo `cinematch.db` bằng schema Backend
cũ, hãy sao lưu rồi xóa database local đó trước khi seed lại. Việc này không
ảnh hưởng dữ liệu đã commit vì các file `.db` được Git bỏ qua.

## Xác thực

Đăng ký hoặc đăng nhập để nhận JWT, sau đó gửi header:

```http
Authorization: Bearer <access_token>
```

Endpoint `/health`, `/movies` và `/recommend/mock` không yêu cầu token.
Các endpoint profile, rating, room, run và vote yêu cầu token.

## Endpoint chính

| Method | Path | Chức năng |
|---|---|---|
| `GET` | `/health` | Kiểm tra API hoạt động |
| `POST` | `/auth/register` | Tạo tài khoản |
| `POST` | `/auth/login` | Nhận JWT |
| `GET` | `/users/me` | Lấy profile hiện tại |
| `PUT` | `/users/me/preferences` | Cập nhật mô tả sở thích |
| `GET` | `/movies` | Lấy danh mục phim |
| `POST` | `/ratings` | Tạo hoặc cập nhật rating |
| `POST` | `/rooms` | Tạo phòng |
| `GET` | `/rooms/{code}` | Lấy lobby của phòng |
| `POST` | `/rooms/{code}/join` | Tham gia phòng |
| `POST` | `/rooms/{id}/ready` | Đổi trạng thái ready |
| `PUT` | `/rooms/{id}/constraints` | Host cập nhật ràng buộc |
| `POST` | `/rooms/{id}/recommend` | Host tạo run mock trong DB |
| `GET` | `/runs/{id}/items` | Lấy shortlist của run |
| `POST` | `/runs/{id}/votes` | Tạo hoặc đổi lựa chọn của user |
| `POST` | `/runs/{id}/finalize` | Host chốt kết quả vote |
| `GET` | `/runs/{id}/result` | Thành viên lấy kết quả đã chốt |
| `POST` | `/recommend/mock` | Top-K giả theo contract v1 |

## Quy ước ID

- API nhận và trả `movie_id` theo ID gốc MovieLens.
- `movies.id` là khóa nội bộ database và không phải API contract.
- `movie_index` chỉ dành cho embedding của mô hình AI.
- Backend nối kết quả AI với metadata bằng `movie_id`/`movielens_id`.

## Recommendation mock và AI thật

`POST /recommend/mock` là endpoint không cần database, giúp Frontend làm
song song khi AI thật chưa hoàn thành. Response đầy đủ fairness fields,
được mô tả trong `docs/RECOMMENDATION_CONTRACT.md`.

`POST /rooms/{id}/recommend` hiện tạo shortlist xác định từ movie catalog
để kiểm thử luồng room/vote. Khi tích hợp AI, thay phần adapter trong
`app/services/ai_service.py`; không đổi response schema đã chốt.

## Trạng thái và quy tắc vote

- Phòng mới ở trạng thái `OPEN`; chỉ host được chạy recommendation.
- Mọi thành viên phải `ready` trước khi host tạo run.
- Run ở trạng thái `VOTING` mới nhận vote.
- Mỗi user chỉ có một lựa chọn trong một run; gửi lại sẽ đổi phim đã chọn.
- Chỉ phim thuộc shortlist của run mới được vote.
- Host gọi `finalize` để chốt; hòa phiếu được phá bằng thứ hạng ban đầu.
- `GET result` không làm thay đổi database và chỉ hoạt động sau khi chốt.

## Kiểm thử

```bash
python -m pytest -v
```

Test Backend dùng SQLite in-memory riêng và không xóa hay sửa
`cinematch.db` của lập trình viên.
