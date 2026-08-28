# CineMatch Recommendation Contract v1

Tài liệu này chốt khuôn dữ liệu Group Recommendation dùng chung giữa
Sơn (Group Recommendation), Chúc (Backend) và Dương (Frontend).

## Quy ước ID

- `movie_id` là ID gốc MovieLens, dùng khi giao tiếp với Data và AI.
- `database_id` là khóa nội bộ của Backend và không được gửi thay cho
  `movie_id`.
- `movie_index` chỉ dùng bên trong embedding của mô hình AI.

## Endpoint mock tuần 1

```http
POST /recommend/mock
```

Đây là endpoint chỉ phục vụ phát triển và demo tuần 1. Nó không gọi mô
hình AI, không yêu cầu dữ liệu phòng thật và không ghi vào database.
Kết quả được tạo cố định để Frontend và test nhận cùng một response ở
mọi lần chạy.

Request:

```json
{
  "room_id": 1,
  "strategy": "average",
  "top_k": 10
}
```

Ba strategy hợp lệ:

```text
average
least_misery
average_without_misery
```

Response mẫu nằm trong
`docs/examples/recommendation_response.json`.

## Ý nghĩa các trường kết quả

| Trường | Ý nghĩa |
|---|---|
| `schema_version` | Phiên bản hợp đồng dữ liệu |
| `room_id` | Phòng được tính đề xuất |
| `strategy` | Chiến lược tổng hợp đang dùng |
| `recommendations` | Danh sách Top-K phim |
| `movie_id` | ID gốc MovieLens |
| `rank` | Thứ hạng, bắt đầu từ 1 |
| `group_score` | Điểm nhóm của riêng phim đó |
| `minimum_score` | Điểm thành viên thấp nhất |
| `disagreement` | Mức độ bất đồng giữa thành viên |
| `member_scores` | Điểm dự đoán của từng thành viên |
| `misery_warning` | Có người dưới misery threshold hay không |
| `explanations` | Các lý do phim được đề xuất |

`title`, `genres`, `poster_url` và `runtime_minutes` do Backend tra từ
movie catalog. Nếu MovieLens chưa có poster hoặc runtime thì trả `null`,
không tự tạo metadata sai.

## Quy tắc nghiệp vụ

- `group_score`, `minimum_score` và `disagreement` được tính riêng cho
  từng phim.
- Kết quả được sắp xếp theo `rank` tăng dần.
- Mỗi phim phải có điểm của các thành viên trong `member_scores`.
- Sơn chịu trách nhiệm chốt công thức, misery threshold và tie-break.
- Tuần 1 được dùng điểm giả, nhưng request/response phải giữ đúng schema.
- `average_without_misery` loại phim có `minimum_score < 2.5` trong dữ
  liệu mock.
- Bản nháp cũ dùng `top_movies` và `score` vẫn được Pydantic chấp nhận
  khi validate; output chuẩn mới dùng `recommendations` và `group_score`.
