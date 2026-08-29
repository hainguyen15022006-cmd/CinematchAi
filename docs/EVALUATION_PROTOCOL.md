# CineMatch Offline Evaluation Protocol

## 1. Mục tiêu

Tài liệu này định nghĩa quy trình đánh giá offline thống nhất cho
Most Popular, MF, GMF, NCF và Hybrid NCF.

Mọi mô hình phải sử dụng cùng:

- Temporal split.
- ID mapping.
- Positive threshold.
- Seen-item policy.
- Candidate set.
- Negative sampling.
- Top K.
- Random seed.
- Ranking metrics.

Không được thay đổi candidate set riêng cho từng mô hình vì điều đó
làm kết quả so sánh không công bằng.

## 2. Nguồn dữ liệu

Dữ liệu được lấy từ các artifact do Data pipeline tạo:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/movies.csv`
- `data/processed/id_mappings.json`

Ba file interaction sử dụng cùng schema:

| Cột | Ý nghĩa |
|---|---|
| `user_id` | ID user gốc của MovieLens |
| `movie_id` | ID phim gốc của MovieLens |
| `user_index` | Index liên tục dùng cho model |
| `movie_index` | Index liên tục dùng cho model |
| `rating` | Rating từ 1.0 đến 5.0 |
| `timestamp` | Thời gian rating |

Đánh giá offline và model sử dụng `user_index` và `movie_index`.
API và giao diện sử dụng `movie_id`. Việc chuyển đổi phải dùng
`id_mappings.json`.

## 3. Temporal split

MovieLens 100K được chia theo thời gian riêng cho từng user:

- 80% tương tác cũ nhất: train.
- 10% tiếp theo: validation.
- 10% mới nhất: test.

Quy mô hiện tại:

| Partition | Số dòng | Số user |
|---|---:|---:|
| Train | 80.014 | 943 |
| Validation | 10.132 | 943 |
| Test | 9.854 | 943 |

Chỉ train được dùng để cập nhật trọng số model.

Validation được dùng để chọn checkpoint hoặc hyperparameter.

Test chỉ được dùng để báo cáo kết quả cuối cùng. Không được sử
dụng test để huấn luyện, early stopping hoặc chọn cấu hình.

Audit hiện tại xác nhận:

- Không có interaction overlap giữa các partition.
- Không có train-validation temporal violation.
- Không có validation-test temporal violation.
- Không có user cold-start.

## 4. Seen items

Với mỗi user:

```text
seen_items = train_movie_indices ∪ validation_movie_indices