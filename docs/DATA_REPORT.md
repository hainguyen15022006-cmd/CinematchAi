# CineMatch Data Report

## 1. Dataset overview

CineMatch sử dụng MovieLens 100K làm dữ liệu nền để
huấn luyện và đánh giá các mô hình recommendation.

Dữ liệu rating được đọc từ `u.data` và có bốn cột:
`user_id`, `movie_id`, `rating` và `timestamp`.

| Thuộc tính | Giá trị |
|---|---:|
| Ratings | 100.000 |
| Users | 943 |
| Movies | 1.682 |
| Minimum rating | 1 |
| Maximum rating | 5 |
| Mean rating | 3,5299 |

## 2. Data quality

Pipeline thực hiện kiểm tra schema, missing values,
rating range, user ID, movie ID và timestamp.

Kết quả:

- Missing values: 0.
- Rating ngoài khoảng 1-5: 0.
- Exact duplicate rows: 0.
- Duplicate user-movie rows: 0.
- User ID không hợp lệ: 0.
- Movie ID không hợp lệ: 0.
- Timestamp không hợp lệ: 0.

Dữ liệu rating đạt các điều kiện chất lượng cơ bản
để chuyển sang bước preprocessing.

## 3. Rating distribution

| Rating | Count | Percentage |
|---:|---:|---:|
| 1 | 6.110 | 6,11% |
| 2 | 11.370 | 11,37% |
| 3 | 27.145 | 27,15% |
| 4 | 34.174 | 34,17% |
| 5 | 21.201 | 21,20% |

Rating 4 xuất hiện nhiều nhất. Tổng số rating từ 4
trở lên là 55.375, tương đương khoảng 55,38%.

Dữ liệu có xu hướng nghiêng về các rating tích cực.
CineMatch sử dụng rating từ 4 trở lên làm positive
interaction trong ranking evaluation.

## 4. User interaction distribution

| Thống kê | Ratings per user |
|---|---:|
| Minimum | 20 |
| Maximum | 737 |
| Mean | 106,04 |
| Median | 65 |

Mean lớn hơn median cho thấy một số user có số lượng
rating rất lớn. Tuy nhiên, mỗi user có ít nhất 20 rating,
đủ để thực hiện per-user temporal split.

## 5. Movie interaction distribution

| Thống kê | Ratings per movie |
|---|---:|
| Minimum | 1 |
| Maximum | 583 |
| Mean | 59,45 |
| Median | 27 |

Phân bố rating theo phim có tính long-tail. Một số phim
nhận nhiều rating, trong khi nhiều phim chỉ có ít tương tác.
Điều này có thể gây popularity bias và làm model học kém
đối với các phim ít phổ biến.

## 6. Matrix sparsity

Số cặp user-phim có thể có:

```text
943 × 1.682 = 1.586.126