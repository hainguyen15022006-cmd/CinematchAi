# CineMatch Data Dictionary

## 1. Raw ratings: `u.data`

| Cột | Kiểu dữ liệu | Ý nghĩa | Dùng để train |
|---|---|---|---|
| `user_id` | `int64` | ID người dùng của MovieLens | Sau khi ánh xạ sang `user_index` |
| `movie_id` | `int64` | ID phim của MovieLens | Sau khi ánh xạ sang `movie_index` |
| `rating` | `float32` | Điểm người dùng chấm, từ 1 đến 5 | Có |
| `timestamp` | `int64` | Thời điểm rating theo Unix time | Dùng để chia dữ liệu, không phải feature chính |

## 2. Raw movie catalog: `u.item`

| Cột | Kiểu dữ liệu | Ý nghĩa | Chính sách xử lý |
|---|---|---|---|
| `movie_id` | `int64` | ID phim | Giữ nguyên để join với rating |
| `title` | `string` | Tên phim, thường kèm năm | Dùng để hiển thị |
| `release_date` | `string` | Ngày phát hành trong catalog | Parse nhưng không dùng để xóa rating |
| `video_release_date` | `string` | Ngày phát hành video | Không dùng trong MVP |
| `imdb_url` | `string` | URL IMDb cũ | Metadata hiển thị, cho phép thiếu |
| 19 cột genre | `int8` | Vector multi-hot thể loại | Dùng cho Hybrid NCF |

Các cột genre nhận giá trị `0` hoặc `1`. Một phim có thể có
nhiều thể loại cùng lúc. `unknown` là một cột genre hợp lệ.

## 3. Processed movie catalog: `movies.csv`

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `movie_id` | `int64` | ID phim gốc |
| `title` | `string` | Tên phim |
| `release_date` | datetime/ISO date | Ngày phát hành đã parse; có thể thiếu |
| `release_year` | nullable integer | Năm phát hành; có thể thiếu |
| `release_date_missing` | `int8` | `1` nếu thiếu ngày phát hành, ngược lại `0` |
| `imdb_url` | `string` | URL hiển thị; có thể thiếu |
| 19 cột genre | `int8` | Vector multi-hot thể loại |

`release_date` và `release_year` là metadata phụ. Phiên bản
MVP không sử dụng chúng làm điều kiện lọc hoặc feature chính
của MF, GMF và NCF.

## 4. Processed interactions

Ba file `train.csv`, `validation.csv` và `test.csv` có cùng
schema:

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `user_id` | `int64` | ID người dùng gốc để tra cứu |
| `movie_id` | `int64` | ID phim gốc để tra cứu |
| `user_index` | `int64` | Index liên tục dùng cho user embedding |
| `movie_index` | `int64` | Index liên tục dùng cho movie embedding |
| `rating` | `float32` | Điểm rating từ 1 đến 5 |
| `timestamp` | `int64` | Thời gian rating và bằng chứng temporal split |

Các mô hình chỉ được fit bằng `train.csv`. `validation.csv`
dùng để chọn siêu tham số hoặc early stopping. `test.csv` chỉ
được dùng cho đánh giá cuối cùng.

## 5. ID mapping: `id_mappings.json`

File mapping có version và hai danh sách ID gốc:

| Trường | Ý nghĩa |
|---|---|
| `version` | Phiên bản schema mapping |
| `users.entity_name` | Loại thực thể `user` |
| `users.external_ids` | ID user theo thứ tự `user_index` |
| `movies.entity_name` | Loại thực thể `movie` |
| `movies.external_ids` | ID phim theo thứ tự `movie_index` |

Vị trí của ID trong `external_ids` chính là index embedding.
Backend phải dùng cùng file này để chuyển kết quả mô hình từ
`movie_index` về `movie_id`.

## 6. Split audit: `split_audit.json`

Báo cáo chứa số dòng, user, phim, rating distribution,
positive rate, cold-start item, overlap và temporal violation
của ba partition. Item cold-start được báo cáo nhưng không bị
xóa tự động.
