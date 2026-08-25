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
```

Trong 1.586.126 tương tác có thể có, tập dữ liệu chỉ chứa
100.000 rating. Mật độ là 6,3047% và độ thưa là 93,6953%.
Đây là đặc trưng bình thường của dữ liệu recommendation và
là lý do các mô hình MF, GMF và NCF sử dụng embedding.

## 7. Movie metadata quality

Catalog `u.item` chứa 1.682 phim và tất cả movie ID xuất hiện
trong rating đều tồn tại trong catalog.

| Kiểm tra | Kết quả |
|---|---:|
| Thiếu tiêu đề | 0 |
| Thiếu ngày phát hành | 1 |
| Thiếu IMDb URL | 3 |
| Phim có thể loại `unknown` | 2 |
| Phim có nhiều hơn một thể loại | 849 |
| Số thể loại trung bình mỗi phim | 1,72 |
| Rating trước ngày phát hành ghi trong catalog | 231 |
| Phim có bất nhất thời gian | 24 |

IMDb URL chỉ là metadata hiển thị và không được sử dụng để
huấn luyện. Phim thiếu ngày phát hành hoặc có thể loại
`unknown` vẫn được giữ vì chúng có tương tác rating hợp lệ.

Ngày phát hành trong `u.item` không hoàn toàn nhất quán với
timestamp của rating. Vì vậy, pipeline không dùng ngày phát
hành làm điều kiện xóa rating hoặc lọc phim. Temporal split
chỉ dựa trên timestamp của rating.

Trong catalog processed, `release_date` được parse sang kiểu
ngày, `release_year` là số nguyên cho phép thiếu và
`release_date_missing` ghi rõ bản ghi nào thiếu ngày. Ngày
phát hành hiện chỉ là metadata phụ, không phải đầu vào chính
của MF, GMF hoặc NCF.

## 8. Data processing policy

- Không chỉnh sửa các file trong `data/raw`.
- Không xóa rating dựa trên ngày phát hành.
- Không tự điền ngày phát hành hoặc IMDb URL còn thiếu.
- Giữ đủ 19 cột thể loại, bao gồm `unknown`.
- Dùng timestamp của rating cho per-user temporal split.
- Lưu dữ liệu đã biến đổi trong `data/processed`.

Các quyết định trên giúp pipeline có thể tái hiện và tránh
đưa giả định không kiểm chứng vào dữ liệu gốc.

## 9. Per-user temporal split

Sau khi ánh xạ user và movie ID, rating của từng user được
sắp xếp tăng dần theo `timestamp`. `movie_id` được dùng làm
khóa phụ khi hai rating có cùng timestamp để kết quả luôn
có thể tái hiện.

Mỗi user được chia gần theo tỷ lệ:

- 80% tương tác cũ nhất cho train.
- 10% tiếp theo cho validation.
- 10% mới nhất cho test.

Vì số tương tác của từng user là số nguyên và không phải lúc
nào cũng chia hết theo tỷ lệ 80/10/10, pipeline dùng phương
pháp largest remainder để phân bổ phần dư. Kết quả trên toàn
bộ MovieLens 100K là:

| Partition | Rows | Percentage |
|---|---:|---:|
| Train | 80.014 | 80,014% |
| Validation | 10.132 | 10,132% |
| Test | 9.854 | 9,854% |
| Total | 100.000 | 100% |

Cả ba partition đều chứa đủ 943 user. Pipeline kiểm tra mỗi
tương tác xuất hiện đúng một lần và không bị thất lạc hoặc
trùng giữa các partition. Ngày phát hành phim không tham gia
vào quá trình chia dữ liệu.

## 10. Post-split audit and cold-start

Post-split audit confirms that the three partitions contain
100,000 interactions in total and that no interaction appears
in more than one partition.

| Check | Train | Validation | Test |
|---|---:|---:|---:|
| Rows | 80,014 | 10,132 | 9,854 |
| Users | 943 | 943 | 943 |
| Movies | 1,611 | 1,323 | 1,377 |
| Positive rate | 57.2750% | 48.1938% | 47.3310% |

Integrity results:

- User cold-start count: 0.
- Validation item cold-start: 33 movies and 36 interactions.
- Test item cold-start: 45 movies and 52 interactions.
- Cross-partition interaction overlap: 0.
- Train-validation temporal violations: 0.
- Validation-test temporal violations: 0.

Cold-start interactions are retained because they represent a
real limitation of collaborative filtering rather than corrupt
data. Evaluation must report metrics on the complete temporal
test set and may additionally report warm-start metrics using
only movies observed in the training partition.

All compared recommendation models must use the same evaluation
protocol and candidate construction rules.

## 11. Data handoff and known limitations

Nguồn cấu hình dùng chung là `configs/cinematch.yaml`. Pipeline
kiểm tra dataset phải có đúng 100.000 rating, 943 user và
1.682 phim trước khi tạo dữ liệu model-ready.

Phần Data bàn giao cho AI gồm:

- `train.csv`, `validation.csv` và `test.csv`.
- `movies.csv` với metadata và 19 cột thể loại.
- `id_mappings.json` để encode/decode ID ổn định.
- `split_audit.json` ghi kết quả kiểm tra sau split.
- Loader có schema rõ ràng trong `cinematch.data.io`.

MovieLens 100K không cung cấp poster hiện đại hoặc thời lượng
phim. Vì vậy, poster và ràng buộc thời lượng của giao diện phải
được bổ sung từ một nguồn metadata khác; không được tự bịa giá
trị trong data training. Văn bản sở thích tiếng Việt là dữ liệu
onboarding do hệ thống thu thập sau này, không có sẵn trong
MovieLens 100K.

Pipeline hiện chưa tạo negative samples. Việc negative sampling
phụ thuộc objective và evaluation protocol, nên phải được người
làm AI và Evaluation thống nhất, đồng thời dùng cùng candidate
set cho MF, GMF và NCF.

Với các giới hạn trên, dữ liệu hiện tại đủ để huấn luyện và so
sánh baseline, MF, GMF và NCF trên explicit rating; đồng thời
cung cấp genre features cho Hybrid NCF.
