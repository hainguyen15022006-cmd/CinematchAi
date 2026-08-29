# Group Recommendation Theory

## 1. Mục tiêu

Module Group Recommendation nhận điểm dự đoán cá nhân của
từng thành viên cho từng phim, sau đó tổng hợp thành một điểm
chung để xếp hạng phim cho cả nhóm.

Module này không huấn luyện mô hình Deep Learning. Điểm cá
nhân được cung cấp bởi Most Popular, MF, GMF, NCF hoặc Hybrid
NCF.

Trong tuần 1, điểm dự đoán sử dụng thang từ 1 đến 5.

## 2. Dữ liệu đầu vào minh họa

Giả sử nhóm có bốn thành viên và ba phim:

| Movie | Member 1 | Member 2 | Member 3 | Member 4 |
|---|---:|---:|---:|---:|
| A | 5.0 | 5.0 | 5.0 | 1.5 |
| B | 4.0 | 4.0 | 4.0 | 4.0 |
| C | 4.5 | 3.5 | 4.0 | 3.0 |

Mỗi hàng là một phim. Các cột Member là điểm mà mô hình dự
đoán cho từng thành viên.

## 3. Average

Average tính trung bình cộng điểm của tất cả thành viên:

GroupScore(i) = sum(score(u, i)) / number_of_members

### Tính tay

Movie A:

Average(A) = (5.0 + 5.0 + 5.0 + 1.5) / 4
           = 16.5 / 4
           = 4.125

Movie B:

Average(B) = (4.0 + 4.0 + 4.0 + 4.0) / 4
           = 4.0

Movie C:

Average(C) = (4.5 + 3.5 + 4.0 + 3.0) / 4
           = 3.75

Thứ tự theo Average:

1. Movie A: 4.125
2. Movie B: 4.0
3. Movie C: 3.75

### Nhận xét

Average tối ưu mức hài lòng trung bình. Tuy nhiên, Movie A
được xếp đầu dù Member 4 chỉ có điểm 1.5. Như vậy, điểm cao
của đa số có thể che khuất sự phản đối mạnh của một thành viên.

## 4. Least Misery

Least Misery dùng điểm thấp nhất của các thành viên:

GroupScore(i) = min(score(u, i))

### Tính tay

LeastMisery(A) = min(5.0, 5.0, 5.0, 1.5) = 1.5

LeastMisery(B) = min(4.0, 4.0, 4.0, 4.0) = 4.0

LeastMisery(C) = min(4.5, 3.5, 4.0, 3.0) = 3.0

Thứ tự theo Least Misery:

1. Movie B: 4.0
2. Movie C: 3.0
3. Movie A: 1.5

### Nhận xét

Least Misery bảo vệ thành viên ít hài lòng nhất. Một thành
viên có thể gần như phủ quyết một phim bằng cách có điểm rất
thấp. Nhược điểm là chiến lược này có thể quá thận trọng và
không tận dụng được mức hài lòng cao của đa số.

## 5. Average Without Misery

Average Without Misery thực hiện hai giai đoạn:

1. Loại phim có minimum score nhỏ hơn misery threshold.
2. Tính Average cho những phim còn lại.

Trong ví dụ này:

misery_threshold = 2.0

Quy tắc:

- minimum_score < 2.0: phim bị loại.
- minimum_score >= 2.0: phim được giữ lại.

### Tính tay

Movie A:

minimum_score = 1.5

Vì 1.5 < 2.0 nên Movie A bị loại.

Movie B:

minimum_score = 4.0
average_score = 4.0

Movie B được giữ lại với GroupScore bằng 4.0.

Movie C:

minimum_score = 3.0
average_score = 3.75

Movie C được giữ lại với GroupScore bằng 3.75.

Thứ tự theo Average Without Misery:

1. Movie B: 4.0
2. Movie C: 3.75

Movie A không xuất hiện vì vi phạm misery threshold.

### Nhận xét

Average Without Misery cân bằng giữa Average và Least
Misery. Minimum score được dùng như điều kiện loại phim,
nhưng những phim vượt qua điều kiện vẫn được xếp hạng bằng
Average.

## 6. Minimum score

Minimum score là điểm dự đoán thấp nhất trong nhóm:

MinimumScore(i) = min(score(u, i))

Kết quả:

| Movie | Minimum score |
|---|---:|
| A | 1.5 |
| B | 4.0 |
| C | 3.0 |

Minimum score giúp giải thích mức hài lòng của thành viên ít
hài lòng nhất.

## 7. Disagreement

Disagreement đo mức phân tán điểm giữa các thành viên. Trong
tuần 1, module sử dụng population standard deviation:

Disagreement(i) =
sqrt(sum((score(u, i) - average(i))^2) / number_of_members)

Giá trị nhỏ thể hiện mức đồng thuận cao. Giá trị lớn thể hiện
các thành viên có ý kiến khác nhau.

### Movie A

Average(A) = 4.125

Squared differences:

- (5.0 - 4.125)^2 = 0.765625
- (5.0 - 4.125)^2 = 0.765625
- (5.0 - 4.125)^2 = 0.765625
- (1.5 - 4.125)^2 = 6.890625

Sum = 9.1875

Variance = 9.1875 / 4 = 2.296875

Disagreement(A) = sqrt(2.296875) = 1.5155

### Movie B

Tất cả thành viên đều có điểm 4.0 nên:

Disagreement(B) = 0.0

### Movie C

Average(C) = 3.75

Squared differences:

- (4.5 - 3.75)^2 = 0.5625
- (3.5 - 3.75)^2 = 0.0625
- (4.0 - 3.75)^2 = 0.0625
- (3.0 - 3.75)^2 = 0.5625

Sum = 1.25

Variance = 1.25 / 4 = 0.3125

Disagreement(C) = sqrt(0.3125) = 0.5590

### Kết quả

| Movie | Average | Minimum | Disagreement |
|---|---:|---:|---:|
| A | 4.125 | 1.5 | 1.5155 |
| B | 4.000 | 4.0 | 0.0000 |
| C | 3.750 | 3.0 | 0.5590 |

Movie B có sự đồng thuận cao nhất. Movie A có mức bất đồng
cao nhất.

## 8. Phân biệt hai threshold

Positive rating threshold và misery threshold không có cùng
mục đích.

### Positive rating threshold

positive_rating_threshold = 4.0

Được dùng trong evaluation. Một phim user chấm từ 4 trở lên
được xem là relevant.

### Misery threshold

misery_threshold = 2.0

Được dùng trong Average Without Misery. Phim bị loại nếu có
thành viên nhận predicted score dưới 2.0.

Không được dùng hai threshold thay thế cho nhau.

## 9. Quy tắc tie-break dự kiến

Khi hai phim có cùng GroupScore, hệ thống dự kiến ưu tiên:

1. Minimum score cao hơn.
2. Disagreement thấp hơn.
3. Movie ID nhỏ hơn để kết quả có thể tái lập.

Quy tắc này sẽ được kiểm thử trước khi tích hợp Backend.

## 10. So sánh các chiến lược

| Strategy | Cách tính | Ưu điểm | Hạn chế |
|---|---|---|---|
| Average | Trung bình điểm | Tối ưu hài lòng trung bình | Có thể bỏ qua người phản đối |
| Least Misery | Điểm thấp nhất | Bảo vệ người ít hài lòng nhất | Có thể quá thận trọng |
| Average Without Misery | Lọc theo minimum, sau đó lấy average | Cân bằng hai mục tiêu | Phụ thuộc misery threshold |

## 11. Nguồn tham khảo

- Stratigi et al., "Sequential group recommendations based on
  satisfaction and disagreement scores", Journal of Intelligent
  Information Systems.
  https://link.springer.com/article/10.1007/s10844-021-00652-x