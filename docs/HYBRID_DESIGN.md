# Thiết kế Hybrid NCF tuần 1

## 1. Mục tiêu

Hybrid NCF giữ user/movie embeddings của NCF và bổ sung metadata để mô
hình không chỉ phụ thuộc vào ID. Thứ tự ghép feature là một contract cố
định giữa Data, AI training và Backend serving.

```text
[user embedding, movie embedding, genres, year, history, text]
```

## 2. Kích thước feature

| Thành phần | Số chiều | Nguồn |
|---|---:|---|
| User embedding | 32 | Học từ `user_index` |
| Movie embedding | 32 | Học từ `movie_index` |
| Movie genres | 19 | Các cột genre trong `movies.csv` |
| Release year | 1 | `release_year` được chuẩn hóa |
| User genre history | 19 | Hồ sơ thể loại tổng hợp từ rating cũ |
| Vietnamese preference text | 128 | `VietnameseTextEncoder` |
| **Side features** | **167** | 19 + 1 + 19 + 128 |
| **Tổng input MLP** | **231** | 32 + 32 + 167 |

Các hằng số và thứ tự concatenate nằm trong
`cinematch.features.hybrid_features`. Không tự đổi thứ tự ở Backend.

## 3. Ý nghĩa từng nhóm

### ID embeddings

`user_index` và `movie_index` đại diện tín hiệu collaborative. ID gốc
MovieLens vẫn được giữ bên ngoài model để join dữ liệu và trả response.

### Genres

Vector multi-hot 19 chiều lấy trực tiếp từ `movies.csv`. Một phim có thể
có nhiều giá trị 1.

### Release year

Chỉ là feature phụ, không dùng làm điều kiện xóa rating. Trước khi train,
year cần được scale bằng thống kê chỉ lấy từ train set.

### User history

Vector 19 chiều biểu diễn sở thích thể loại từ các rating trước thời điểm
dự đoán. Không được lấy rating tương lai từ validation/test vì sẽ gây data
leakage.

### Text

Câu onboarding tiếng Việt được chuyển thành vector 128 chiều. Baseline
tuần 1 dùng signed feature hashing để kiểm tra contract. Phương pháp này
không hiểu ngữ nghĩa sâu và có thể được thay bằng encoder pretrained ở
giai đoạn sau nếu nhóm còn thời gian.

## 4. Forward contract

```python
side_features = build_hybrid_side_features(
    genres,
    normalized_year,
    history_profile,
    text_vector,
)
predictions = model(user_indices, movie_indices, side_features)
```

`side_features` bắt buộc có shape `[batch_size, 167]`. Hybrid trả tensor
`[batch_size]` và mỗi prediction nằm trong khoảng 1–5.

## 5. Phạm vi tuần 1

Smoke training dùng metadata giả đúng shape và text vector thật để chứng
minh toàn bộ forward/backward chạy được. Nó chưa phải kết quả thí nghiệm
MovieLens. Việc join metadata thật và tính history không leakage thuộc
pipeline huấn luyện tiếp theo.

Chạy:

```bash
python scripts/train_hybrid_ncf.py
python -m pytest tests/test_hybrid_ncf.py tests/test_text_encoder.py -v
```
