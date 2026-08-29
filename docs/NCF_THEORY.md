# Lý thuyết Neural Collaborative Filtering của CineMatch

## 1. Bài toán

NCF dự đoán rating của một người dùng cho một bộ phim. Đầu vào của mô
hình không phải `user_id` và `movie_id` gốc mà là `user_index` và
`movie_index` liên tục do Data pipeline tạo ra.

```text
user_index ──> user embedding ──┐
                                ├─> concatenate ─> MLP ─> rating 1–5
movie_index ─> movie embedding ─┘
```

Embedding biến một index rời rạc thành vector số thực có thể học được.
Trong cấu hình tuần 1, cả user embedding và movie embedding có 32 chiều.

## 2. NCF khác MF và GMF như thế nào?

- MF lấy tích vô hướng giữa hai embedding. Quan hệ user–movie được mô
  hình hóa bằng một phép tương tác tuyến tính cố định.
- GMF nhân từng phần tử của hai embedding rồi dùng một lớp tuyến tính để
  học trọng số cho các chiều ẩn.
- NCF nối hai embedding và đưa qua nhiều lớp fully connected. ReLU giúp
  MLP biểu diễn quan hệ phi tuyến phức tạp hơn.

NCF không mặc nhiên tốt hơn MF/GMF. Các mô hình phải được huấn luyện và
đánh giá trên cùng temporal split trước khi kết luận.

## 3. MLP, activation và dropout

Mỗi block hiện gồm:

```text
Linear -> LayerNorm -> ReLU -> Dropout
```

- `Linear` học cách kết hợp các đặc trưng đầu vào.
- `LayerNorm` giữ phân phối activation ổn định và vẫn dùng được khi batch
  cuối chỉ có một mẫu.
- `ReLU` tạo tính phi tuyến.
- `Dropout` ngẫu nhiên tắt một phần neuron khi train, làm mô hình khó phụ
  thuộc vào một vài neuron và giảm nguy cơ overfitting.

Khi gọi `model.eval()`, Dropout tự động được tắt.

## 4. Output rating

Lớp tuyến tính cuối sinh một logit không bị chặn. CineMatch chuyển logit
về rating MovieLens bằng:

```text
predicted_rating = 1 + 4 * sigmoid(logit)
```

Vì sigmoid nằm trong khoảng 0–1, prediction cuối luôn nằm trong khoảng
1–5. Test phải kiểm tra shape, giá trị hữu hạn, gradient và rating range.

## 5. Training smoke tuần 1

Training smoke chỉ xác nhận forward, loss và backpropagation hoạt động.
Dữ liệu ngẫu nhiên trong smoke test không được dùng để báo cáo chất lượng
mô hình. Tuần sau NCF phải dùng `train.csv`, chọn mô hình bằng
`validation.csv` và chỉ đánh giá cuối cùng trên `test.csv`.

Chạy:

```bash
python scripts/train_ncf.py
python -m pytest tests/test_ncf.py -v
```

## 6. Các khái niệm cần phân biệt

- Cold-start user đã có slot embedding nhưng chưa có đủ rating để học
  vector tốt.
- Unknown user chưa có index trong mapping và cần fallback trước khi gọi
  embedding.
- Overfitting xảy ra khi train loss tiếp tục giảm nhưng validation loss
  tăng.
- Smoke loss giảm chỉ chứng minh pipeline có thể tối ưu, không chứng minh
  recommendation tốt.
