# CineMatch — Kết quả AI Baseline tuần 1

## 1. Mục tiêu thí nghiệm

Thí nghiệm tuần 1 xây dựng các mô hình tham chiếu để đánh giá những mô
hình sâu hơn ở các giai đoạn sau. Ba mô hình được so sánh gồm:

- **Most Popular:** baseline không cá nhân hóa, xếp hạng phim bằng điểm
  trung bình đã làm trơn theo số lượng rating.
- **Matrix Factorization (MF):** học embedding user, embedding movie,
  user bias và movie bias để dự đoán rating.
- **Generalized Matrix Factorization (GMF):** nhân từng phần tử giữa hai
  embedding rồi dùng một lớp tuyến tính học trọng số của từng chiều ẩn.

Kết quả trong tài liệu này là mốc tham chiếu. NCF và Hybrid NCF chỉ được
coi là cải thiện khi được huấn luyện và đánh giá trên cùng dữ liệu, cùng
candidate protocol và cùng metrics.

## 2. Dữ liệu và giao thức đánh giá

Nhóm sử dụng MovieLens 100K với:

| Thuộc tính | Giá trị |
|---|---:|
| Số rating | 100.000 |
| Số user | 943 |
| Số phim | 1.682 |
| Miền rating | 1–5 |
| Độ thưa của ma trận user–movie | 93,6953% |

Dữ liệu được chia theo thời gian riêng cho từng user, không random split:

| Partition | Số rating | Vai trò |
|---|---:|---|
| Train | 80.014 | Cập nhật tham số mô hình |
| Validation | 10.132 | Chọn epoch và checkpoint tốt nhất |
| Test | 9.854 | Báo cáo kết quả cuối cùng |

Các quy tắc bắt buộc:

1. Chỉ `train.csv` được dùng để cập nhật trọng số.
2. Early stopping và lựa chọn checkpoint dựa trên validation RMSE.
3. `test.csv` không được dùng để điều chỉnh hyperparameter.
4. Cả ba baseline sử dụng cùng `user_index`, `movie_index` và temporal
   split do Data pipeline tạo ra.
5. Prediction chỉ được clamp về khoảng 1–5 khi tính metrics, không clamp
   trong forward pass khi huấn luyện.

## 3. Cấu hình chạy

Cấu hình được đọc từ `configs/cinematch.yaml`:

| Hyperparameter | Giá trị |
|---|---:|
| Random seed | 42 |
| Embedding dimension | 32 |
| Batch size | 1.024 |
| Epoch tối đa | 20 |
| Learning rate | 0,001 |
| Weight decay | 0,00001 |
| Early-stopping patience | 3 epoch |
| Optimizer | Adam |
| Training loss | MSE |
| Most Popular prior count | 20 |

Thí nghiệm kiểm chứng được chạy bằng Python 3.12 trên CPU. Phiên bản cụ
thể của Python, PyTorch, pandas và NumPy nên được ghi lại cho mỗi lần
chạy chính thức vì chúng có thể khác giữa các máy.

Có thể in môi trường bằng lệnh:

```bash
python --version
python -c "import torch, pandas, numpy; print('torch=', torch.__version__); print('pandas=', pandas.__version__); print('numpy=', numpy.__version__)"
```

## 4. Kết quả kiểm chứng

| Mô hình | Validation RMSE | Validation MAE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Most Popular | 1,0780 | 0,8656 | 1,1115 | 0,8989 |
| MF | **0,9781** | 0,7694 | 1,0285 | **0,8148** |
| GMF | 0,9799 | **0,7692** | **1,0268** | 0,8149 |

Kết quả được làm tròn đến bốn chữ số thập phân. Khi chạy trên phiên bản
PyTorch hoặc phần cứng khác, kết quả có thể dao động nhẹ dù cùng seed.

Checkpoint tốt nhất trong lần kiểm chứng:

| Mô hình | Epoch tốt nhất |
|---|---:|
| MF | 16 |
| GMF | 11 |

## 5. Nhận xét kết quả

### 5.1 Most Popular là mốc thấp hợp lý

Most Popular đạt test RMSE 1,1115. Mô hình này không sử dụng danh tính
user nên mọi user nhận cùng điểm dự đoán cho một phim. Nó vẫn cần thiết
vì chứng minh rằng mô hình cá nhân hóa phải tốt hơn một chiến lược đơn
giản chỉ dựa trên độ phổ biến.

### 5.2 MF và GMF đều cải thiện so với Most Popular

MF giảm test RMSE từ 1,1115 xuống 1,0285. GMF giảm xuống 1,0268. Điều
này cho thấy embedding user và movie đã học được tín hiệu sở thích cá
nhân mà Most Popular không thể biểu diễn.

Tính theo test RMSE, GMF cải thiện khoảng 7,6% so với Most Popular:

```text
(1,1115 - 1,0268) / 1,1115 × 100% ≈ 7,6%
```

### 5.3 Chưa thể kết luận GMF vượt trội MF

Test RMSE của GMF chỉ thấp hơn MF khoảng 0,0017. Chênh lệch này rất nhỏ
và có thể thay đổi giữa các lần chạy. Báo cáo đúng nên kết luận rằng MF
và GMF có hiệu năng gần tương đương trong cấu hình tuần 1, không tuyên
bố GMF chắc chắn tốt hơn về mặt thống kê.

### 5.4 Cách dùng GMF trong CineMatch

GMF gốc trong Neural Collaborative Filtering thường được trình bày cho
implicit feedback. CineMatch tuần 1 đang giải bài toán dự đoán explicit
rating 1–5, vì vậy mô hình dùng đầu ra tuyến tính và MSE loss. Đây là một
biến thể GMF dành cho regression và phải được nói rõ khi trình bày.

## 6. Metrics có ý nghĩa gì?

- **MSE:** trung bình bình phương sai số, đồng thời là loss dùng để train.
- **RMSE:** căn bậc hai của MSE, phạt mạnh các dự đoán sai nhiều và có
  cùng đơn vị với rating.
- **MAE:** trung bình trị tuyệt đối sai số. Test MAE khoảng 0,815 nghĩa là
  dự đoán lệch trung bình khoảng 0,815 điểm rating.

RMSE thấp hơn là tốt hơn. Tuy nhiên RMSE/MAE chỉ đo rating prediction,
chưa trực tiếp đo chất lượng danh sách Top-K.

## 7. Artifact được tạo sau khi train

Lệnh:

```bash
python scripts/train_baseline.py --device cpu
```

tạo:

```text
outputs/baselines/
├── mf_checkpoint.pt
├── gmf_checkpoint.pt
└── metrics.json
```

- Checkpoint chứa model config, state dictionary, best epoch, seed và
  format version.
- `metrics.json` chứa validation metrics, test metrics và lịch sử train.
- `outputs/` và checkpoint không commit lên GitHub; tài liệu kết quả này
  được commit để lưu lại bằng chứng thí nghiệm.

Để in bảng kết quả:

```bash
python scripts/evaluate_baselines.py
```

## 8. Kiểm thử

Chạy riêng test baseline:

```bash
python -m pytest tests/test_baselines.py -v
```

Kết quả kiểm chứng của tuần 1:

```text
10 passed
```

Test bao phủ Most Popular, loại phim đã xem, fallback cho phim lạ, forward
và backward của MF/GMF, kiểm tra shape đầu vào, regression metrics và một
smoke test chứng minh MF có thể học.

Trước khi merge phải chạy thêm:

```bash
python -m pytest -v
git diff --check
git status --short
```

## 9. Hạn chế còn lại và công việc tuần sau

1. Bổ sung warm-start metrics vì validation và test còn một số phim chưa
   xuất hiện trong train.
2. Không dùng test metrics để điều chỉnh learning rate, embedding size
   hoặc epoch.
3. Xây dựng candidate set và negative-sampling protocol dùng chung.
4. Bổ sung Hit Rate@10, Recall@10 và NDCG@10 cho Top-K recommendation.
5. So sánh baseline với NCF và Hybrid NCF bằng cùng split và protocol.
6. Bàn giao predicted member scores cho Group Recommendation để thử
   Average, Least Misery và Average Without Misery.

## 10. Kết luận tuần 1

Most Popular, MF và GMF đều đã được cài đặt, kiểm thử và chạy thành công
trên temporal split chung. MF và GMF cải thiện rõ rệt so với baseline
không cá nhân hóa, chứng minh embedding có học được tín hiệu user–movie.

Kết quả này đủ làm mốc baseline tuần 1. Nó chưa phải kết quả cuối của
CineMatch vì hệ thống vẫn cần NCF/Hybrid NCF, Top-K evaluation, xử lý
cold-start và các chiến lược tổng hợp điểm cho nhóm.
