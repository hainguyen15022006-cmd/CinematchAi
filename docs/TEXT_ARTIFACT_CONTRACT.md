# Text encoder artifact contract v1

## 1. Mục đích

Artifact giúp training và serving tái tạo đúng cùng một text encoder.
Baseline tuần 1 là deterministic feature hashing nên không có trọng số
học được; artifact chỉ cần lưu cấu hình.

## 2. JSON schema

Ví dụ `artifacts/text_encoder.json`:

```json
{
  "schema_version": "1.0",
  "encoder_type": "signed_feature_hashing",
  "dimension": 128,
  "lowercase": true,
  "ngram_range": [1, 2]
}
```

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `schema_version` | string | Phiên bản contract, hiện là `1.0` |
| `encoder_type` | string | Thuật toán encoder |
| `dimension` | integer | Số chiều của mỗi text vector |
| `lowercase` | boolean | Có chuyển văn bản thành chữ thường hay không |
| `ngram_range` | array[integer] | Kích thước unigram/bigram được hash |

Loader phải từ chối artifact thiếu trường, sai version hoặc sai encoder
type. Artifact và checkpoint được sinh lại nên không commit vào Git.

## 3. Vector contract

- Input: một chuỗi tiếng Việt không rỗng.
- Output: `torch.float32`, shape `[128]`.
- Batch output: shape `[batch_size, 128]`.
- Tất cả phần tử hữu hạn.
- Vector được L2-normalize.
- Cùng text và cùng artifact phải tạo cùng vector.

Text rỗng bị từ chối bằng `ValueError` để Backend yêu cầu người dùng nhập
lại hoặc sử dụng fallback được thống nhất riêng.

## 4. Demo

```bash
python experiments/text_encoder_demo.py
```

Demo encode ba câu sở thích tiếng Việt và in shape, dtype, trạng thái
finite cùng norm. Test save/load contract nằm trong
`tests/test_text_encoder.py`.

## 5. Giới hạn

Feature hashing nhận biết token và cụm từ giống nhau nhưng không hiểu hai
câu đồng nghĩa. Nếu thay bằng PhoBERT hoặc Sentence Transformer, nhóm phải
tăng `schema_version`, khai báo model name/revision và giữ nguyên nguyên
tắc dimension cố định, deterministic inference và artifact validation.
