# Báo cáo kiểm tra máy sạch — Tuần 1

**Người thực hiện:** Hoàng Anh
**Môi trường chính thức của dự án:** Python 3.12, Node.js 20 (khớp version dùng trong
`ci.yml`).
**Môi trường máy cá nhân dùng để kiểm tra bổ sung:** Windows, PowerShell, VS Code,
Python 3.14.3 (venv riêng, chỉ để kiểm tra khả năng tương thích với bản Python mới hơn
— không dùng làm chuẩn báo cáo), Node.js v24.19.0.

## Mục tiêu
Clone repo vào máy hoàn toàn mới, làm đúng từng bước theo README, ghi lại mọi bước bị
thiếu, gây nhầm lẫn hoặc lỗi thật — không dùng cache/setup cũ nào.

## Kết quả tổng quan
| Hạng mục | Trạng thái |
|---|---|
| Clone + cài Python deps | FAIL lần đầu (bug `httpx2`) → **đã sửa (RESOLVED)** trong PR này |
| Chạy pytest | PASS (sau khi tải và xử lý MovieLens 100K; 4 test phụ thuộc data artifact skip trên CI, không skip khi chạy thủ công có data) |
| Chạy backend (uvicorn) | PASS |
| Chạy frontend (npm) | PASS (sau khi cài thêm Node.js — máy sạch chưa có sẵn) |
| Luồng end-to-end (đăng ký → rating → mock Top 10) | PASS |

## Bug đã xử lý (RESOLVED trong PR này)

### 1. `httpx2` thay vì `httpx` trong `pyproject.toml`
`pip install -e ".[dev]"` khai `httpx2>=2,<3` — sai cả tên package lẫn version
constraint (package `httpx` thật trên PyPI chỉ có bản `0.x`, không có bản `2.x`).
`fastapi.testclient.TestClient` cần `httpx` thật để hoạt động, nên
`tests/test_api_basic.py` lỗi `ModuleNotFoundError` trên máy sạch.

**Fix đã áp dụng:** đổi thành `"httpx>=0.27,<1"` trong `pyproject.toml`. Đã xác nhận
`pip install -e ".[dev]"` và `python -m pytest -v` chạy PASS toàn bộ sau khi sửa.

## Việc còn lại (chưa xử lý trong PR này — đề xuất cho tuần sau)

### 2. README thiếu hướng dẫn riêng cho PowerShell
README hiện dùng cú pháp bash (`&&`, `source .venv/bin/activate`), không chạy được
trên PowerShell mặc định của Windows (lỗi `The token '&&' is not a valid statement
separator`). Người dùng Windows cần dùng:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```
Nếu bị chặn chạy script: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
**Đề xuất:** thêm mục "Windows (PowerShell)" song song với hướng dẫn bash trong README.

### 3. Node.js không có sẵn trên máy sạch
Máy sạch (chưa từng dev JS trước đó) không có Node.js/npm cài sẵn — cần cài riêng
(`winget install OpenJS.NodeJS.LTS`, khuyến nghị bản **Node 20** để khớp `ci.yml`)
trước khi `npm install` chạy được.
**Đề xuất:** ghi rõ yêu cầu Node.js 20 (kèm version tối thiểu) là pre-requisite trong
README, giống như đã ghi yêu cầu Python 3.12.

### 4. MovieLens 100K không tự tải, cần chạy script riêng
4 test (`test_candidates.py`, `test_data_pipeline.py`, `test_mapping.py`,
`test_splitting.py`) tự SKIP nếu thiếu raw hoặc processed data artifact — đây là thiết
kế đúng, không phải bug. CI hiện tại không tải và không xử lý MovieLens nên các test
này luôn skip trên CI (xem `TEST_PLAN.md`). README nên nhắc rõ hơn là phải chạy cả
`python scripts/download_data.py` và `python scripts/prepare_data.py` trước khi chạy
full test suite thủ công; nếu không, người mới dễ hiểu nhầm là test đã pass hết dù
thực ra có test bị skip.

## Kết luận
Luồng end-to-end (Data → Backend → Frontend → Mock Top 10) chạy được trên máy sạch.
Bug blocker duy nhất (`httpx2`) đã được sửa trong chính PR này. 3 việc còn lại (mục
2–4) không phải blocker, chỉ là cải thiện tài liệu — đề xuất xử lý ở PR sau, không cần
chặn merge PR hiện tại.
