# Báo cáo kiểm tra máy sạch — Tuần 1

**Người thực hiện:** Hoàng Anh
**Môi trường:** Windows, PowerShell, VS Code, Python 3.14.3, Node.js v24.19.0

## Mục tiêu
Clone repo vào máy hoàn toàn mới, làm đúng từng bước theo README, ghi lại mọi bước bị
thiếu, gây nhầm lẫn hoặc lỗi thật — không dùng cache/setup cũ nào.

## Kết quả tổng quan
| Hạng mục | Trạng thái |
|---|---|
| Clone + cài Python deps | FAIL lần đầu (bug `httpx2`), PASS sau khi tự sửa |
| Chạy pytest | PASS (sau khi tải MovieLens 100K) |
| Chạy backend (uvicorn) | PASS |
| Chạy frontend (npm) | PASS (sau khi cài thêm Node.js — máy sạch chưa có sẵn) |
| Luồng end-to-end (đăng ký → rating → mock Top 10) | PASS |

## Vấn đề phát hiện

### 1. Bug: `httpx2` thay vì `httpx` trong `pyproject.toml` (Blocker)
`pip install -e ".[dev]"` cài đặt package `httpx2` (một package khác hoàn toàn trên
PyPI) thay vì `httpx`. `fastapi.testclient.TestClient` cần `httpx` thật để hoạt động,
nên `tests/test_api_basic.py` lỗi `ModuleNotFoundError` trên máy sạch.
**Đề xuất:** sửa `pyproject.toml`, đổi `httpx2` → `httpx`.

### 2. README thiếu hướng dẫn riêng cho PowerShell
README hiện dùng cú pháp bash (`&&`, `source .venv/bin/activate`), không chạy được
trên PowerShell mặc định của Windows (lỗi `The token '&&' is not a valid statement
separator`). Người dùng Windows cần:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```
Nếu bị chặn chạy script: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
**Đề xuất:** thêm mục "Windows (PowerShell)" song song với hướng dẫn bash trong README.

### 3. Node.js không có sẵn trên máy sạch
Máy sạch (chưa từng dev JS trước đó) không có Node.js/npm cài sẵn — cần cài riêng
(`winget install OpenJS.NodeJS.LTS`) trước khi `npm install` chạy được.
**Đề xuất:** ghi rõ yêu cầu Node.js (kèm version tối thiểu) là pre-requisite trong
README, giống như đã ghi yêu cầu Python 3.12.

### 4. MovieLens 100K không tự tải, cần chạy script riêng
3 test (`test_data_pipeline.py`, `test_mapping.py`, `test_splitting.py`) tự SKIP nếu
chưa có `data/raw/ml-100k/u.data` — đây là thiết kế đúng, không phải bug, nhưng README
nên nhắc rõ hơn là **bắt buộc** chạy `python scripts/download_data.py` trước khi chạy
full test suite, nếu không người mới dễ hiểu nhầm là test đã pass hết.

## Kết luận
Luồng end-to-end (Data → Backend → Frontend → Mock Top 10) chạy được trên máy sạch sau
khi xử lý các vấn đề trên. Không có blocker nghiêm trọng nào không sửa được trong ngày.
Đề xuất ưu tiên sửa Bug #1 trước khi thành viên khác clone máy mới, để tránh mất thời
gian debug giống mình.
