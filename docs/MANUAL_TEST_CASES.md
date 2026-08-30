# Manual Test Cases — Tuần 1

Ghi kết quả PASS/FAIL + ảnh chụp hoặc log vào cột "Kết quả" mỗi lần chạy.

## 1. Setup máy sạch
| ID | Bước | Kỳ vọng | Kết quả |
|---|---|---|---|
| SETUP-01 | Clone repo vào thư mục mới, làm theo README từ đầu | Không bước nào lỗi/thiếu hướng dẫn | PASS (đã fix riêng vấn đề PowerShell venv activation — xem ghi chú cuối file) |
| SETUP-02 | `python -m pip install -e ".[dev]"` | Cài đặt thành công, không lỗi dependency | FAIL trên máy sạch — xem Bug #1 (`httpx2`) cuối file |
| SETUP-03 | `python -m pytest -v` | Toàn bộ test pass | PASS sau khi tải MovieLens 100K (`scripts/download_data.py`); 3 test ban đầu SKIP vì thiếu data, không phải lỗi |
| SETUP-04 | `cp .env.example .env && python scripts/seed_movies.py && uvicorn app.main:app --reload` | Server chạy, `/health` trả 200 | PASS |
| SETUP-05 | `cd frontend && npm install && npm run dev` | Frontend chạy ở cổng 5173 | PASS |

## 2. Auth (Backend + Frontend)
| ID | Bước | Kỳ vọng | Kết quả |
|---|---|---|---|
| AUTH-01 | Đăng ký tài khoản mới qua UI | Tạo thành công, chuyển hướng đúng | PASS |
| AUTH-02 | Đăng ký với email đã tồn tại | Báo lỗi rõ ràng, không crash | PASS — lần 2 trả 400, message "Email already registered" |
| AUTH-03 | Đăng nhập đúng thông tin | Nhận token, lưu vào client | PASS |
| AUTH-04 | Đăng nhập sai mật khẩu | Báo lỗi rõ ràng cho người dùng | PASS — hiển thị "invalid password" |
| AUTH-05 | Gọi API cần auth mà không có token | Trả 401, không lộ dữ liệu | PASS — trả 401, message "Missing or malformed token" |

## 3. Movies & Ratings
| ID | Bước | Kỳ vọng | Kết quả |
|---|---|---|---|
| RATE-01 | Xem danh sách phim trên onboarding | Hiển thị đủ poster/tên/thể loại | PASS |
| RATE-02 | Gửi rating hợp lệ (1-5) | Lưu thành công (`POST /ratings` → 201 Created) | PASS |
| RATE-03 | Gửi rating ngoài khoảng | Bị từ chối, thông báo lỗi | PASS |
| RATE-04 | Gửi rating khi chưa đăng nhập | Bị chặn / yêu cầu đăng nhập | PASS |

## 4. Mock Recommendation
| ID | Bước | Kỳ vọng | Kết quả |
|---|---|---|---|
| REC-01 | Gọi mock recommend qua UI ("Tạo Top 10") | Trả đúng schema (group_score, minimum_score, disagreement, member_scores, explanations) | PASS — response khớp đầy đủ contract |
| REC-02 | Đổi giữa 3 strategy (Average / Least Misery / Average Without Misery) | Mỗi strategy cho kết quả khác nhau hợp lý | PASS |
| REC-03 | Gọi với Room ID không tồn tại (9999, 9999999) | (Mock) vẫn trả kết quả; cần validate khi nối backend thật | PASS cho mock, nhưng **ghi chú**: chưa có validation Room ID tồn tại — cần theo dõi khi API thật thay thế mock (không phải bug ở tuần 1, vì README ghi rõ đây là mock endpoint chưa cần đăng nhập/validate) |

## 5. Integration end-to-end
| ID | Bước | Kỳ vọng | Kết quả |
|---|---|---|---|
| E2E-01 | Đăng ký → rating → xem mock Top 10 | Toàn bộ luồng không lỗi console/network | PASS |
| E2E-02 | Reload trang giữa luồng | Không mất trạng thái đăng nhập | PASS |

## Bug tìm thấy trong quá trình test

### Bug #1 — `httpx2` thay vì `httpx` trong `pyproject.toml`
- **Mức độ:** Blocker cho clean-machine setup
- **Bước tái hiện:** Clone repo mới → `python -m pip install -e ".[dev]"` → `python -m pytest -v`
- **Kết quả thực tế:** `tests/test_api_basic.py` lỗi `ModuleNotFoundError: httpx` vì `pyproject.toml` khai `httpx2>=2,<3` — đây là package khác trên PyPI, không cung cấp module `httpx` mà FastAPI `TestClient` cần.
- **Đề xuất fix:** Đổi `httpx2` → `httpx` trong dev-dependencies của `pyproject.toml`.
- **Owner đề xuất:** Chúc (Backend) hoặc người quản lý `pyproject.toml`.

## Ghi chú môi trường
- Test trên Windows, PowerShell, Python 3.14.3 (venv riêng cho project), Node.js v24.19.0 / npm 11.17.0.
- PowerShell không hỗ trợ cú pháp `&&`/`source` như bash — README nên bổ sung hướng dẫn PowerShell riêng (activate bằng `.venv\Scripts\Activate.ps1`, chạy `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` nếu bị chặn script).
