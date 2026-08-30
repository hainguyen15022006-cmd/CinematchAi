# Manual Test Cases — Week 1

Record PASS/FAIL plus screenshots or logs in the "Result" column on each run.

## 1. Clean machine setup
| ID | Steps | Expected | Result |
|---|---|---|---|
| SETUP-01 | Clone the repo into a new directory, follow the README from the beginning | No step fails or lacks instructions | PASS (the PowerShell venv activation issue was fixed separately — see the note at the end of the file) |
| SETUP-02 | `python -m pip install -e ".[dev]"` | Installation succeeds, no dependency errors | PASS after changing `httpx2` to `httpx`; the first run FAILED and is recorded as Bug #1 |
| SETUP-03 | `python -m pytest -v` | All tests pass | PASS after downloading and processing MovieLens 100K; 4 tests initially SKIPPED due to missing data artifacts, not an error |
| SETUP-04 | `cp .env.example .env && python scripts/seed_movies.py && uvicorn app.main:app --reload` | Server runs, `/health` returns 200 | PASS |
| SETUP-05 | `cd frontend && npm install && npm run dev` | Frontend runs on port 5173 | PASS |

## 2. Auth (Backend + Frontend)
| ID | Steps | Expected | Result |
|---|---|---|---|
| AUTH-01 | Register a new account via the UI | Created successfully, correct redirect | PASS |
| AUTH-02 | Register with an existing email | Clear error message, no crash | PASS — second attempt returns 400, message "Email already registered" |
| AUTH-03 | Log in with correct credentials | Token received, stored on the client | PASS |
| AUTH-04 | Log in with wrong password | Clear error message shown to the user | PASS — displays "invalid password" |
| AUTH-05 | Call an auth-required API without a token | Returns 401, no data exposed | PASS — returns 401, message "Missing or malformed token" |

## 3. Movies & Ratings
| ID | Steps | Expected | Result |
|---|---|---|---|
| RATE-01 | View the movie list on onboarding | Displays movie cards, titles, genres and a poster placeholder in the MVP | PASS |
| RATE-02 | Submit a valid rating (1-5) | Saved successfully (`POST /ratings` → 201 Created) | PASS |
| RATE-03 | Submit an out-of-range rating | Rejected with an error message | PASS |
| RATE-04 | Submit a rating while not logged in | Blocked / login required | PASS |

## 4. Mock Recommendation
| ID | Steps | Expected | Result |
|---|---|---|---|
| REC-01 | Call mock recommend via the UI ("Generate Top 10") | Returns the correct schema (group_score, minimum_score, disagreement, member_scores, explanations) | PASS — response fully matches the contract |
| REC-02 | Switch between the 3 strategies (Average / Least Misery / Average Without Misery) | Each strategy gives reasonably different results | PASS |
| REC-03 | Call with a non-existent Room ID (9999, 9999999) | (Mock) still returns a result; needs validation when connected to the real backend | PASS for mock, but **note**: no validation yet that the Room ID exists — needs to be tracked when the real API replaces the mock (not a bug in week 1, since the README clearly states this is a mock endpoint that does not yet require login/validation) |

## 5. Integration end-to-end
| ID | Steps | Expected | Result |
|---|---|---|---|
| E2E-01 | Register → rating → view mock Top 10 | Entire flow with no console/network errors | PASS |
| E2E-02 | Reload the page mid-flow | Login state is not lost | PASS |

## Bugs found during testing

### Bug #1 — `httpx2` instead of `httpx` in `pyproject.toml`
- **Severity:** Blocker for clean-machine setup
- **Steps to reproduce:** Clone a fresh repo → `python -m pip install -e ".[dev]"` → `python -m pytest -v`
- **Actual result:** `tests/test_api_basic.py` fails with `ModuleNotFoundError: httpx` because `pyproject.toml` declares `httpx2>=2,<3` — this is a different package on PyPI that does not provide the `httpx` module required by FastAPI `TestClient`.
- **Fix applied:** Changed `httpx2` → `httpx` in the dev-dependencies of `pyproject.toml`.
- **Suggested owner:** Chúc (Backend) or whoever maintains `pyproject.toml`.

## Environment notes
- Tested on Windows, PowerShell, Python 3.14.3 (separate venv for the project), Node.js v24.19.0 / npm 11.17.0.
- PowerShell does not support the `&&`/`source` syntax like bash — the README should add separate PowerShell instructions (activate with `.venv\Scripts\Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` if script execution is blocked).
