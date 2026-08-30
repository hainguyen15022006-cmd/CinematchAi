# Clean Machine Verification Report — Week 1

**Performed by:** Hoàng Anh
**Official project environment:** Python 3.12, Node.js 20 (matching the versions used in
`ci.yml`).
**Personal machine environment used for additional checks:** Windows, PowerShell, VS Code,
Python 3.14.3 (separate venv, only to check compatibility with a newer Python release
— not used as the reporting standard), Node.js v24.19.0.

## Objective
Clone the repo onto a completely fresh machine, follow the README step by step, and record every
step that is missing, confusing or actually broken — without using any existing cache/setup.

## Overall results
| Item | Status |
|---|---|
| Clone + install Python deps | FAIL on first attempt (`httpx2` bug) → **fixed (RESOLVED)** in this PR |
| Run pytest | PASS (after downloading and processing MovieLens 100K; 4 tests depending on data artifacts skip on CI, do not skip when run manually with data) |
| Run backend (uvicorn) | PASS |
| Run frontend (npm) | PASS (after additionally installing Node.js — not preinstalled on the clean machine) |
| End-to-end flow (register → rating → mock Top 10) | PASS |

## Bug handled (RESOLVED in this PR)

### 1. `httpx2` instead of `httpx` in `pyproject.toml`
`pip install -e ".[dev]"` declares `httpx2>=2,<3` — both the package name and the version
constraint are wrong (the real `httpx` package on PyPI only has `0.x` releases, there is no `2.x`).
`fastapi.testclient.TestClient` needs the real `httpx` to work, so
`tests/test_api_basic.py` fails with `ModuleNotFoundError` on a clean machine.

**Fix applied:** changed to `"httpx>=0.27,<1"` in `pyproject.toml`. Confirmed that
`pip install -e ".[dev]"` and `python -m pytest -v` PASS entirely after the fix.

## Remaining work (not handled in this PR — proposed for next week)

### 2. README lacks separate instructions for PowerShell
The README currently uses bash syntax (`&&`, `source .venv/bin/activate`), which does not run
on the default Windows PowerShell (error `The token '&&' is not a valid statement
separator`). Windows users need to use:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```
If script execution is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
**Proposal:** add a "Windows (PowerShell)" section alongside the bash instructions in the README.

### 3. Node.js not available on the clean machine
The clean machine (never used for JS development before) has no Node.js/npm preinstalled — it must be
installed separately (`winget install OpenJS.NodeJS.LTS`, **Node 20** recommended to match `ci.yml`)
before `npm install` can run.
**Proposal:** clearly state the Node.js 20 requirement (with a minimum version) as a prerequisite in the
README, just as the Python 3.12 requirement is stated.

### 4. MovieLens 100K is not downloaded automatically; a separate script must be run
4 tests (`test_candidates.py`, `test_data_pipeline.py`, `test_mapping.py`,
`test_splitting.py`) SKIP themselves if the raw or processed data artifacts are missing — this is
the correct design, not a bug. The current CI does not download or process MovieLens, so these tests
always skip on CI (see `TEST_PLAN.md`). The README should state more clearly that both
`python scripts/download_data.py` and `python scripts/prepare_data.py` must be run before running
the full test suite manually; otherwise, newcomers may easily assume all tests passed when
some tests were actually skipped.

## Conclusion
The end-to-end flow (Data → Backend → Frontend → Mock Top 10) runs on a clean machine.
The only blocker bug (`httpx2`) was fixed in this very PR. The 3 remaining items (sections
2–4) are not blockers, only documentation improvements — proposed for a later PR, with no need
to block merging the current PR.
