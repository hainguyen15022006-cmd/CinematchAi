# Backend image for CineMatch. Built for a CPU-only host such as the free
# plans of Render, Railway or Fly.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the CPU-only PyTorch build FIRST. The default PyPI wheel drags in
# several gigabytes of CUDA libraries that a CPU host never loads, which is
# what makes a naive build run out of disk or memory on a free plan.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"

# Dependency metadata first so Docker can cache the slow install layer.
COPY pyproject.toml ./
COPY src ./src
COPY app ./app
RUN pip install -e .

# Application code and the small serving bundle (model, mappings, catalog).
COPY scripts ./scripts
COPY configs ./configs
COPY artifacts ./artifacts

# SQLite lives next to the code. On a free plan this disk is wiped on every
# deploy, so set DATABASE_URL to a managed Postgres for durable accounts.
ENV DATABASE_URL=sqlite:////app/data/cinematch.db \
    SERVING_ARTIFACTS_DIR=/app/artifacts/serving \
    PORT=8000
RUN mkdir -p /app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
