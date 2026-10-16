FROM python:3.12-slim

WORKDIR /app

COPY workers/source-extract/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flat cross-worker client — this worker calls tavily-search over HTTP via
# `from workers import tavily_search` (the one sanctioned worker-calling-
# worker exception, per Story 2.1). Build context is the repo root
# (see docker-compose.yml) so these are reachable.
COPY workers/__init__.py ./workers/__init__.py
COPY workers/tavily_search.py ./workers/tavily_search.py

COPY workers/source-extract/src/ ./src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
