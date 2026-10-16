import os

PLACEHOLDER_TAVILY_KEY = "tvly-dev-test-key-for-pytest-only"

os.environ.setdefault("TAVILY_API_KEY_1", PLACEHOLDER_TAVILY_KEY)


import pytest


@pytest.fixture(autouse=True)
def _fast_rate_limiter(monkeypatch):
    """Existing tests exercise the 429 ladder with millisecond backoffs; the real pool-wide
    limiter (60 calls/min, 30 s pause after a 429) would stall them. limiter.py has its own tests."""
    from src.config import settings
    monkeypatch.setattr(settings, "max_requests_per_minute", 100_000)
    monkeypatch.setattr(settings, "rate_limit_pause_seconds", 0.0)
