"""Shared helpers for @pytest.mark.live tests -- real api.tavily.com, no
mocking. Mirrors pipelines/technik/phases/resolve_proof/tests/live/conftest.py's
require_env()/run() pattern (per ARCHITECTURE-SPINE.md AD-9), duplicated
here rather than imported since workers/* tests are self-contained
(CLAUDE.md invariant #4 / AD-10).

Load-bearing gotcha (see spec-2-1-1): workers/tavily-search/tests/conftest.py
runs `os.environ.setdefault("TAVILY_API_KEY_1", "tvly-dev-test-key-for-pytest-only")`
for EVERY test collected under tests/, including this tests/live/
subdirectory -- pytest conftest.py files cascade down the directory tree.
A naive `if not os.environ.get("TAVILY_API_KEY_1")` guard would therefore
ALWAYS see a value set, even with zero real keys configured, and would
never skip. require_real_tavily_key() below instead reuses
settings.key_pool (the same TAVILY_API_KEY_\\d+ scan main.py's lifespan
uses to build the real key pool) and filters out that exact placeholder
value before deciding whether to skip.

Preconditions to actually run these (never required for the default
`pytest` run -- see the root pyproject.toml's `addopts = "-m 'not live'"`):
  1. Real TAVILY_API_KEY_* value(s) in workers/tavily-search/.env, exported
     into the shell that runs pytest (e.g. `set -a; source .env; set +a`)
  2. `pytest -m live workers/tavily-search/tests/live/`

Each test hits the real Tavily API with a fixed, previously-verified input
(see spec-2-1-1's Design Notes -- every assertion here was independently
confirmed via a live curl call during spec-writing, not assumed) and a
test SKIPS (not fails) when no real key is configured, so a dev machine
with no .env still runs the mocked suite cleanly.
"""
from __future__ import annotations

import asyncio

import pytest

from src.config import settings
from tests.conftest import PLACEHOLDER_TAVILY_KEY as _PLACEHOLDER_KEY


def require_real_tavily_key() -> list[str]:
    """Returns the real (non-placeholder) TAVILY_API_KEY_* values, or skips
    the test -- the standard guard every live test in this directory opens
    with. Reuses settings.key_pool rather than re-deriving the
    TAVILY_API_KEY_\\d+ env scan by hand, then filters out the
    tests/conftest.py placeholder that's ALWAYS present under tests/."""
    real_keys = [k for k in settings.key_pool if k != _PLACEHOLDER_KEY]
    if not real_keys:
        pytest.skip(
            "live test requires a real TAVILY_API_KEY_* env var "
            "(only the tests/conftest.py placeholder is set)"
        )
    return real_keys


def run(coro):
    """Same asyncio.run() wrapper the phase-level live tests use -- no
    pytest-asyncio dependency needed for these, even though this worker's
    own pytest.ini sets asyncio_mode = auto for the mocked async def tests
    elsewhere in tests/."""
    return asyncio.run(coro)
