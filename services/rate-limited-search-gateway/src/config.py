import os
import re

from pydantic_settings import BaseSettings, SettingsConfigDict

_NUMBERED_KEY_PATTERN = re.compile(r"^TAVILY_API_KEY_(\d+)$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tavily_base_url: str = "https://api.tavily.com"

    # Raised from 20.0 (2026-09-27) — Tavily has been fast in practice, no
    # timeouts observed at 20s, so 30s just gives some extra headroom.
    http_timeout_seconds: float = 30.0

    # Real ceiling on concurrent calls to the actual Tavily API, across
    # EVERY key combined. Bumped from 5 (2026-08-28) — at the time,
    # confirmed live that Tavily had no per-key rate limit to model. That
    # assumption no longer holds (see per_key_concurrency below) — this
    # global ceiling stays as the separate, orthogonal backend-congestion
    # gate, unchanged. See .claude/specs/tavily-search-key-rotation.md.
    #
    # 2026-10-11: lowered 50 -> 10, together with per_key_concurrency 5 -> 1. A live
    # run (50 requests in flight, 3 attempts each with a 1-2 s pause, 5 s cooldown,
    # 300 s per request) sent ~900 calls/min = ~110/min per key, above Tavily's
    # 100 RPM limit for a development key, and every call came back 429 for 11
    # minutes. A small steady tap beats a big valve with retry storms: with one
    # call at a time per key a key cannot exceed ~60 calls/min.
    max_concurrent: int = 10

    # Per-key concurrency gate (2026-09-19) — live-confirmed real 429s from
    # Tavily under a burst of concurrent verify loops, contradicting this
    # worker's original "no per-key rate limit observed" assumption.
    # Mirrors openrouter-free's KeyState.semaphore/per_key_concurrency —
    # the reference architecture for a key pool with two independent
    # concurrency layers (per-key fairness/backoff + a global backend-
    # congestion ceiling). See .claude/specs/tavily-search-per-key-semaphore.md.
    per_key_concurrency:    int   = 1
    # Attempts against the SAME key on a 429 before giving up on it and
    # letting the outer pool pick a different key — held through the whole
    # sequence, see search.py's own docstring for why.
    per_key_max_retries:    int   = 3
    # Exponential backoff base for a same-key 429 retry: attempt N sleeps
    # per_key_backoff_base_s * 2**N seconds, plus jitter (see search.py).
    per_key_backoff_base_s: float = 1.0
    # After exhausting per_key_max_retries on repeated 429s, this key is
    # excluded from pick() for this long (KeyState.rate_limited_until) —
    # short and transient, unlike `exhausted` (432, permanent). Without
    # this, the outer pool could immediately re-pick the very key that
    # just proved itself rate-limited.
    per_key_rate_limit_cooldown_s: float = 5.0

    # Global limiter over ALL upstream calls (see limiter.py): at most this many calls to
    # Tavily in any 60 s, every attempt counted. Two live runs got 429 on everything after
    # ~110-120 successful /extract calls in a minute, with ~35 calls/min per key — so the
    # binding limit is not the documented 100/min per key. 60 leaves a wide margin.
    max_requests_per_minute: int = 60
    # How long every caller leaves Tavily alone after a 429 that carried no Retry-After.
    rate_limit_pause_seconds: float = 30.0

    # Wall-clock cap on retrying across the whole key pool (free + paygo)
    # for one logical search() call — mirrors openrouter-free's
    # retry_wall_clock_cap_seconds, just longer since there's no
    # jitter/backoff eating into it.
    retry_wall_clock_cap_seconds: float = 300.0

    # Tavily gives no reset timestamp anywhere (confirmed live and absent
    # from official docs) — a background task periodically re-checks every
    # exhausted key's /usage (free, uncounted) to catch a quiet reset while
    # the process was running. 6h is a deliberately loose interval — we
    # can't know the exact reset moment, so this just needs to eventually
    # notice, not be precise.
    recheck_interval_seconds: float = 6 * 3600.0

    log_level: str = "INFO"

    @property
    def key_pool(self) -> list[str]:
        """TAVILY_API_KEY_1, _2, _3, ... — one key per line, numbers need
        not be contiguous. Tier (free/paygo) is NOT declared here — it's
        determined live from each key's own /usage response at startup
        (see keys.build_key_pool) since Tavily's current_plan field is a
        static account attribute, not a lagged counter like plan_usage —
        unlike OpenRouter's key limits, there's no bootstrap problem here
        to work around with manual config."""
        numbered = sorted(
            ((int(m.group(1)), v.strip()) for k, v in os.environ.items() if (m := _NUMBERED_KEY_PATTERN.match(k)) and v.strip()),
            key=lambda pair: pair[0],
        )
        seen: set[str] = set()
        result: list[str] = []
        for _, key in numbered:
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result


settings = Settings()
