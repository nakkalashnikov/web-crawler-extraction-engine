from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable

import httpx
import structlog

from .config import settings
from .keys import KeyState, TavilyKeyPool, search_credit_cost
from .limiter import RateLimitWaitExceeded
from .models import Depth, SearchResult, Topic

log = structlog.get_logger()


class TavilySearchFailedError(RuntimeError):
    """Every key in the pool (free + paygo) was exhausted, or the
    wall-clock deadline was hit, without a successful call to either
    /search or /extract — carries Tavily's own last error verbatim rather
    than a bare unhandled httpx.HTTPStatusError (the exact lesson learned
    on openrouter-free's CompletionsFailedError this session)."""

    def __init__(self, status_code: int, error_body: dict | str):
        self.status_code = status_code
        self.error_body = error_body
        super().__init__(f"Tavily request failed after exhausting the key pool: {status_code} — {error_body}")


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """The API's own Retry-After (seconds), when it sent a usable one."""
    try:
        value = float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _rate_limit_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() == "retry-after" or "ratelimit" in k.lower() or "rate-limit" in k.lower()}


def _safe_body(resp: httpx.Response) -> dict | str:
    try:
        return resp.json()
    except ValueError:
        return resp.text


async def _call_with_key_pool(
    http: httpx.AsyncClient,
    key_pool: TavilyKeyPool,
    attempt_fn: Callable[[KeyState], Awaitable[httpx.Response]],
    deadline: float,
) -> httpx.Response:
    """Shared retry/rotation ladder for both search() and extract() — pick
    a key, hold its per-key semaphore, POST via attempt_fn, and react to
    the response:

    - 432 ("plan limit exceeded") — immediate switch to a different key,
      no backoff, since a 432 isn't transient and waiting on this key
      gains nothing.
    - 429 (rate limited) — retried against the SAME key with exponential
      backoff, up to per_key_max_retries times, with that key's own
      semaphore held CONTINUOUSLY across every attempt and every backoff
      sleep (see KeyState.semaphore's own docstring for why releasing
      mid-backoff would defeat the point). Only after exhausting the
      per-key retry budget does control return to the outer loop to pick
      a different key.
    - any other error status raises immediately via raise_for_status(),
      unchanged from before this pool ever existed.
    - a confirmed 2xx is returned as-is to the caller. Deliberately does
      NOT call key_pool.record_attempt() itself — that requires the real
      credit cost, which depends on endpoint-specific response parsing
      (e.g. extract's per-successful-url rate) that only search()/
      extract() know how to compute. Callers must record the attempt
      themselves, from this returned response, after this call returns."""
    last_status = 432
    last_body: dict | str = "key pool exhausted"

    while time.monotonic() < deadline:
        key_state = key_pool.pick()
        if key_state is None:
            if not key_pool.any_available():
                # Nothing left at all, ever (every key permanently
                # exhausted) — same fast-fail as before this pool existed.
                raise TavilySearchFailedError(last_status, last_body)
            # Every key is either exhausted or mid-429-cooldown, but at
            # least one WILL come back — a short sleep avoids a tight spin
            # loop while we wait, bounded by the same wall-clock deadline.
            await asyncio.sleep(0.1)
            continue

        log.info(
            "tavily key picked", key_suffix=key_state.key[-4:], tier=key_state.tier,
            remaining=key_state.remaining(), weight=key_state.weight(),
        )

        async with key_state.semaphore:
            for attempt in range(settings.per_key_max_retries):
                try:
                    # every upstream call, retries included, takes a slot of the pool-wide limiter
                    await key_pool.limiter.acquire(deadline)
                except RateLimitWaitExceeded:
                    raise TavilySearchFailedError(last_status if last_status != 432 else 429, last_body)
                try:
                    resp = await attempt_fn(key_state)
                except httpx.TransportError as exc:
                    # A raw connection/timeout error (not an HTTP response at
                    # all) used to propagate straight out of this loop,
                    # skipping the entire 429/432 retry-rotation ladder below
                    # — a single-shot failure even though the SAME url/key
                    # often succeeds on a later attempt (live-confirmed:
                    # alternating success/failure for one url across
                    # consecutive calls). Treated exactly like a 429 on this
                    # key: retried with backoff, semaphore held, before
                    # falling through to the outer loop's next key.
                    last_status, last_body = 599, str(exc)
                    if attempt < settings.per_key_max_retries - 1:
                        delay = settings.per_key_backoff_base_s * (2 ** attempt) + random.uniform(0, 1)
                        log.info(
                            "tavily call raised a transport error — backing off, retrying same key (semaphore held)",
                            key_suffix=key_state.key[-4:], attempt=attempt + 1, delay_s=round(delay, 2), error=str(exc),
                        )
                        await asyncio.sleep(delay)
                    continue

                if resp.status_code == 429:
                    last_status, last_body = resp.status_code, _safe_body(resp)
                    # Leave Tavily alone: nobody calls it until the pause is over. Retry-After
                    # when it sent one, else the default. The body and rate-limit headers are
                    # logged because the first live 429s did not show which limit was hit.
                    pause = _retry_after_seconds(resp) or settings.rate_limit_pause_seconds
                    key_pool.limiter.penalize(pause)
                    log.warning(
                        "tavily 429 — pausing every call", key_suffix=key_state.key[-4:], pause_s=pause,
                        headers=_rate_limit_headers(resp), body=str(last_body)[:300],
                    )
                    if attempt < settings.per_key_max_retries - 1:
                        delay = settings.per_key_backoff_base_s * (2 ** attempt) + random.uniform(0, 1)
                        log.info(
                            "tavily key rate-limited — backing off, retrying same key (semaphore held)",
                            key_suffix=key_state.key[-4:], attempt=attempt + 1, delay_s=round(delay, 2),
                        )
                        await asyncio.sleep(delay)
                    continue

                if resp.status_code == 432:
                    key_pool.mark_exhausted(key_state.key)
                    last_status, last_body = resp.status_code, _safe_body(resp)
                    log.info("tavily key exhausted", key_suffix=key_state.key[-4:], tier=key_state.tier)
                    break

                resp.raise_for_status()
                log.info("tavily call succeeded", key_suffix=key_state.key[-4:], tier=key_state.tier)
                return resp
            else:
                # for-loop ran out of attempts WITHOUT hitting break/return —
                # every attempt was a 429 or a transport error. Short cooldown
                # so the outer loop's
                # next pick() doesn't immediately re-select the very key
                # that just proved itself rate-limited (a 432 already
                # excludes itself via mark_exhausted above; this is the same
                # idea, just transient).
                key_state.rate_limited_until = time.monotonic() + settings.per_key_rate_limit_cooldown_s
                log.info(
                    "tavily key cooling down after repeated 429s", key_suffix=key_state.key[-4:],
                    cooldown_s=settings.per_key_rate_limit_cooldown_s,
                )
            # Per-key retry budget exhausted on repeated 429s, or a 432 broke
            # out above — either way fall through here, releasing this key's
            # semaphore (async with exits), and let the outer loop's next
            # pick() (top of this while loop) pick a different key.

    raise TavilySearchFailedError(last_status, last_body)


def _key_from_response(resp: httpx.Response) -> str:
    return resp.request.headers["authorization"].removeprefix("Bearer ")


async def search(
    http: httpx.AsyncClient, key_pool: TavilyKeyPool, query: str, include_domains: list[str], max_results: int | None,
    search_depth: Depth = "advanced", topic: Topic = "general",
) -> list[SearchResult]:
    """POST /search — "basic" was tried first but verified to miss relevant
    results entirely (not just coarser ranking) — e.g. a same-domain,
    same-query search for a real, on-topic article returned 0/5 hits on
    "basic" and found it at #1 on "advanced", reproducibly. Defaults to
    "advanced" (2 credits/call) for that reason; callers may still pass
    "basic" (1 credit/call) explicitly when that empirical risk is
    acceptable for their use case.

    Retry/rotation ladder is shared with extract() via
    _call_with_key_pool. `key_pool.record_attempt` fires only once, here,
    after a confirmed 2xx — never inside the ladder, and never on a
    429/432/other non-2xx branch.

    `topic` is sent to Tavily only when it is "news" — "general" is Tavily's
    own default, so existing callers' requests stay byte-identical."""
    deadline = time.monotonic() + settings.retry_wall_clock_cap_seconds

    payload = {
        "query":           query,
        "include_domains": include_domains,
        "search_depth":    search_depth,
    }
    # Omitted when None so Tavily applies its own default instead of ours.
    if max_results is not None:
        payload["max_results"] = max_results
    if topic != "general":
        payload["topic"] = topic

    async def attempt_fn(key_state: KeyState) -> httpx.Response:
        return await http.post(
            f"{settings.tavily_base_url}/search",
            headers = {"Authorization": f"Bearer {key_state.key}"},
            json = payload,
            timeout = settings.http_timeout_seconds,
        )

    resp = await _call_with_key_pool(http, key_pool, attempt_fn, deadline)
    key_pool.record_attempt(_key_from_response(resp), search_credit_cost(search_depth))
    data = resp.json()
    return [
        SearchResult(
            url=r["url"], title=r["title"], content=r["content"], score=r["score"],
            published_date=r.get("published_date"),
        )
        for r in data.get("results", [])
    ]
