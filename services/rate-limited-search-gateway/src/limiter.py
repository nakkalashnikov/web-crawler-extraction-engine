"""Global request-rate limiter for every call to Tavily, across all keys (2026-10-11).

Why: two live runs of a research phase tripped Tavily's 429 after the same ~110-120
successful /extract calls inside one minute, and then answered 429 to every call for
as long as the run went on, although each key was getting only ~35 calls/min (the
documented development-key limit is 100/min per key). So the limit that bites is not
per key, and limiting CONCURRENCY does not keep a fast API under a rate limit. This
limiter does what Tavily's own best-practices advice says: stay under a request rate.

- a sliding window: at most `max_per_minute` upstream calls in any 60 s, counting every
  attempt (a retry is a request too);
- a global pause: after a 429 nobody calls Tavily until the pause is over (the
  `Retry-After` the API sent, else a default), so a rate-limited service is left alone
  instead of being hit ~900 times a minute by retries.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Awaitable, Callable


class RateLimitWaitExceeded(RuntimeError):
    """The deadline of the logical call passed while it waited for a slot."""


class RateLimiter:
    def __init__(
        self, max_per_minute: int, window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_per_minute < 1:
            raise ValueError("max_per_minute must be at least 1")
        self._max = max_per_minute
        self._window = window_s
        self._clock, self._sleep = clock, sleep
        self._stamps: deque[float] = deque()
        self._paused_until = 0.0

    @property
    def paused_for(self) -> float:
        return max(0.0, self._paused_until - self._clock())

    def penalize(self, seconds: float) -> None:
        """A 429 arrived: pause every caller for `seconds` (never shortens a longer pause)."""
        self._paused_until = max(self._paused_until, self._clock() + seconds)

    async def acquire(self, deadline: float) -> None:
        """Waits for a free slot and for the pause to end; raises RateLimitWaitExceeded past `deadline`."""
        while True:
            now = self._clock()
            if now >= deadline:
                raise RateLimitWaitExceeded("waited for a rate-limiter slot past the deadline")
            if now < self._paused_until:
                await self._sleep(min(self._paused_until - now, deadline - now))
                continue
            while self._stamps and self._stamps[0] <= now - self._window:
                self._stamps.popleft()
            if len(self._stamps) < self._max:
                self._stamps.append(now)   # no await between the check and this line: one waiter wins the slot
                return
            await self._sleep(min(self._stamps[0] + self._window - now, deadline - now))
