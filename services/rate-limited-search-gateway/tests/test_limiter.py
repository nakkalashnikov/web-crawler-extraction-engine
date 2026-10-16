"""RateLimiter (src/limiter.py): sliding window + global pause, on a fake clock."""
import pytest

from src.limiter import RateLimitWaitExceeded, RateLimiter


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


def _limiter(n: int, clock: _Clock) -> RateLimiter:
    return RateLimiter(n, clock=clock, sleep=clock.sleep)


async def test_calls_under_the_limit_never_wait():
    c = _Clock()
    lim = _limiter(3, c)
    for _ in range(3):
        await lim.acquire(deadline=c.now + 100)
    assert c.slept == []


async def test_the_call_over_the_limit_waits_until_the_oldest_leaves_the_window():
    c = _Clock()
    lim = _limiter(2, c)
    await lim.acquire(c.now + 100)
    c.now += 10
    await lim.acquire(c.now + 100)
    await lim.acquire(c.now + 100)          # third one inside 60 s of the first
    assert c.slept == [50.0]                # 60 - 10 s since the first call


async def test_a_penalty_pauses_everyone_and_never_shortens_a_longer_pause():
    c = _Clock()
    lim = _limiter(100, c)
    lim.penalize(30)
    lim.penalize(5)
    assert lim.paused_for == 30
    await lim.acquire(c.now + 100)
    assert c.slept == [30.0]


async def test_waiting_past_the_deadline_raises_instead_of_waiting_forever():
    c = _Clock()
    lim = _limiter(1, c)
    await lim.acquire(c.now + 100)
    with pytest.raises(RateLimitWaitExceeded):
        await lim.acquire(c.now + 10)       # next slot is 60 s away


async def test_at_most_n_calls_in_any_window():
    c = _Clock()
    lim = _limiter(5, c)
    stamps = []
    for _ in range(40):
        await lim.acquire(c.now + 10_000)
        stamps.append(c.now)
    assert all(sum(1 for s in stamps if t - 60 < s <= t) <= 5 for t in stamps)


def test_a_limit_below_one_is_rejected():
    with pytest.raises(ValueError):
        RateLimiter(0)
