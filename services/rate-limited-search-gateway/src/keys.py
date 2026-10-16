"""Two-tier key pool: free keys are always tried first (weighted by
remaining monthly quota), paygo keys only once every free key is
exhausted. See .claude/specs/tavily-search-key-rotation.md.

No reset timestamp is available from Tavily (unlike openrouter-free's
exhausted_until, derived from OpenRouter's X-RateLimit-Reset) — confirmed
live AND absent from Tavily's own docs, so there's no exact reset date to
compute at all, not even a guess. A key marked exhausted stays excluded
from selection until a periodic background recheck (see
recheck_exhausted_keys, called from main.py's lifespan) confirms via a
fresh /usage snapshot that it actually has headroom again — this is the
ONLY way to recover a key that quietly reset while the process kept
running, since Tavily gives us nothing to compute a wake-up time from."""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx
import structlog

from .config import settings
from .limiter import RateLimiter
from .models import Depth

log = structlog.get_logger()

Tier = Literal["free", "paygo"]

_FREE_PLAN_NAME = "Researcher"   # confirmed via Tavily's own docs — the only free-tier plan name


def search_credit_cost(depth: Depth) -> float:
    """Real Tavily cost for one successful /search call, keyed by
    search_depth. TODO(unverified): assumes credits are billed from a
    single shared per-key pool regardless of endpoint (search vs. extract)
    and that this flat per-call rate doesn't vary with result count or
    plan tier — confirm against a live Tavily billing statement before
    treating this as exact."""
    if depth not in ("basic", "advanced"):
        raise ValueError(f"unknown depth: {depth!r}")
    return 2.0 if depth == "advanced" else 1.0


def extract_credit_cost(depth: Depth, successful_urls: int) -> float:
    """Real Tavily cost for one successful /extract call: a fixed
    per-resolved-URL rate (by depth) times the number of URLs that
    actually resolved (`results`, never `failed_results` — those cost
    nothing here). TODO(unverified): assumes per-call rounding of a
    per-URL rate rather than some continuous/aggregate billing rate, and
    assumes a failed URL is genuinely free — neither confirmed against a
    live Tavily billing statement."""
    if depth not in ("basic", "advanced"):
        raise ValueError(f"unknown depth: {depth!r}")
    per_url = 0.4 if depth == "advanced" else 0.2
    return successful_urls * per_url


@dataclass
class KeyState:
    key: str
    tier: Tier
    plan_limit: int = 1000        # seeded from /usage at startup; meaningless for paygo (no cap to model)
    used: float = 0.0             # local counter, incremented by real credit cost on every confirmed success since startup
    exhausted: bool = False       # permanent for this process's lifetime — see module docstring
    # Per-key concurrency gate (2026-09-19), mirrors openrouter-free's
    # KeyState.semaphore — held continuously through a whole attempt +
    # backoff + retry sequence in search.py, never released mid-backoff
    # (releasing there would let another caller slip into the just-freed
    # slot and hit this same key at the exact moment it's known to be
    # rate-limited). Set once at pool-build time (build_key_pool), never
    # reassigned. See .claude/specs/tavily-search-per-key-semaphore.md.
    semaphore: asyncio.Semaphore = field(default=None)
    # Short, transient cooldown after exhausting per-key 429 retries — NOT
    # the same thing as `exhausted` (432, permanent for the process, no
    # reset timestamp available from Tavily at all). A 429 is transient by
    # definition, so unlike `exhausted` this DOES have a real wake-up time:
    # whatever the caller sets it to (search.py, right after giving up on
    # this key for now). time.monotonic()-based, process-local, never
    # persisted. See .claude/specs/tavily-search-per-key-semaphore.md.
    rate_limited_until: float | None = None

    def is_available(self, now: float) -> bool:
        return self.rate_limited_until is None or self.rate_limited_until <= now

    def remaining(self) -> float:
        return max(self.plan_limit - self.used, 0)

    def weight(self) -> float:
        """No floor here, unlike openrouter-free's KeyState.weight — that
        floor existed because OpenRouter's real per-key limit was only
        ever discoverable by risking a real call past an assumed cap.
        Tavily hands us the real plan_limit directly (no guessing), and
        the free /usage endpoint (see recheck_exhausted_keys) is how a
        key ever proves it has headroom again — never by spending a real
        search credit on a key we're confident is empty. A clean 0 here
        is correct, not a gap."""
        return self.remaining() ** 2

    def check_exhaustion(self) -> None:
        """Proactively marks exhausted the moment our own accounting says
        remaining hits 0 — called after every local increment AND right
        after seeding at startup, so a key that's already at/over its
        limit at boot (real, observed live: 1015/1000) never sits in the
        candidate pool with a literal-0 weight (random.choices raises if
        EVERY candidate weight is 0)."""
        if self.remaining() <= 0:
            self.exhausted = True


class TavilyKeyPool:
    def __init__(self, free_keys: list[KeyState], paygo_keys: list[KeyState], limiter: RateLimiter | None = None):
        if not free_keys and not paygo_keys:
            raise ValueError("TavilyKeyPool requires at least one key")
        # one limiter for the whole pool: the observed 429s were not per key (see limiter.py)
        self.limiter = limiter or RateLimiter(settings.max_requests_per_minute)
        self._free = free_keys
        self._paygo = paygo_keys
        self._paygo_next_index = 0

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_paygo(self) -> int:
        return len(self._paygo)

    def any_available(self) -> bool:
        return any(not s.exhausted for s in self._free) or any(not s.exhausted for s in self._paygo)

    def pick(self) -> KeyState | None:
        """Weighted-random among available free keys first; only considers
        paygo keys (plain round-robin — no meaningful "remaining" concept
        for an unlimited-billing key) once every free key is exhausted.
        None when nothing is eligible RIGHT NOW — either permanently
        exhausted (432) or currently cooling down after a 429 (see
        KeyState.is_available). The two are different to the caller: see
        search.py, which treats a None here as "wait and retry" unless
        any_available() also says false (truly nothing left, ever)."""
        now = time.monotonic()
        free_candidates = [s for s in self._free if not s.exhausted and s.is_available(now)]
        if free_candidates:
            weights = [s.weight() for s in free_candidates]
            return random.choices(free_candidates, weights=weights, k=1)[0]

        n = len(self._paygo)
        for offset in range(n):
            idx = (self._paygo_next_index + offset) % n
            state = self._paygo[idx]
            if not state.exhausted and state.is_available(now):
                self._paygo_next_index = (idx + 1) % n
                return state
        return None

    def mark_exhausted(self, key: str) -> None:
        for state in (*self._free, *self._paygo):
            if state.key == key:
                state.exhausted = True
                return

    def record_attempt(self, key: str, credit_cost: float) -> None:
        """Called only after a confirmed 2xx from Tavily (see search.py's
        and extract.py's shared ladder helper) — never on a 429/432/other
        non-2xx branch, since those never actually spent a credit.
        `credit_cost` comes from search_credit_cost()/extract_credit_cost()
        above, computed by the caller from the real outcome (e.g. 0.0 for
        an /extract call whose single url fully failed)."""
        for state in (*self._free, *self._paygo):
            if state.key == key:
                state.used += credit_cost
                state.check_exhaustion()
                return

    def exhausted_states(self) -> list[KeyState]:
        return [s for s in (*self._free, *self._paygo) if s.exhausted]

    def reset_key(self, key: str, plan_limit: int, used: int) -> None:
        """Called by recheck_exhausted_keys once a fresh /usage snapshot
        confirms real headroom — un-exhausts the key and re-seeds its
        counters from the fresh numbers (never assumes 0)."""
        for state in (*self._free, *self._paygo):
            if state.key == key:
                state.exhausted = False
                state.plan_limit = plan_limit
                state.used = used
                return


async def fetch_usage(http: httpx.AsyncClient, key: str) -> dict | None:
    """GET /usage — free, uncounted (confirmed live: repeated calls don't
    move the counter), but undocumented/unofficial and confirmed to lag
    real usage by 45s+ — used here only as a one-time startup snapshot and
    tier cross-check, never as a live gate."""
    try:
        resp = await http.get(f"{settings.tavily_base_url}/usage", headers={"Authorization": f"Bearer {key}"})
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("tavily /usage unreachable at startup — continuing with defaults", key_suffix=key[-4:], error=str(e))
        return None


async def build_key_pool(http: httpx.AsyncClient, keys: list[str]) -> TavilyKeyPool:
    """Tier is determined live per key from /usage's current_plan — no
    manual .env declaration needed. Unlike plan_usage/plan_limit (lagged
    counters, confirmed live: 45s+ stale after real usage), current_plan is
    a static account attribute that doesn't have that problem, so there's
    no bootstrap issue here to work around with explicit config (unlike
    OpenRouter's key limits, where the real value is only ever revealed
    reactively on a failure). /usage unreachable at startup → defaults to
    "free" (the safe direction: worst case a paygo key gets tried earlier
    than ideal, never later — no risk of overspending from this alone)."""
    free_states: list[KeyState] = []
    paygo_states: list[KeyState] = []

    for key in keys:
        usage = await fetch_usage(http, key)
        if usage is None:
            free_states.append(KeyState(key=key, tier="free", semaphore=asyncio.Semaphore(settings.per_key_concurrency)))
            continue

        account = usage.get("account", {})
        current_plan = account.get("current_plan")
        tier: Tier = "free" if current_plan == _FREE_PLAN_NAME else "paygo"
        state = KeyState(key=key, tier=tier, semaphore=asyncio.Semaphore(settings.per_key_concurrency))
        if tier == "free":
            if account.get("plan_limit") is not None:
                state.plan_limit = account["plan_limit"]
            state.used = account.get("plan_usage", 0)
            state.check_exhaustion()   # already at/over limit at boot (real, observed live: 1015/1000)
            free_states.append(state)
        else:
            paygo_states.append(state)

        log.info(
            "tavily key seeded at startup", key_suffix=key[-4:], tier=tier,
            plan_limit=state.plan_limit, used=state.used, remaining=state.remaining(), weight=state.weight(),
        )

    return TavilyKeyPool(free_states, paygo_states)


async def recheck_exhausted_keys(http: httpx.AsyncClient, key_pool: TavilyKeyPool) -> None:
    """Periodic background job (see main.py's lifespan) — the only way to
    recover a key that reset while the process was running, since Tavily
    gives no reset timestamp to schedule around. /usage is free/uncounted,
    so this costs nothing quota-wise no matter how often it runs."""
    for state in key_pool.exhausted_states():
        usage = await fetch_usage(http, state.key)
        if usage is None:
            continue
        account = usage.get("account", {})
        plan_usage = account.get("plan_usage")
        plan_limit = account.get("plan_limit")
        if plan_usage is None or plan_limit is None:
            continue
        if plan_usage < plan_limit:
            key_pool.reset_key(state.key, plan_limit, plan_usage)
            log.info("tavily key recovered after apparent reset", key_suffix=state.key[-4:], tier=state.tier, plan_usage=plan_usage, plan_limit=plan_limit)
