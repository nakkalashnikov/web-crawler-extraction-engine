import httpx
import pytest
import respx

from src.keys import (
    KeyState,
    TavilyKeyPool,
    build_key_pool,
    extract_credit_cost,
    fetch_usage,
    recheck_exhausted_keys,
    search_credit_cost,
)


class TestKeyState:
    def test_remaining_never_negative(self):
        state = KeyState(key="k1", tier="free", plan_limit=1000, used=1500)
        assert state.remaining() == 0

    def test_weight_is_a_clean_zero_when_remaining_is_zero(self):
        """Unlike openrouter-free's floor-of-1: Tavily's plan_limit is a
        real number from the API, not a guess, so there's no bootstrap
        problem to protect against here — a genuinely empty key gets a
        literal 0, and check_exhaustion() (not weight) is what keeps it
        out of the weighted-random pool entirely."""
        state = KeyState(key="k1", tier="free", plan_limit=1000, used=1000)
        assert state.weight() == 0

    def test_check_exhaustion_marks_exhausted_when_remaining_hits_zero(self):
        state = KeyState(key="k1", tier="free", plan_limit=1000, used=1000)
        state.check_exhaustion()
        assert state.exhausted is True

    def test_check_exhaustion_leaves_key_with_headroom_alone(self):
        state = KeyState(key="k1", tier="free", plan_limit=1000, used=500)
        state.check_exhaustion()
        assert state.exhausted is False


class TestTavilyKeyPool:
    def test_empty_pool_rejected(self):
        with pytest.raises(ValueError):
            TavilyKeyPool([], [])

    def test_picks_free_over_paygo_while_free_available(self):
        free = [KeyState(key="f1", tier="free")]
        paygo = [KeyState(key="p1", tier="paygo")]
        pool = TavilyKeyPool(free, paygo)
        picks = {pool.pick().key for _ in range(20)}
        assert picks == {"f1"}

    def test_falls_through_to_paygo_once_all_free_exhausted(self):
        free = [KeyState(key="f1", tier="free")]
        paygo = [KeyState(key="p1", tier="paygo")]
        pool = TavilyKeyPool(free, paygo)
        pool.mark_exhausted("f1")
        assert pool.pick().key == "p1"

    def test_paygo_round_robins(self):
        pool = TavilyKeyPool([], [KeyState(key="p1", tier="paygo"), KeyState(key="p2", tier="paygo")])
        picks = [pool.pick().key for _ in range(4)]
        assert picks == ["p1", "p2", "p1", "p2"]

    def test_pick_returns_none_when_entire_pool_exhausted(self):
        pool = TavilyKeyPool([KeyState(key="f1", tier="free")], [KeyState(key="p1", tier="paygo")])
        pool.mark_exhausted("f1")
        pool.mark_exhausted("p1")
        assert pool.pick() is None
        assert pool.any_available() is False

    def test_weighted_pick_favors_key_with_more_remaining(self):
        rich = KeyState(key="rich", tier="free", plan_limit=1000, used=0)
        poor = KeyState(key="poor", tier="free", plan_limit=1000, used=990)
        pool = TavilyKeyPool([rich, poor], [])
        picks = [pool.pick().key for _ in range(200)]
        assert picks.count("rich") > picks.count("poor")

    def test_record_attempt_increments_used_by_credit_cost(self):
        pool = TavilyKeyPool([KeyState(key="f1", tier="free")], [])
        pool.record_attempt("f1", 2.0)
        pool.record_attempt("f1", 1.0)
        assert pool._free[0].used == 3.0

    def test_record_attempt_proactively_exhausts_key_at_limit(self):
        """No need to wait for a real 432 — once our own accounting says
        remaining hit 0, the key is excluded immediately."""
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", plan_limit=1, used=0)], [])
        pool.record_attempt("f1", 1.0)
        assert pool._free[0].exhausted is True
        assert pool.pick() is None

    def test_record_attempt_with_zero_cost_does_not_exhaust(self):
        """A confirmed 2xx whose business outcome fully failed (e.g. an
        /extract call whose single url never resolved) still calls
        record_attempt, but with credit_cost=0.0 — must not move `used`."""
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", plan_limit=1000, used=500)], [])
        pool.record_attempt("f1", 0.0)
        assert pool._free[0].used == 500.0
        assert pool._free[0].exhausted is False


class TestCreditCostFunctions:
    def test_search_credit_cost_advanced_is_2(self):
        assert search_credit_cost("advanced") == 2.0

    def test_search_credit_cost_basic_is_1(self):
        assert search_credit_cost("basic") == 1.0

    def test_extract_credit_cost_advanced_per_url_is_0_4(self):
        assert extract_credit_cost("advanced", 3) == pytest.approx(1.2)

    def test_extract_credit_cost_basic_per_url_is_0_2(self):
        assert extract_credit_cost("basic", 3) == pytest.approx(0.6)

    def test_extract_credit_cost_zero_successful_urls_is_zero(self):
        assert extract_credit_cost("advanced", 0) == 0.0

    def test_pick_does_not_crash_when_all_free_keys_seeded_already_over_limit(self):
        """Real case caught live: a key seeded at 1015/1000 must not sit in
        the weighted-random pool with a literal-0 weight — random.choices
        raises if every candidate's weight is 0."""
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", plan_limit=1000, used=1015)], [])
        pool._free[0].check_exhaustion()
        assert pool.pick() is None


class TestFetchUsage:
    @respx.mock
    async def test_returns_parsed_json_on_success(self):
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher"}}),
        )
        async with httpx.AsyncClient() as http:
            result = await fetch_usage(http, "k1")
        assert result["account"]["current_plan"] == "Researcher"

    @respx.mock
    async def test_returns_none_on_failure_no_crash(self):
        respx.get("https://api.tavily.com/usage").mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient() as http:
            result = await fetch_usage(http, "k1")
        assert result is None


class TestBuildKeyPool:
    @respx.mock
    async def test_seeds_plan_limit_and_used_from_usage(self):
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 250}}),
        )
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, ["k1"])
        assert pool.n_free == 1
        assert pool._free[0].plan_limit == 1000
        assert pool._free[0].used == 250

    @respx.mock
    async def test_key_already_over_limit_at_boot_is_proactively_exhausted(self):
        """Real case caught live this session: a key seeded at 1015/1000."""
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 1015}}),
        )
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, ["k1"])
        assert pool._free[0].exhausted is True
        assert pool.pick() is None

    @respx.mock
    async def test_usage_unreachable_defaults_to_free_tier_no_crash(self):
        respx.get("https://api.tavily.com/usage").mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, ["k1"])
        assert pool.n_free == 1
        assert pool._free[0].plan_limit == 1000
        assert pool._free[0].used == 0

    @respx.mock
    async def test_tier_auto_detected_as_paygo_for_non_free_plan(self):
        """No manual .env declaration needed — a key whose /usage reports
        anything other than the free plan name is automatically routed
        into the paygo pool."""
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Pay as you go"}}),
        )
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, ["k1"])
        assert pool.n_free == 0
        assert pool.n_paygo == 1
        assert pool._paygo[0].tier == "paygo"

    @respx.mock
    async def test_tier_auto_detected_as_free_for_researcher_plan(self):
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 0}}),
        )
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, ["k1"])
        assert pool.n_free == 1
        assert pool.n_paygo == 0

    @respx.mock
    async def test_mixed_pool_splits_by_live_detected_tier(self):
        respx.get("https://api.tavily.com/usage").mock(
            side_effect=[
                httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 0}}),
                httpx.Response(200, json={"account": {"current_plan": "Pay as you go"}}),
            ],
        )
        async with httpx.AsyncClient() as http:
            pool = await build_key_pool(http, ["free-key", "paid-key"])
        assert pool.n_free == 1
        assert pool.n_paygo == 1


class TestExhaustedStatesAndReset:
    def test_exhausted_states_returns_only_exhausted_across_both_tiers(self):
        pool = TavilyKeyPool(
            [KeyState(key="f1", tier="free"), KeyState(key="f2", tier="free", exhausted=True)],
            [KeyState(key="p1", tier="paygo", exhausted=True)],
        )
        keys = {s.key for s in pool.exhausted_states()}
        assert keys == {"f2", "p1"}

    def test_reset_key_unexhausts_and_reseeds_counters(self):
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", exhausted=True, used=1000)], [])
        pool.reset_key("f1", plan_limit=1000, used=0)
        state = pool._free[0]
        assert state.exhausted is False
        assert state.used == 0
        assert state.plan_limit == 1000


class TestRecheckExhaustedKeys:
    @respx.mock
    async def test_recovers_key_with_real_headroom(self):
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", exhausted=True, used=1000)], [])
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 0}}),
        )
        async with httpx.AsyncClient() as http:
            await recheck_exhausted_keys(http, pool)
        assert pool._free[0].exhausted is False
        assert pool._free[0].used == 0

    @respx.mock
    async def test_leaves_genuinely_still_exhausted_key_alone(self):
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", exhausted=True, used=1000)], [])
        respx.get("https://api.tavily.com/usage").mock(
            return_value=httpx.Response(200, json={"account": {"current_plan": "Researcher", "plan_limit": 1000, "plan_usage": 1000}}),
        )
        async with httpx.AsyncClient() as http:
            await recheck_exhausted_keys(http, pool)
        assert pool._free[0].exhausted is True

    @respx.mock
    async def test_usage_unreachable_leaves_key_exhausted_no_crash(self):
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", exhausted=True, used=1000)], [])
        respx.get("https://api.tavily.com/usage").mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient() as http:
            await recheck_exhausted_keys(http, pool)
        assert pool._free[0].exhausted is True

    @respx.mock
    async def test_skips_keys_that_are_not_exhausted(self):
        """Only exhausted keys get rechecked — a live key isn't touched."""
        pool = TavilyKeyPool([KeyState(key="f1", tier="free", exhausted=False)], [])
        route = respx.get("https://api.tavily.com/usage")
        async with httpx.AsyncClient() as http:
            await recheck_exhausted_keys(http, pool)
        assert route.call_count == 0
