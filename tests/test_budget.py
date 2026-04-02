"""Tests for the Budget Gate & Circuit Breaker (Phase 1 — Task 1).

Verification scenarios:
1. Normal usage within budget succeeds.
2. Token limit breach triggers BudgetExceededException.
3. Cost limit breach triggers BudgetExceededException.
4. Circuit breaker trips permanently — all subsequent calls are blocked.
5. Pre-check blocks calls when budget is already exhausted.
6. BudgetGate integrates tracker + provider correctly.
7. Simulated infinite-loop scenario: rapid repeated calls hit the wall.
"""

from __future__ import annotations

import pytest

from kappa.config import BudgetConfig
from kappa.exceptions import BudgetExceededException
from kappa.budget.tracker import BudgetTracker
from kappa.budget.gate import BudgetGate, LLMResponse, LLMProvider


# ── Fake provider for testing (no real API calls) ─────────────────

class FakeProvider:
    """Deterministic mock that returns fixed token counts."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 200) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_count = 0

    def call(self, *, messages: list[dict], model: str, max_tokens: int) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content=f"fake response #{self.call_count}",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            model=model,
            stop_reason="end_turn",
        )


# ── BudgetTracker unit tests ─────────────────────────────────────

class TestBudgetTracker:

    def _make_tracker(self, max_tokens: int = 1000, max_cost: float = 10.0) -> BudgetTracker:
        return BudgetTracker(BudgetConfig(
            max_total_tokens=max_tokens,
            max_cost_usd=max_cost,
        ))

    def test_normal_usage_within_budget(self):
        tracker = self._make_tracker(max_tokens=1000)
        tracker.record_usage(prompt_tokens=100, completion_tokens=200)

        assert tracker.total_tokens == 300
        assert tracker.call_count == 1
        assert not tracker.is_exceeded
        assert tracker.remaining_tokens == 700

    def test_token_limit_breach_raises(self):
        tracker = self._make_tracker(max_tokens=500)
        # First call: 300 tokens — fine
        tracker.record_usage(prompt_tokens=100, completion_tokens=200)

        # Second call: pushes total to 600 — exceeds 500 limit
        with pytest.raises(BudgetExceededException) as exc_info:
            tracker.record_usage(prompt_tokens=100, completion_tokens=200)

        assert exc_info.value.tokens_used == 600
        assert tracker.is_tripped

    def test_cost_limit_breach_raises(self):
        # Set a very low cost limit but high token limit
        tracker = self._make_tracker(max_tokens=10_000_000, max_cost=0.001)

        # Even a small call will exceed $0.001 with enough output tokens
        with pytest.raises(BudgetExceededException):
            tracker.record_usage(prompt_tokens=1000, completion_tokens=1000)

    def test_circuit_breaker_trips_permanently(self):
        tracker = self._make_tracker(max_tokens=100)

        # Trip the breaker
        with pytest.raises(BudgetExceededException):
            tracker.record_usage(prompt_tokens=50, completion_tokens=60)

        # Subsequent calls are blocked even with zero tokens
        with pytest.raises(BudgetExceededException, match="already tripped"):
            tracker.record_usage(prompt_tokens=0, completion_tokens=0)

    def test_pre_check_blocks_exhausted_budget(self):
        tracker = self._make_tracker(max_tokens=100)

        # Exhaust the budget
        with pytest.raises(BudgetExceededException):
            tracker.record_usage(prompt_tokens=50, completion_tokens=60)

        # Pre-check should also block
        with pytest.raises(BudgetExceededException, match="already exhausted"):
            tracker.pre_check()


# ── BudgetGate integration tests ─────────────────────────────────

class TestBudgetGate:

    def test_gate_allows_call_within_budget(self):
        provider = FakeProvider(prompt_tokens=50, completion_tokens=50)
        config = BudgetConfig(max_total_tokens=1000, max_cost_usd=10.0)
        gate = BudgetGate(provider=provider, budget_config=config)

        response = gate.call(messages=[{"role": "user", "content": "hello"}])

        assert response.content == "fake response #1"
        assert gate.tracker.total_tokens == 100
        assert provider.call_count == 1

    def test_gate_blocks_after_budget_exceeded(self):
        provider = FakeProvider(prompt_tokens=200, completion_tokens=200)
        config = BudgetConfig(max_total_tokens=500, max_cost_usd=100.0)
        gate = BudgetGate(provider=provider, budget_config=config)

        # First call: 400 tokens — fine
        gate.call(messages=[{"role": "user", "content": "hello"}])

        # Second call: would push to 800 — exceeds limit, trips breaker
        with pytest.raises(BudgetExceededException):
            gate.call(messages=[{"role": "user", "content": "hello again"}])

        # Third call: pre-check blocks immediately (provider never called)
        with pytest.raises(BudgetExceededException):
            gate.call(messages=[{"role": "user", "content": "blocked"}])

        # Provider was called only twice (second call went through but recording tripped)
        assert provider.call_count == 2

    def test_simulated_infinite_loop_hits_budget_wall(self):
        """Simulates an agent stuck in a loop making repeated API calls.
        The budget gate MUST forcefully stop execution."""
        provider = FakeProvider(prompt_tokens=100, completion_tokens=100)
        config = BudgetConfig(max_total_tokens=1000, max_cost_usd=100.0)
        gate = BudgetGate(provider=provider, budget_config=config)

        calls_made = 0
        with pytest.raises(BudgetExceededException):
            for _ in range(1000):  # simulate runaway loop
                gate.call(messages=[{"role": "user", "content": "loop"}])
                calls_made += 1

        # 5th call (1000 tokens) trips the breaker inside record_usage,
        # so calls_made stays at 4 (increment never reached), but the
        # provider was physically called 5 times.
        assert calls_made == 4
        assert provider.call_count == 5
        assert gate.tracker.is_tripped
        assert gate.tracker.total_tokens == 1000

        # Verify the wall holds — no more calls possible
        with pytest.raises(BudgetExceededException):
            gate.call(messages=[{"role": "user", "content": "one more"}])
        # Provider must NOT have been called again (pre-check blocks it)
        assert provider.call_count == 5


# ── Provider protocol check ──────────────────────────────────────

class TestProviderProtocol:

    def test_fake_provider_satisfies_protocol(self):
        assert isinstance(FakeProvider(), LLMProvider)
