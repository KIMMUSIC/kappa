"""BudgetGate: LLM API call wrapper with circuit-breaker enforcement.

Provides a provider-agnostic interface for calling LLMs. The Anthropic
implementation is the default; swapping to another provider requires only
implementing a new `LLMProvider` protocol.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kappa.budget.tracker import BudgetTracker
from kappa.config import BudgetConfig


# ── Provider abstraction ──────────────────────────────────────────

@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    stop_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that any LLM backend must satisfy."""

    def call(self, *, messages: list[dict], model: str, max_tokens: int) -> LLMResponse: ...


# ── Anthropic provider ────────────────────────────────────────────

class AnthropicProvider:
    """Concrete LLM provider backed by the Anthropic SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(self, *, messages: list[dict], model: str, max_tokens: int) -> LLMResponse:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return LLMResponse(
            content=response.content[0].text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
        )


# ── Budget Gate ───────────────────────────────────────────────────

class BudgetGate:
    """Wraps an LLM provider with pre/post budget enforcement.

    Flow:
        1. pre_check()  — block if budget already exhausted
        2. provider.call() — actual API call
        3. record_usage() — update tracker; trip breaker if over limit
    """

    def __init__(
        self,
        provider: LLMProvider,
        tracker: BudgetTracker | None = None,
        budget_config: BudgetConfig | None = None,
    ) -> None:
        self._provider = provider
        self._tracker = tracker or BudgetTracker(budget_config or BudgetConfig())

    @property
    def tracker(self) -> BudgetTracker:
        return self._tracker

    def call(
        self,
        *,
        messages: list[dict],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 16384,
    ) -> LLMResponse:
        """Execute a budget-guarded LLM call.

        Raises:
            BudgetExceededException: Before the call if budget is
                already exhausted, or after the call if the response
                pushed cumulative usage over the limit.
        """
        # 1) Pre-flight check
        self._tracker.pre_check()

        # 2) Actual API call
        response = self._provider.call(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )

        # 3) Record usage (may raise BudgetExceededException)
        self._tracker.record_usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

        return response
