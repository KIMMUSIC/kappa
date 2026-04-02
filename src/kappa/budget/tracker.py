"""Session-level budget tracker with thread-safe token/cost accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from kappa.config import BudgetConfig
from kappa.exceptions import BudgetExceededException


@dataclass
class UsageRecord:
    """Single API call usage snapshot."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetTracker:
    """Tracks cumulative token usage and estimated cost for a session.

    Thread-safe: all mutations are guarded by a reentrant lock so that
    concurrent API calls within a single session cannot race past the
    budget limit.
    """

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config = config or BudgetConfig()
        self._lock = threading.RLock()
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._call_count: int = 0
        self._tripped: bool = False  # circuit breaker state

    # ── Read-only properties ──────────────────────────────────────

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        input_cost = (self._total_prompt_tokens / 1_000_000) * self._config.input_cost_per_million
        output_cost = (self._total_completion_tokens / 1_000_000) * self._config.output_cost_per_million
        return input_cost + output_cost

    @property
    def remaining_tokens(self) -> int:
        return max(0, self._config.max_total_tokens - self.total_tokens)

    @property
    def remaining_cost_usd(self) -> float:
        return max(0.0, self._config.max_cost_usd - self.estimated_cost_usd)

    @property
    def is_exceeded(self) -> bool:
        return (
            self._tripped
            or self.total_tokens >= self._config.max_total_tokens
            or self.estimated_cost_usd >= self._config.max_cost_usd
        )

    @property
    def is_tripped(self) -> bool:
        """True if the circuit breaker has been permanently tripped."""
        return self._tripped

    @property
    def call_count(self) -> int:
        return self._call_count

    # ── Mutation ──────────────────────────────────────────────────

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from an API call and enforce budget limits.

        Raises:
            BudgetExceededException: If cumulative usage exceeds any limit
                after recording. The circuit breaker is permanently tripped.
        """
        with self._lock:
            if self._tripped:
                raise BudgetExceededException(
                    "Circuit breaker already tripped — no further calls allowed.",
                    tokens_used=self.total_tokens,
                    cost_used=self.estimated_cost_usd,
                )

            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._call_count += 1

            if self.is_exceeded:
                self._tripped = True
                raise BudgetExceededException(
                    f"Budget exceeded: {self.total_tokens} tokens "
                    f"(limit {self._config.max_total_tokens}), "
                    f"${self.estimated_cost_usd:.4f} "
                    f"(limit ${self._config.max_cost_usd:.2f})",
                    tokens_used=self.total_tokens,
                    cost_used=self.estimated_cost_usd,
                )

    def pre_check(self) -> None:
        """Pre-flight check before initiating an API call.

        Raises:
            BudgetExceededException: If budget is already exhausted or
                the circuit breaker has been tripped.
        """
        with self._lock:
            if self._tripped or self.is_exceeded:
                raise BudgetExceededException(
                    "Budget already exhausted — API call blocked.",
                    tokens_used=self.total_tokens,
                    cost_used=self.estimated_cost_usd,
                )
