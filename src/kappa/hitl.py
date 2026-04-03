"""Human-in-the-loop (HITL) interceptor and policy engine.

Provides a callback-based approval mechanism that integrates with
``OrchestratorGraph`` without modifying the core graph topology.

Architecture::

    OrchestratorGraph._dispatcher_node()
        │
        └──  approval_callback(task, context)  ──→  "approve" | "deny"
                      │
             ┌────────▼────────┐
             │  HITLInterceptor │
             │                  │
             │  policy.check()  │──→ auto-approve if no triggers
             │  prompt_fn()     │──→ ask human if triggered
             └──────────────────┘

The interceptor never raises — it always returns a decision string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ── Policy ─────────────────────────────────────────────────────


# Dangerous keywords in subtask goals
_DESTRUCTIVE_PATTERNS: tuple[str, ...] = (
    r"\brm\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\bkill\b",
    r"\bremove\b",
    r"\btruncate\b",
    r"\bformat\b",
    r"\bdestroy\b",
    r"\bshutdown\b",
    r"\breset\b",
)

_DESTRUCTIVE_RE = re.compile(
    "|".join(_DESTRUCTIVE_PATTERNS), re.IGNORECASE
)


@dataclass(frozen=True)
class HITLTrigger:
    """Describes why HITL approval is required."""

    reason: str
    severity: str = "warning"  # "warning" | "critical"


@dataclass(frozen=True)
class HITLPolicy:
    """Configurable policy that decides when human approval is required.

    Attributes:
        budget_threshold: Fraction of budget remaining below which
            approval is required (0.2 = triggers at 80% consumed).
        max_auto_attempts: Auto-approve up to this many attempts;
            beyond this, require human sign-off.
        approve_all: If True, skip all checks (auto-approve everything).
        destructive_patterns: Compiled regex for dangerous keywords.
    """

    budget_threshold: float = 0.2
    max_auto_attempts: int = 2
    approve_all: bool = False
    destructive_patterns: re.Pattern[str] = field(
        default=_DESTRUCTIVE_RE, repr=False
    )

    def check(
        self,
        task: dict[str, Any],
        budget_remaining_ratio: float = 1.0,
    ) -> list[HITLTrigger]:
        """Evaluate a subtask against all policy rules.

        Args:
            task: SubTask dict with ``goal``, ``attempts``, etc.
            budget_remaining_ratio: Value between 0.0 and 1.0 representing
                the fraction of budget still available.

        Returns:
            List of triggered ``HITLTrigger`` objects.
            Empty list means auto-approve.
        """
        if self.approve_all:
            return []

        triggers: list[HITLTrigger] = []

        # Rule 1: Budget warning
        if budget_remaining_ratio <= self.budget_threshold:
            triggers.append(HITLTrigger(
                reason=f"Budget low: {budget_remaining_ratio:.0%} remaining",
                severity="critical",
            ))

        # Rule 2: Destructive command detection
        goal = task.get("goal", "")
        if self.destructive_patterns.search(goal):
            triggers.append(HITLTrigger(
                reason=f"Potentially destructive operation detected in goal",
                severity="critical",
            ))

        # Rule 3: Excessive retries
        attempts = task.get("attempts", 0)
        if attempts >= self.max_auto_attempts:
            triggers.append(HITLTrigger(
                reason=f"Task has been attempted {attempts} times",
                severity="warning",
            ))

        return triggers


# ── Prompt Function Protocol ───────────────────────────────────


@runtime_checkable
class ApprovalPrompt(Protocol):
    """Protocol for functions that ask the human for approval."""

    def __call__(
        self,
        task: dict[str, Any],
        triggers: list[HITLTrigger],
    ) -> str:
        """Return ``"approve"`` or ``"deny"``."""
        ...


def auto_approve(
    task: dict[str, Any],
    triggers: list[HITLTrigger],
) -> str:
    """Always approve — useful for testing and non-interactive mode."""
    return "approve"


def auto_deny(
    task: dict[str, Any],
    triggers: list[HITLTrigger],
) -> str:
    """Always deny — useful for testing."""
    return "deny"


# ── Interceptor ────────────────────────────────────────────────


class HITLInterceptor:
    """Orchestrator-compatible approval callback.

    Combines a ``HITLPolicy`` (decides *when* to ask) with an
    ``ApprovalPrompt`` (decides *how* to ask).

    Usage::

        interceptor = HITLInterceptor(policy=policy, prompt_fn=my_prompt)
        orchestrator = OrchestratorGraph(..., approval_callback=interceptor)

    The interceptor is callable: ``interceptor(task, context) -> str``.

    Args:
        policy: Determines which tasks need approval.
        prompt_fn: Called when approval is needed.  Receives the task
            and list of triggers.  Must return ``"approve"`` or ``"deny"``.
        budget_ratio_fn: Optional callable that returns current budget
            remaining ratio (0.0–1.0).  If not provided, budget checks
            are skipped.
    """

    def __init__(
        self,
        policy: HITLPolicy | None = None,
        prompt_fn: Callable[..., str] | None = None,
        budget_ratio_fn: Callable[[], float] | None = None,
    ) -> None:
        self._policy = policy or HITLPolicy()
        self._prompt_fn = prompt_fn or auto_approve
        self._budget_ratio_fn = budget_ratio_fn
        self._decisions: list[dict[str, Any]] = []

    @property
    def policy(self) -> HITLPolicy:
        return self._policy

    @property
    def decisions(self) -> list[dict[str, Any]]:
        """Audit log of all decisions made."""
        return list(self._decisions)

    def __call__(
        self,
        task: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Evaluate a task and return ``"approve"`` or ``"deny"``.

        This is the callback signature expected by ``OrchestratorGraph``.

        Args:
            task: SubTask dict.
            context: Optional extra context (budget state, etc.).

        Returns:
            ``"approve"`` or ``"deny"``.
        """
        budget_ratio = 1.0
        if self._budget_ratio_fn:
            budget_ratio = self._budget_ratio_fn()

        triggers = self._policy.check(task, budget_remaining_ratio=budget_ratio)

        if not triggers:
            decision = "approve"
        else:
            decision = self._prompt_fn(task, triggers)

        self._decisions.append({
            "task_id": task.get("id", "unknown"),
            "triggers": [t.reason for t in triggers],
            "decision": decision,
        })

        return decision
