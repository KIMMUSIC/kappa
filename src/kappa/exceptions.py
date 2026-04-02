"""Custom exceptions for the Kappa harness."""

from __future__ import annotations


class KappaError(Exception):
    """Base exception for all Kappa errors."""


class BudgetExceededException(KappaError):
    """Raised when a session exceeds its allocated token or cost budget.

    This exception triggers immediate process shutdown — no further
    LLM API calls are permitted once raised.
    """

    def __init__(self, message: str, *, tokens_used: int = 0, cost_used: float = 0.0):
        self.tokens_used = tokens_used
        self.cost_used = cost_used
        super().__init__(message)


class ParsingError(KappaError):
    """Raised when LLM output violates the expected XML anchor format."""


class SandboxExecutionError(KappaError):
    """Raised when the sandbox environment itself fails (not the user code)."""


class ToolExecutionError(KappaError):
    """Raised when tool infrastructure fails (not a user-facing tool error)."""


class SemanticLoopException(KappaError):
    """Raised when the semantic loop detector identifies repetitive behaviour.

    This triggers early termination before ``max_attempts`` is reached,
    saving budget by preventing the agent from repeating the same mistake.
    """

    def __init__(self, message: str, *, similarity: float = 0.0):
        self.similarity = similarity
        super().__init__(message)
