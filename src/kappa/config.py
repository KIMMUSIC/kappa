"""Global configuration for the Kappa harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BudgetConfig:
    """Budget gate thresholds. Immutable once created."""

    max_total_tokens: int = int(os.getenv("BUDGET_MAX_TOKENS", "100000"))
    max_cost_usd: float = float(os.getenv("BUDGET_MAX_COST_USD", "5.00"))

    # Anthropic Claude 3.5 Sonnet pricing (per 1M tokens)
    input_cost_per_million: float = 3.00
    output_cost_per_million: float = 15.00


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox execution constraints."""

    timeout_seconds: int = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30"))
    memory_limit_mb: int = int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", "256"))
    network_enabled: bool = False
    docker_image: str = "python:3.11-slim"


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""

    model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    max_self_heal_retries: int = int(os.getenv("MAX_SELF_HEAL_RETRIES", "3"))
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
