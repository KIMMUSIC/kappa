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
class ExecutionConfig:
    """Host execution constraints."""

    timeout_seconds: int = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "300"))
    workspace_dir: str | None = os.getenv("SANDBOX_WORKSPACE_DIR", ".")
    output_dir: str | None = os.getenv("SANDBOX_OUTPUT_DIR", "output")
    auto_approve: bool = os.getenv("EXECUTION_AUTO_APPROVE", "").lower() in ("1", "true", "yes")


# Backward-compatible alias
SandboxConfig = ExecutionConfig


@dataclass(frozen=True)
class MemoryConfig:
    """VFS-based long-term memory configuration."""

    workspace_root: str = os.getenv("MEMORY_WORKSPACE_ROOT", ".kappa_workspace")
    auto_inject_files: tuple[str, ...] = ("LEARNINGS.md",)


@dataclass(frozen=True)
class SemanticConfig:
    """Semantic loop detection thresholds."""

    window_size: int = 5
    similarity_threshold: float = 0.85
    min_samples: int = 3


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""

    model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    max_self_heal_retries: int = int(os.getenv("MAX_SELF_HEAL_RETRIES", "3"))
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)


# ── Phase 3 additions ───────────────────────────────────────────


@dataclass(frozen=True)
class BackoffConfig:
    """Decorrelated Jitter exponential backoff thresholds."""

    base_delay: float = float(os.getenv("BACKOFF_BASE_DELAY", "1.0"))
    max_delay: float = float(os.getenv("BACKOFF_MAX_DELAY", "60.0"))
    max_retries: int = int(os.getenv("BACKOFF_MAX_RETRIES", "5"))


@dataclass(frozen=True)
class SessionLaneConfig:
    """Per-key serialisation lane thresholds."""

    timeout: float = float(os.getenv("SESSION_LANE_TIMEOUT", "30.0"))


@dataclass(frozen=True)
class OrchestratorConfig:
    """Orchestrator Super-Graph configuration (Task 2)."""

    max_retries_per_task: int = 3
    max_plan_retries: int = 2
    max_subtasks: int = 10
    max_parallel_workers: int = 3
    planner_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    reviewer_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")


@dataclass(frozen=True)
class TelemetryConfig:
    """Agent-RRM telemetry configuration (Task 3)."""

    enabled: bool = True
    log_path: str = os.getenv(
        "TELEMETRY_LOG_PATH", ".kappa_telemetry/trajectories.jsonl"
    )


# ── Phase 4 additions ───────────────────────────────────────────


@dataclass(frozen=True)
class MCPConfig:
    """MCP bridge client configuration."""

    request_timeout: float = float(os.getenv("MCP_REQUEST_TIMEOUT", "30.0"))
    tool_name_prefix: str = "mcp"
    max_retries: int = int(os.getenv("MCP_MAX_RETRIES", "2"))


@dataclass(frozen=True)
class RAGConfig:
    """RAG pipeline configuration."""

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.0"))


# ── Phase 6 additions ───────────────────────────────────────────


@dataclass(frozen=True)
class MetaPromptConfig:
    """Meta-prompting pipeline configuration."""

    ambiguity_threshold: float = float(os.getenv("META_AMBIGUITY_THRESHOLD", "0.4"))
    max_interview_questions: int = int(os.getenv("META_MAX_INTERVIEW_QUESTIONS", "5"))
    skip_interview: bool = False
    skip_plan_approval: bool = False
