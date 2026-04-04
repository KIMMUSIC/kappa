"""Host-based code execution with path safety and HITL approval.

Replaces the Docker sandbox with direct subprocess execution on the
host OS.  Safety is enforced by:
1. AST-based path validation (sandbox.safety)
2. Optional human-in-the-loop approval callback
3. Timeout enforcement via subprocess
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from kappa.config import ExecutionConfig
from kappa.exceptions import ExecutionError
from kappa.sandbox.safety import SafetyValidator


@dataclass(frozen=True)
class SandboxResult:
    """Structured output from code execution.

    Name kept for backward compatibility with downstream consumers
    (orchestrator, reviewer, finalizer, tests).
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# ── HITL approval protocol ───────────────────────────────────────

@runtime_checkable
class CodeApprovalFn(Protocol):
    """Callback asked before executing code on the host.

    Args:
        code: The Python source about to be executed.
        violations: Safety violations found by the AST validator.

    Returns:
        True to proceed, False to block execution.
    """

    def __call__(self, code: str, violations: list[str]) -> bool: ...


def auto_approve(code: str, violations: list[str]) -> bool:
    """Approve automatically when there are no safety violations."""
    return len(violations) == 0


def always_approve(code: str, violations: list[str]) -> bool:
    """Approve all code unconditionally (use with caution)."""
    return True


# ── Host executor ────────────────────────────────────────────────

class HostExecutor:
    """Executes Python code directly on the host via subprocess.

    Safety guarantees:
    - AST scanning blocks dangerous patterns before execution
    - HITL callback gates every execution attempt
    - Timeout enforcement via subprocess.run
    - Working directory set to output_dir (files land there by default)
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        approval_fn: CodeApprovalFn | Callable[..., bool] | None = None,
    ) -> None:
        self._config = config or ExecutionConfig()
        self._approval_fn = approval_fn or auto_approve

        # Build allowed directories list for safety validator
        allowed_dirs: list[str | Path] = []
        if self._config.workspace_dir:
            allowed_dirs.append(Path(self._config.workspace_dir).resolve())
        if self._config.output_dir:
            out = Path(self._config.output_dir)
            if not out.is_absolute() and self._config.workspace_dir:
                out = Path(self._config.workspace_dir).resolve() / out
            allowed_dirs.append(out.resolve())
        # If no dirs configured, allow CWD
        if not allowed_dirs:
            allowed_dirs.append(Path.cwd())

        self._validator = SafetyValidator(allowed_dirs)

    @property
    def config(self) -> ExecutionConfig:
        return self._config

    def execute(self, code: str) -> SandboxResult:
        """Execute a Python code string on the host.

        Flow:
            1. AST safety scan
            2. HITL approval (with violation info)
            3. subprocess.run with timeout
            4. Return SandboxResult

        User-code errors are captured as non-zero exit_code, not
        raised as exceptions.
        """
        # 1) Safety scan
        scan = self._validator.validate(code)

        # 2) HITL approval
        if not self._approval_fn(code, scan.violations):
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="Execution blocked: user denied approval.",
                timed_out=False,
            )

        # 3) Prepare output directory
        cwd = self._resolve_output_dir()

        # 4) Execute
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                cwd=str(cwd),
            )
            return SandboxResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {self._config.timeout_seconds}s.",
                timed_out=True,
            )
        except Exception as exc:
            raise ExecutionError(f"Host execution failed: {exc}") from exc

    def _resolve_output_dir(self) -> Path:
        """Resolve and create the output directory."""
        if self._config.output_dir:
            out = Path(self._config.output_dir)
            if not out.is_absolute() and self._config.workspace_dir:
                out = Path(self._config.workspace_dir).resolve() / out
            out = out.resolve()
        elif self._config.workspace_dir:
            out = Path(self._config.workspace_dir).resolve()
        else:
            out = Path.cwd()
        out.mkdir(parents=True, exist_ok=True)
        return out


# Backward-compatible alias
SandboxExecutor = HostExecutor
