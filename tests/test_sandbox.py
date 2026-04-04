"""Tests for the Deterministic Sandbox (Phase 1 — Task 2).

Verification scenarios:
1. Normal code execution returns structured result (exit_code, stdout, stderr).
2. Code with errors returns non-zero exit_code with stderr captured.
3. Destructive commands (rm -rf) are contained — host protected, error returned.
4. Fork bomb contained — isolated in container, not host.
5. Timeout enforced on infinite loops — execution halted, timed_out=True.
6. SandboxConfig values correctly propagated to the runtime.
7. Network disable/enable flag forwarded correctly.
8. Infrastructure failure raises SandboxExecutionError (not silently swallowed).
9. Each execution gets a fresh container — no state leaking between runs.
10. SandboxResult is immutable.
"""

from __future__ import annotations

import pytest

from kappa.config import SandboxConfig
from kappa.exceptions import SandboxExecutionError
from kappa.sandbox.executor import (
    ContainerRuntime,
    SandboxExecutor,
    SandboxResult,
)


# ── Fake runtime for testing (no Docker required) ───────────────


class FakeRuntime:
    """Deterministic mock container runtime for unit testing."""

    def __init__(
        self,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        raise_error: Exception | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.raise_error = raise_error
        self.calls: list[dict] = []

    def run(
        self,
        *,
        image: str,
        command: list[str],
        mem_limit: str,
        network_disabled: bool,
        timeout: int,
        volumes: dict | None = None,
    ) -> SandboxResult:
        self.calls.append(
            {
                "image": image,
                "command": command,
                "mem_limit": mem_limit,
                "network_disabled": network_disabled,
                "timeout": timeout,
                "volumes": volumes,
            }
        )
        if self.raise_error:
            raise self.raise_error
        return SandboxResult(
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            timed_out=self.timed_out,
        )


# ── SandboxResult unit tests ───────────────────────────────────


class TestSandboxResult:

    def test_result_is_immutable(self):
        result = SandboxResult(exit_code=0, stdout="ok", stderr="")
        with pytest.raises(AttributeError):
            result.exit_code = 1  # type: ignore[misc]

    def test_result_defaults(self):
        result = SandboxResult(exit_code=0, stdout="", stderr="")
        assert result.timed_out is False


# ── SandboxExecutor unit tests ──────────────────────────────────


class TestSandboxExecutor:

    def test_successful_execution_returns_structured_result(self):
        runtime = FakeRuntime(exit_code=0, stdout="hello world\n", stderr="")
        executor = SandboxExecutor(runtime=runtime)

        result = executor.execute("print('hello world')")

        assert result.exit_code == 0
        assert result.stdout == "hello world\n"
        assert result.stderr == ""
        assert result.timed_out is False

    def test_code_error_returns_nonzero_exit_code_with_stderr(self):
        runtime = FakeRuntime(
            exit_code=1,
            stdout="",
            stderr="Traceback (most recent call last):\n  NameError: name 'x' is not defined\n",
        )
        executor = SandboxExecutor(runtime=runtime)

        result = executor.execute("print(x)")

        assert result.exit_code == 1
        assert "NameError" in result.stderr

    def test_destructive_command_contained_in_sandbox(self):
        """Destructive system commands (rm -rf /) run inside the container, NOT on host.

        Verifies:
        - Code is passed to the container runtime (never subprocess on host)
        - Runtime receives the exact code string
        - Structured error result returned from container
        """
        runtime = FakeRuntime(
            exit_code=1,
            stdout="",
            stderr="rm: cannot remove '/': Permission denied\n",
        )
        executor = SandboxExecutor(runtime=runtime)

        destructive_code = "import os; os.system('rm -rf /')"
        result = executor.execute(destructive_code)

        # Code was sent to the container, not executed on host
        assert len(runtime.calls) == 1
        assert runtime.calls[0]["command"] == ["python", "-c", destructive_code]
        assert result.exit_code == 1
        assert "Permission denied" in result.stderr

    def test_fork_bomb_contained_in_sandbox(self):
        """Fork bomb code is isolated in the container, not the host."""
        runtime = FakeRuntime(exit_code=137, stdout="", stderr="Killed\n")
        executor = SandboxExecutor(runtime=runtime)

        fork_bomb = "import os\nwhile True: os.fork()"
        result = executor.execute(fork_bomb)

        assert len(runtime.calls) == 1
        assert runtime.calls[0]["command"] == ["python", "-c", fork_bomb]
        assert result.exit_code == 137

    def test_timeout_on_infinite_loop(self):
        """Infinite loop must be halted by timeout, returning timed_out=True."""
        runtime = FakeRuntime(
            exit_code=-1,
            stdout="",
            stderr="Execution timed out after 30s.\n",
            timed_out=True,
        )
        executor = SandboxExecutor(runtime=runtime)

        result = executor.execute("while True: pass")

        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()

    def test_config_propagated_to_runtime(self):
        """SandboxConfig values must be correctly forwarded to the runtime."""
        config = SandboxConfig(
            timeout_seconds=10,
            memory_limit_mb=128,
            network_enabled=False,
            docker_image="python:3.12-slim",
        )
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        executor.execute("pass")

        call = runtime.calls[0]
        assert call["image"] == "python:3.12-slim"
        assert call["mem_limit"] == "128m"
        assert call["network_disabled"] is True
        assert call["timeout"] == 10

    def test_network_enabled_propagated(self):
        """network_enabled=True in config → network_disabled=False to runtime."""
        config = SandboxConfig(network_enabled=True)
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        executor.execute("import urllib.request")

        assert runtime.calls[0]["network_disabled"] is False

    def test_infrastructure_failure_raises_sandbox_error(self):
        """Container runtime failure must surface as SandboxExecutionError."""
        runtime = FakeRuntime(
            raise_error=SandboxExecutionError("Docker daemon not available")
        )
        executor = SandboxExecutor(runtime=runtime)

        with pytest.raises(SandboxExecutionError, match="Docker daemon"):
            executor.execute("print('hi')")

    def test_multiple_executions_each_get_fresh_container(self):
        """Each execute() call creates a new container — no state leaking."""
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime)

        executor.execute("x = 1")
        executor.execute("print(x)")  # would fail in real container (x undefined)
        executor.execute("import sys")

        assert len(runtime.calls) == 3
        for call in runtime.calls:
            assert call["image"] == "python:3.11-slim"

    def test_default_config_applied_when_none_provided(self):
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime)

        assert executor.config.timeout_seconds == 30
        assert executor.config.memory_limit_mb == 256
        assert executor.config.network_enabled is False
        assert executor.config.docker_image == "python:3.11-slim"

    def test_workspace_dir_creates_volume_mount(self, tmp_path):
        """With workspace_dir set, volumes dict is passed to runtime."""
        ws = str(tmp_path / "ws")
        config = SandboxConfig(workspace_dir=ws)
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        executor.execute("print('hi')")

        call = runtime.calls[0]
        assert call["volumes"] is not None
        # Resolved absolute path should be a key
        resolved = str(tmp_path / "ws")
        assert any(resolved in k for k in call["volumes"])
        mount = list(call["volumes"].values())[0]
        assert mount["bind"] == "/workspace"
        assert mount["mode"] == "rw"

    def test_no_workspace_dir_no_volumes(self):
        """Explicit workspace_dir=None produces volumes=None."""
        config = SandboxConfig(workspace_dir=None)
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        executor.execute("pass")

        assert runtime.calls[0]["volumes"] is None

    def test_default_workspace_dir_is_cwd(self):
        """Default config uses CWD as workspace_dir."""
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime)

        executor.execute("pass")

        call = runtime.calls[0]
        assert call["volumes"] is not None

    def test_workspace_dir_creates_directory(self, tmp_path):
        """workspace_dir is created if it doesn't exist."""
        ws = tmp_path / "new_workspace"
        config = SandboxConfig(workspace_dir=str(ws))
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        executor.execute("pass")

        assert ws.exists()

    def test_container_workspace_path_customizable(self, tmp_path):
        """Custom container_workspace_path appears in volumes."""
        ws = str(tmp_path / "ws")
        config = SandboxConfig(workspace_dir=ws, container_workspace_path="/data")
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        executor.execute("pass")

        mount = list(runtime.calls[0]["volumes"].values())[0]
        assert mount["bind"] == "/data"

    def test_filesystem_root_mount_rejected(self):
        """Mounting / as workspace raises SandboxExecutionError."""
        config = SandboxConfig(workspace_dir="/")
        runtime = FakeRuntime(exit_code=0, stdout="", stderr="")
        executor = SandboxExecutor(runtime=runtime, config=config)

        with pytest.raises(SandboxExecutionError, match="filesystem root"):
            executor.execute("pass")


# ── Protocol conformance ────────────────────────────────────────


class TestContainerRuntimeProtocol:

    def test_fake_runtime_satisfies_protocol(self):
        assert isinstance(FakeRuntime(), ContainerRuntime)
