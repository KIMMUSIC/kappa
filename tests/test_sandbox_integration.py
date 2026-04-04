"""Integration tests for the Host Executor (replaces Docker sandbox).

These tests run real code via subprocess on the host.  No Docker required.

Verification scenarios:
1. Normal code → exit_code=0, stdout captured
2. Syntax error → exit_code != 0, stderr has SyntaxError
3. Runtime error → exit_code=1, stderr has traceback
4. Timeout → timed_out=True, exit_code=-1
5. stdout/stderr separated correctly
6. Each execution is isolated (no shared state between calls)
"""

from __future__ import annotations

import pytest

from kappa.config import ExecutionConfig
from kappa.sandbox.executor import HostExecutor, SandboxResult, always_approve


@pytest.fixture
def executor(tmp_path) -> HostExecutor:
    config = ExecutionConfig(
        timeout_seconds=15,
        workspace_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
    )
    return HostExecutor(config=config, approval_fn=always_approve)


@pytest.fixture
def fast_executor(tmp_path) -> HostExecutor:
    """Short timeout for infinite-loop tests."""
    config = ExecutionConfig(
        timeout_seconds=3,
        workspace_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
    )
    return HostExecutor(config=config, approval_fn=always_approve)


class TestHostExecutor:

    def test_normal_execution(self, executor: HostExecutor):
        """Normal code → exit_code=0, stdout captured."""
        result = executor.execute("print('hello from sandbox')")

        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout
        assert result.timed_out is False

    def test_syntax_error_returns_nonzero(self, executor: HostExecutor):
        """Syntax error → exit_code != 0, stderr has SyntaxError."""
        result = executor.execute("def f(:")

        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    def test_runtime_error_captured(self, executor: HostExecutor):
        """Runtime error → exit_code=1, stderr has traceback."""
        result = executor.execute("print(undefined_variable)")

        assert result.exit_code == 1
        assert "NameError" in result.stderr

    def test_infinite_loop_timeout(self, fast_executor: HostExecutor):
        """Infinite loop → timeout enforced."""
        result = fast_executor.execute("while True: pass")

        assert result.timed_out is True
        assert result.exit_code == -1

    def test_stdout_stderr_separated(self, executor: HostExecutor):
        """stdout and stderr are captured separately."""
        result = executor.execute(
            "import sys\n"
            "print('out-msg')\n"
            "print('err-msg', file=sys.stderr)"
        )

        assert "out-msg" in result.stdout
        assert "err-msg" in result.stderr

    def test_each_execution_is_independent(self, executor: HostExecutor):
        """Variables from one execute() call don't leak into the next."""
        # First call defines a variable
        executor.execute("x = 42")

        # Second call: x should not be defined
        result = executor.execute(
            "try:\n"
            "    print(x)\n"
            "    exit(1)\n"
            "except NameError:\n"
            "    print('ISOLATED')"
        )

        assert "ISOLATED" in result.stdout
        assert result.exit_code == 0

    def test_returns_sandbox_result_type(self, executor: HostExecutor):
        """execute() always returns a SandboxResult instance."""
        result = executor.execute("pass")
        assert isinstance(result, SandboxResult)

    def test_empty_code_succeeds(self, executor: HostExecutor):
        """Empty string is valid Python → exit_code=0."""
        result = executor.execute("")
        assert result.exit_code == 0

    def test_multiline_code(self, executor: HostExecutor):
        """Multi-line code executes correctly."""
        code = (
            "total = 0\n"
            "for i in range(5):\n"
            "    total += i\n"
            "print(total)"
        )
        result = executor.execute(code)
        assert result.exit_code == 0
        assert "10" in result.stdout
