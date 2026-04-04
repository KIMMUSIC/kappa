"""Tests for Host Executor and Safety Validator.

Verification scenarios:
1. Normal code execution returns structured result (exit_code, stdout, stderr).
2. Code with errors returns non-zero exit_code with stderr captured.
3. Timeout enforced on long-running code — timed_out=True.
4. SandboxResult is immutable.
5. Safety validator catches dangerous patterns.
6. HITL approval blocks unsafe code.
7. ExecutionConfig values correctly applied.
8. Output directory is created and used as cwd.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kappa.config import ExecutionConfig
from kappa.exceptions import ExecutionError
from kappa.sandbox.executor import (
    HostExecutor,
    SandboxResult,
    always_approve,
    auto_approve,
)
from kappa.sandbox.safety import SafetyValidator


# ── SandboxResult unit tests ───────────────────────────────────


class TestSandboxResult:

    def test_result_is_immutable(self):
        result = SandboxResult(exit_code=0, stdout="ok", stderr="")
        with pytest.raises(AttributeError):
            result.exit_code = 1  # type: ignore[misc]

    def test_result_defaults(self):
        result = SandboxResult(exit_code=0, stdout="", stderr="")
        assert result.timed_out is False


# ── HostExecutor unit tests ────────────────────────────────────


class TestHostExecutor:

    def test_successful_execution(self, tmp_path):
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        result = executor.execute("print('hello world')")

        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.timed_out is False

    def test_code_error_returns_nonzero(self, tmp_path):
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        result = executor.execute("raise ValueError('test error')")

        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    def test_timeout_on_long_running_code(self, tmp_path):
        config = ExecutionConfig(
            timeout_seconds=2,
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        result = executor.execute("import time; time.sleep(10)")

        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()

    def test_approval_denied_blocks_execution(self, tmp_path):
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )

        def deny_all(code, violations):
            return False

        executor = HostExecutor(config=config, approval_fn=deny_all)
        result = executor.execute("print('should not run')")

        assert result.exit_code == -1
        assert "denied" in result.stderr.lower()

    def test_auto_approve_blocks_unsafe_code(self, tmp_path):
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        executor = HostExecutor(config=config, approval_fn=auto_approve)
        result = executor.execute("import subprocess; subprocess.run(['ls'])")

        assert result.exit_code == -1
        assert "denied" in result.stderr.lower()

    def test_auto_approve_allows_safe_code(self, tmp_path):
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        executor = HostExecutor(config=config, approval_fn=auto_approve)
        result = executor.execute("print(1 + 1)")

        assert result.exit_code == 0
        assert "2" in result.stdout

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "new_output"
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(out),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        executor.execute("pass")

        assert out.exists()

    def test_cwd_is_output_dir(self, tmp_path):
        out = tmp_path / "out"
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(out),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        result = executor.execute("import os; print(os.getcwd())")

        assert result.exit_code == 0
        assert str(out.resolve()) in result.stdout.strip()

    def test_config_accessible(self, tmp_path):
        config = ExecutionConfig(
            timeout_seconds=42,
            workspace_dir=str(tmp_path),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        assert executor.config.timeout_seconds == 42

    def test_explicit_config_applied(self):
        config = ExecutionConfig(timeout_seconds=42, workspace_dir=None, output_dir=None)
        executor = HostExecutor(config=config, approval_fn=always_approve)
        assert executor.config.timeout_seconds == 42

    def test_file_written_to_output_dir(self, tmp_path):
        out = tmp_path / "out"
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(out),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)
        result = executor.execute(
            "with open('test.txt', 'w') as f: f.write('hello')"
        )

        assert result.exit_code == 0
        assert (out / "test.txt").read_text() == "hello"

    def test_multiple_executions_isolated(self, tmp_path):
        config = ExecutionConfig(
            workspace_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        executor = HostExecutor(config=config, approval_fn=always_approve)

        r1 = executor.execute("x = 42; print(x)")
        r2 = executor.execute("print('hello')")

        assert r1.exit_code == 0
        assert r2.exit_code == 0


# ── Safety Validator unit tests ────────────────────────────────


class TestSafetyValidator:

    def test_safe_code_passes(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("print('hello')")
        assert result.safe is True
        assert result.violations == []

    def test_subprocess_blocked(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("import subprocess; subprocess.run(['ls'])")
        assert result.safe is False
        assert any("subprocess" in v for v in result.violations)

    def test_os_system_blocked(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("import os; os.system('rm -rf /')")
        assert result.safe is False
        assert any("os.system" in v for v in result.violations)

    def test_eval_blocked(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("eval('1+1')")
        assert result.safe is False

    def test_exec_blocked(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("exec('print(1)')")
        assert result.safe is False

    def test_shutil_rmtree_blocked(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("import shutil; shutil.rmtree('/tmp')")
        assert result.safe is False

    def test_syntax_error_passes_safety(self, tmp_path):
        v = SafetyValidator([tmp_path])
        result = v.validate("def foo(:")
        assert result.safe is True  # syntax errors are not safety issues
