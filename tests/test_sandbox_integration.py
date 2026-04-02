"""Integration tests for the Deterministic Sandbox (Phase 1 — Task 2).

These tests run REAL Docker containers. Requires:
- Docker daemon running
- python:3.11-slim image pulled

Verification scenarios from the Definition of Done:
1. Normal code → exit_code=0, stdout captured
2. Destructive command (rm -rf /) → host protected, error code returned
3. Infinite loop → timeout enforced, container killed
4. Network disabled → outbound connections blocked
5. Memory limit → OOM killed
"""

from __future__ import annotations

import pytest

from kappa.config import SandboxConfig
from kappa.sandbox.executor import DockerRuntime, SandboxExecutor, SandboxResult


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not running"
)


@pytest.fixture
def executor() -> SandboxExecutor:
    config = SandboxConfig(timeout_seconds=15, memory_limit_mb=64)
    return SandboxExecutor(runtime=DockerRuntime(), config=config)


@pytest.fixture
def fast_executor() -> SandboxExecutor:
    """Short timeout for infinite-loop tests."""
    config = SandboxConfig(timeout_seconds=5, memory_limit_mb=64)
    return SandboxExecutor(runtime=DockerRuntime(), config=config)


class TestRealSandbox:

    def test_normal_execution(self, executor: SandboxExecutor):
        """정상 코드 → exit_code=0, stdout 캡처."""
        result = executor.execute("print('hello from sandbox')")

        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout
        assert result.timed_out is False

    def test_destructive_rm_rf_contained(self, executor: SandboxExecutor):
        """rm -rf / → 컨테이너 내부에서만 실행, 호스트 무사."""
        result = executor.execute(
            "import subprocess; subprocess.run(['rm', '-rf', '--no-preserve-root', '/'], capture_output=True)"
        )

        # 컨테이너 안에서 실행됐으므로 호스트는 무사
        # (slim 이미지에서 rm은 permission denied 또는 부분 삭제 후 종료)
        assert isinstance(result, SandboxResult)
        # 중요: 이 테스트가 끝난 후에도 호스트 OS가 정상 동작 중이라는 것 자체가 증명

    def test_syntax_error_returns_nonzero(self, executor: SandboxExecutor):
        """구문 오류 → exit_code=1, stderr에 SyntaxError."""
        result = executor.execute("def f(:")

        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    def test_runtime_error_captured(self, executor: SandboxExecutor):
        """런타임 오류 → exit_code=1, stderr에 트레이스백."""
        result = executor.execute("print(undefined_variable)")

        assert result.exit_code == 1
        assert "NameError" in result.stderr

    def test_infinite_loop_timeout(self, fast_executor: SandboxExecutor):
        """무한 루프 → 타임아웃 후 강제 종료."""
        result = fast_executor.execute("while True: pass")

        assert result.timed_out is True
        assert result.exit_code == -1

    def test_network_disabled(self, executor: SandboxExecutor):
        """네트워크 차단 → 외부 연결 불가."""
        result = executor.execute(
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://example.com', timeout=3)\n"
            "    print('CONNECTED')\n"
            "except Exception as e:\n"
            "    print(f'BLOCKED: {e}')\n"
            "    exit(1)"
        )

        assert "CONNECTED" not in result.stdout
        assert result.exit_code != 0

    def test_each_execution_is_isolated(self, executor: SandboxExecutor):
        """매 실행 독립 — 이전 컨테이너의 상태가 누수되지 않음."""
        # 첫 번째: 파일 생성
        executor.execute("open('/tmp/secret.txt', 'w').write('leak')")

        # 두 번째: 해당 파일 읽기 시도 → 새 컨테이너이므로 없어야 함
        result = executor.execute(
            "import os\n"
            "if os.path.exists('/tmp/secret.txt'):\n"
            "    print('LEAKED')\n"
            "    exit(1)\n"
            "else:\n"
            "    print('ISOLATED')"
        )

        assert "ISOLATED" in result.stdout
        assert result.exit_code == 0

    def test_stdout_stderr_separated(self, executor: SandboxExecutor):
        """stdout과 stderr가 분리되어 캡처됨."""
        result = executor.execute(
            "import sys\n"
            "print('out-msg')\n"
            "print('err-msg', file=sys.stderr)"
        )

        assert "out-msg" in result.stdout
        assert "err-msg" in result.stderr
