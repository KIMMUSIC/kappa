"""Deterministic sandbox for isolated code execution via Docker.

All agent-generated code runs inside ephemeral, resource-limited Docker
containers — never on the host OS.  Each execution creates a fresh
container that is destroyed after completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kappa.config import SandboxConfig
from kappa.exceptions import SandboxExecutionError


@dataclass(frozen=True)
class SandboxResult:
    """Structured output from a sandboxed code execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@runtime_checkable
class ContainerRuntime(Protocol):
    """Interface for container-based code execution backends."""

    def run(
        self,
        *,
        image: str,
        command: list[str],
        mem_limit: str,
        network_disabled: bool,
        timeout: int,
    ) -> SandboxResult: ...


class DockerRuntime:
    """Docker-backed container runtime using the docker-py SDK.

    Each ``run()`` call creates an ephemeral container with strict
    resource limits, collects output, and removes the container —
    regardless of whether execution succeeded or failed.
    """

    def __init__(self) -> None:
        try:
            import docker  # noqa: F811

            self._client = docker.from_env()
        except Exception as exc:
            raise SandboxExecutionError(
                f"Failed to connect to Docker daemon: {exc}"
            ) from exc

    def run(
        self,
        *,
        image: str,
        command: list[str],
        mem_limit: str,
        network_disabled: bool,
        timeout: int,
    ) -> SandboxResult:
        container = None
        timed_out = False
        try:
            container = self._client.containers.run(
                image=image,
                command=command,
                mem_limit=mem_limit,
                network_disabled=network_disabled,
                detach=True,
                stdout=True,
                stderr=True,
            )

            try:
                wait_result = container.wait(timeout=timeout)
                exit_code = wait_result.get("StatusCode", -1)
            except Exception:
                # Timeout or connection error — kill the runaway container
                timed_out = True
                try:
                    container.kill()
                except Exception:
                    pass
                exit_code = -1

            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )

            if timed_out:
                stderr = f"Execution timed out after {timeout}s.\n" + stderr

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
            )
        except SandboxExecutionError:
            raise
        except Exception as exc:
            raise SandboxExecutionError(
                f"Docker execution failed: {exc}"
            ) from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


class SandboxExecutor:
    """Executes code strings in an isolated, ephemeral container.

    Guarantees:
    - Code never runs on the host OS (always containerised)
    - Each execution gets a fresh, disposable environment
    - Resource limits (memory, timeout) are enforced
    - Network access is disabled by default
    - Container is destroyed after execution completes or times out
    """

    def __init__(
        self,
        runtime: ContainerRuntime | None = None,
        config: SandboxConfig | None = None,
    ) -> None:
        self._config = config or SandboxConfig()
        self._runtime = runtime or DockerRuntime()

    @property
    def config(self) -> SandboxConfig:
        return self._config

    def execute(self, code: str) -> SandboxResult:
        """Execute a Python code string inside an isolated container.

        Args:
            code: Python source code to execute.

        Returns:
            SandboxResult with exit_code, stdout, stderr, and timed_out flag.

        Raises:
            SandboxExecutionError: If the sandbox infrastructure itself fails
                (e.g., Docker daemon unavailable).  User-code errors are
                captured as non-zero exit_code, not exceptions.
        """
        return self._runtime.run(
            image=self._config.docker_image,
            command=["python", "-c", code],
            mem_limit=f"{self._config.memory_limit_mb}m",
            network_disabled=not self._config.network_enabled,
            timeout=self._config.timeout_seconds,
        )
