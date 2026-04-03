"""Tests for Phase 3 Task 1: Infrastructure Resilience.

Covers:
  - SyncSessionLane: per-key serialisation, parallel independent keys,
    timeout handling, context manager usage.
  - AsyncSessionLane: same semantics in asyncio.
  - jitter_backoff_sync: decorrelated jitter, retry logic, retryable filter.
  - jitter_backoff (async): same semantics in asyncio.
  - BackoffConfig / SessionLaneConfig: config defaults and overrides.
  - New exceptions: SessionLaneTimeout, OrchestratorError.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from kappa.config import BackoffConfig, SessionLaneConfig
from kappa.exceptions import OrchestratorError, SessionLaneTimeout
from kappa.infra.jitter import _next_delay, jitter_backoff, jitter_backoff_sync
from kappa.infra.session_lane import AsyncSessionLane, SyncSessionLane


# ═════════════════════════════════════════════════════════════════════
# SyncSessionLane
# ═════════════════════════════════════════════════════════════════════


class TestSyncSessionLane:
    """Thread-based per-key serialisation tests."""

    def test_same_key_serialises(self):
        """Two threads competing for the same key must execute sequentially."""
        lane = SyncSessionLane()
        results: list[str] = []
        barrier = threading.Barrier(2)

        def worker(name: str) -> None:
            barrier.wait()  # ensure both start at the same instant
            with lane.lane("shared"):
                results.append(f"{name}_start")
                time.sleep(0.05)
                results.append(f"{name}_end")

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # One must complete fully before the other starts
        assert results[:2] in (["A_start", "A_end"], ["B_start", "B_end"])
        assert results[2:] in (["A_start", "A_end"], ["B_start", "B_end"])

    def test_different_keys_run_in_parallel(self):
        """Threads with different keys must overlap (true parallelism)."""
        lane = SyncSessionLane()
        timestamps: dict[str, list[float]] = {"key_A": [], "key_B": []}
        barrier = threading.Barrier(2)

        def worker(key: str) -> None:
            barrier.wait()
            with lane.lane(key):
                timestamps[key].append(time.monotonic())
                time.sleep(0.05)
                timestamps[key].append(time.monotonic())

        t1 = threading.Thread(target=worker, args=("key_A",))
        t2 = threading.Thread(target=worker, args=("key_B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both started before either finished → parallel
        a_start, a_end = timestamps["key_A"]
        b_start, b_end = timestamps["key_B"]
        assert a_start < b_end and b_start < a_end, "Should overlap"

    def test_timeout_raises(self):
        """Lock held beyond timeout must raise SessionLaneTimeout."""
        config = SessionLaneConfig(timeout=0.05)
        lane = SyncSessionLane(config=config)
        acquired = threading.Event()

        def holder() -> None:
            with lane.lane("x"):
                acquired.set()
                time.sleep(0.3)  # hold far longer than timeout

        t = threading.Thread(target=holder)
        t.start()
        acquired.wait()

        with pytest.raises(SessionLaneTimeout, match="timeout"):
            lane.acquire("x")

        t.join()

    def test_context_manager_releases_on_exception(self):
        """Lock must be released even if the body raises."""
        lane = SyncSessionLane()

        with pytest.raises(ValueError, match="boom"):
            with lane.lane("k"):
                raise ValueError("boom")

        # Lock should be free — acquire must succeed instantly
        lane.acquire("k")
        lane.release("k")

    def test_active_keys(self):
        """active_keys should reflect allocated locks."""
        lane = SyncSessionLane()
        assert lane.active_keys == []

        with lane.lane("alpha"):
            assert "alpha" in lane.active_keys

    def test_reentrant_different_threads(self):
        """Two threads should each get their own turn on the same key."""
        lane = SyncSessionLane()
        order: list[int] = []

        def worker(n: int) -> None:
            with lane.lane("same"):
                order.append(n)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_thread_pool_executor_serialisation(self):
        """Validate serialisation when driven by ThreadPoolExecutor."""
        lane = SyncSessionLane()
        counter = {"value": 0}
        results: list[int] = []

        def increment() -> int:
            with lane.lane("counter"):
                current = counter["value"]
                time.sleep(0.01)  # simulate work
                counter["value"] = current + 1
                results.append(counter["value"])
                return counter["value"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(increment) for _ in range(8)]
            for f in futures:
                f.result()

        # Must be sequential: 1, 2, 3, ..., 8 with no duplicates
        assert results == list(range(1, 9))


# ═════════════════════════════════════════════════════════════════════
# AsyncSessionLane
# ═════════════════════════════════════════════════════════════════════


class TestAsyncSessionLane:
    """Asyncio per-key serialisation tests."""

    @pytest.mark.asyncio
    async def test_same_key_serialises(self):
        lane = AsyncSessionLane()
        results: list[str] = []

        async def worker(name: str) -> None:
            async with lane.lane("shared"):
                results.append(f"{name}_start")
                await asyncio.sleep(0.05)
                results.append(f"{name}_end")

        await asyncio.gather(worker("A"), worker("B"))

        assert results[:2] in (["A_start", "A_end"], ["B_start", "B_end"])
        assert results[2:] in (["A_start", "A_end"], ["B_start", "B_end"])

    @pytest.mark.asyncio
    async def test_different_keys_parallel(self):
        lane = AsyncSessionLane()
        timestamps: dict[str, list[float]] = {"A": [], "B": []}

        async def worker(label: str, key: str) -> None:
            async with lane.lane(key):
                timestamps[label].append(time.monotonic())
                await asyncio.sleep(0.05)
                timestamps[label].append(time.monotonic())

        await asyncio.gather(worker("A", "k1"), worker("B", "k2"))

        a_start, a_end = timestamps["A"]
        b_start, b_end = timestamps["B"]
        assert a_start < b_end and b_start < a_end

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        config = SessionLaneConfig(timeout=0.05)
        lane = AsyncSessionLane(config=config)

        async with lane.lane("x"):
            with pytest.raises(SessionLaneTimeout, match="timeout"):
                await lane.acquire("x")

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        lane = AsyncSessionLane()

        with pytest.raises(ValueError, match="boom"):
            async with lane.lane("k"):
                raise ValueError("boom")

        # Should be acquirable again
        await lane.acquire("k")
        await lane.release("k")


# ═════════════════════════════════════════════════════════════════════
# Jitter backoff internals
# ═════════════════════════════════════════════════════════════════════


class TestNextDelay:
    """Unit tests for the decorrelated jitter delay calculation."""

    def test_delay_within_bounds(self):
        """Delay must always be in [base, cap]."""
        for _ in range(200):
            d = _next_delay(prev=2.0, base=1.0, cap=60.0)
            assert 1.0 <= d <= 60.0

    def test_delay_capped(self):
        """Even with a huge prev, delay must not exceed cap."""
        for _ in range(100):
            d = _next_delay(prev=1000.0, base=1.0, cap=10.0)
            assert d <= 10.0

    def test_delay_varies(self):
        """Successive calls should produce different values (jitter)."""
        delays = {_next_delay(prev=2.0, base=1.0, cap=60.0) for _ in range(50)}
        assert len(delays) > 1, "Jitter should produce variance"


# ═════════════════════════════════════════════════════════════════════
# jitter_backoff_sync
# ═════════════════════════════════════════════════════════════════════


class TestJitterBackoffSync:
    """Synchronous decorrelated jitter retry tests."""

    def test_succeeds_first_try(self):
        fn = MagicMock(return_value=42)
        result = jitter_backoff_sync(fn, config=BackoffConfig(max_retries=3))
        assert result == 42
        assert fn.call_count == 1

    def test_retries_then_succeeds(self):
        fn = MagicMock(side_effect=[ValueError("a"), ValueError("b"), 99])
        cfg = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=5)
        result = jitter_backoff_sync(fn, config=cfg)
        assert result == 99
        assert fn.call_count == 3

    def test_exhausts_retries(self):
        fn = MagicMock(side_effect=ValueError("fail"))
        cfg = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=2)
        with pytest.raises(ValueError, match="fail"):
            jitter_backoff_sync(fn, config=cfg)
        assert fn.call_count == 3  # initial + 2 retries

    def test_non_retryable_raises_immediately(self):
        fn = MagicMock(side_effect=TypeError("bad"))
        cfg = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=5)
        with pytest.raises(TypeError, match="bad"):
            jitter_backoff_sync(
                fn,
                config=cfg,
                retryable=lambda e: isinstance(e, ValueError),
            )
        assert fn.call_count == 1

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        jitter_backoff_sync(fn, 1, 2, config=BackoffConfig(), key="val")
        fn.assert_called_once_with(1, 2, key="val")

    def test_delays_increase_with_jitter(self):
        """Verify that actual sleep happens (wall-clock timing)."""
        call_times: list[float] = []
        call_count = {"n": 0}

        def flaky() -> str:
            call_times.append(time.monotonic())
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return "done"

        cfg = BackoffConfig(base_delay=0.02, max_delay=1.0, max_retries=5)
        result = jitter_backoff_sync(flaky, config=cfg)
        assert result == "done"

        # There should be measurable gaps between calls
        for i in range(1, len(call_times)):
            gap = call_times[i] - call_times[i - 1]
            assert gap >= 0.01, f"Gap {i} too small: {gap}"

    def test_default_config_used(self):
        """Should work without explicit config."""
        fn = MagicMock(return_value="ok")
        assert jitter_backoff_sync(fn) == "ok"


# ═════════════════════════════════════════════════════════════════════
# jitter_backoff (async)
# ═════════════════════════════════════════════════════════════════════


class TestJitterBackoffAsync:
    """Async decorrelated jitter retry tests."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        fn = MagicMock(return_value=42)
        result = await jitter_backoff(fn, config=BackoffConfig(max_retries=3))
        assert result == 42

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        fn = MagicMock(side_effect=[ValueError("a"), ValueError("b"), 99])
        cfg = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=5)
        result = await jitter_backoff(fn, config=cfg)
        assert result == 99
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        fn = MagicMock(side_effect=RuntimeError("fail"))
        cfg = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=2)
        with pytest.raises(RuntimeError, match="fail"):
            await jitter_backoff(fn, config=cfg)
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_awaits_coroutine_fn(self):
        """If fn returns a coroutine, it should be awaited."""

        async def async_fn() -> str:
            return "async_ok"

        result = await jitter_backoff(async_fn, config=BackoffConfig(max_retries=1))
        assert result == "async_ok"

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        fn = MagicMock(side_effect=TypeError("bad"))
        cfg = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=5)
        with pytest.raises(TypeError):
            await jitter_backoff(
                fn,
                config=cfg,
                retryable=lambda e: isinstance(e, ValueError),
            )
        assert fn.call_count == 1


# ═════════════════════════════════════════════════════════════════════
# Config & Exception sanity checks
# ═════════════════════════════════════════════════════════════════════


class TestConfigDefaults:
    """Verify new config dataclasses have correct defaults."""

    def test_backoff_config_defaults(self):
        cfg = BackoffConfig()
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.max_retries == 5

    def test_session_lane_config_defaults(self):
        cfg = SessionLaneConfig()
        assert cfg.timeout == 30.0

    def test_backoff_config_override(self):
        cfg = BackoffConfig(base_delay=0.5, max_delay=10.0, max_retries=3)
        assert cfg.base_delay == 0.5
        assert cfg.max_delay == 10.0
        assert cfg.max_retries == 3

    def test_session_lane_config_override(self):
        cfg = SessionLaneConfig(timeout=5.0)
        assert cfg.timeout == 5.0


class TestNewExceptions:
    """Verify new exception classes exist and inherit correctly."""

    def test_session_lane_timeout_is_kappa_error(self):
        from kappa.exceptions import KappaError

        exc = SessionLaneTimeout("test")
        assert isinstance(exc, KappaError)
        assert str(exc) == "test"

    def test_orchestrator_error_is_kappa_error(self):
        from kappa.exceptions import KappaError

        exc = OrchestratorError("test")
        assert isinstance(exc, KappaError)
        assert str(exc) == "test"
