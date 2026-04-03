"""Per-key serialisation lane for resource contention control.

When multiple workers target the same resource (identified by a string
key), SessionLane ensures they queue up and execute one at a time.
Independent keys run fully in parallel with zero contention.

Two implementations are provided:

* ``SyncSessionLane`` — uses ``threading.Lock`` for synchronous /
  ``ThreadPoolExecutor`` environments (used by the Orchestrator).
* ``AsyncSessionLane`` — uses ``asyncio.Lock`` for native async code.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from kappa.config import SessionLaneConfig
from kappa.exceptions import SessionLaneTimeout


class SyncSessionLane:
    """Thread-safe per-key serialisation using ``threading.Lock``.

    Each unique *key* gets its own lock.  Callers sharing a key are
    serialised (FIFO via the OS thread scheduler).  Callers with
    different keys proceed in parallel without blocking each other.

    Usage::

        lane = SyncSessionLane()
        with lane.lane("file:config.py"):
            # Only one thread at a time for this key
            do_work()

    Args:
        config: Optional ``SessionLaneConfig`` with timeout.
    """

    def __init__(self, config: SessionLaneConfig | None = None) -> None:
        cfg = config or SessionLaneConfig()
        self._timeout = cfg.timeout
        # Guards mutations to ``_locks`` dict itself
        self._meta_lock = threading.Lock()
        # Per-key locks (created lazily)
        self._locks: dict[str, threading.Lock] = {}

    def _get_lock(self, key: str) -> threading.Lock:
        """Return the lock for *key*, creating it lazily if needed."""
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def acquire(self, key: str) -> None:
        """Acquire the lock for *key*, blocking up to ``timeout`` seconds.

        Raises:
            SessionLaneTimeout: If the lock cannot be acquired in time.
        """
        lock = self._get_lock(key)
        acquired = lock.acquire(timeout=self._timeout)
        if not acquired:
            raise SessionLaneTimeout(
                f"SessionLane timeout ({self._timeout}s) waiting for key {key!r}"
            )

    def release(self, key: str) -> None:
        """Release the lock for *key*.

        Raises:
            RuntimeError: If the lock is not currently held.
        """
        lock = self._get_lock(key)
        lock.release()

    @contextmanager
    def lane(self, key: str) -> Iterator[None]:
        """Context manager that acquires/releases the per-key lock.

        Usage::

            with lane.lane("resource:abc"):
                do_exclusive_work()
        """
        self.acquire(key)
        try:
            yield
        finally:
            self.release(key)

    @property
    def active_keys(self) -> list[str]:
        """Return the list of keys that currently have locks allocated."""
        with self._meta_lock:
            return list(self._locks.keys())


class AsyncSessionLane:
    """Async per-key serialisation using ``asyncio.Lock``.

    Semantics mirror ``SyncSessionLane`` but for ``asyncio`` event loops.

    Usage::

        lane = AsyncSessionLane()
        async with lane.lane("file:config.py"):
            await do_work()
    """

    def __init__(self, config: SessionLaneConfig | None = None) -> None:
        cfg = config or SessionLaneConfig()
        self._timeout = cfg.timeout
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def acquire(self, key: str) -> None:
        """Acquire the async lock for *key* with timeout.

        Raises:
            SessionLaneTimeout: On timeout.
        """
        lock = self._get_lock(key)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise SessionLaneTimeout(
                f"AsyncSessionLane timeout ({self._timeout}s) "
                f"waiting for key {key!r}"
            )

    async def release(self, key: str) -> None:
        lock = self._get_lock(key)
        lock.release()

    @asynccontextmanager
    async def lane(self, key: str) -> AsyncIterator[None]:
        await self.acquire(key)
        try:
            yield
        finally:
            await self.release(key)
