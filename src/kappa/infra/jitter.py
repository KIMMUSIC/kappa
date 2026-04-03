"""Decorrelated Jitter exponential backoff.

Prevents the *Thundering Herd* problem when multiple workers hit an
API rate limit (HTTP 429) simultaneously.  Instead of retrying at the
same instant, each worker waits a randomised, exponentially growing
delay — the "decorrelated jitter" algorithm recommended by AWS:

    sleep = min(cap, random_between(base, prev_sleep * 3))

Two entry-points are provided:

* ``jitter_backoff_sync`` — blocking (for ``ThreadPoolExecutor``).
* ``jitter_backoff``      — native ``asyncio``.

Both accept a ``retryable`` predicate to decide which exceptions
warrant a retry (default: any ``Exception``).
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, TypeVar

from kappa.config import BackoffConfig

T = TypeVar("T")


def _next_delay(prev: float, base: float, cap: float) -> float:
    """Compute decorrelated jitter delay.

    Algorithm (AWS recommendation)::

        sleep = min(cap, random.uniform(base, prev * 3))

    Guarantees the delay is always in ``[base, cap]``.
    """
    raw = random.uniform(base, max(base, prev * 3))
    return min(cap, raw)


def jitter_backoff_sync(
    fn: Callable[..., T],
    *args: Any,
    config: BackoffConfig | None = None,
    retryable: Callable[[Exception], bool] | None = None,
    **kwargs: Any,
) -> T:
    """Call *fn* with decorrelated-jitter retries (blocking).

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        config: Backoff parameters (base delay, max delay, max retries).
        retryable: Predicate returning ``True`` for exceptions that should
            trigger a retry.  Defaults to retrying all ``Exception``.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on the first successful call.

    Raises:
        The last exception raised by *fn* if all retries are exhausted.
    """
    cfg = config or BackoffConfig()
    should_retry = retryable or (lambda _exc: True)

    prev_delay = cfg.base_delay
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not should_retry(exc):
                raise
            if attempt >= cfg.max_retries:
                raise

            prev_delay = _next_delay(prev_delay, cfg.base_delay, cfg.max_delay)
            time.sleep(prev_delay)

    # Unreachable, but keeps type-checkers happy
    assert last_exc is not None  # noqa: S101
    raise last_exc


async def jitter_backoff(
    fn: Callable[..., T],
    *args: Any,
    config: BackoffConfig | None = None,
    retryable: Callable[[Exception], bool] | None = None,
    **kwargs: Any,
) -> T:
    """Call *fn* with decorrelated-jitter retries (async).

    Same semantics as ``jitter_backoff_sync`` but uses
    ``asyncio.sleep`` for non-blocking waits.

    If *fn* is a coroutine function the result is awaited automatically.
    """
    cfg = config or BackoffConfig()
    should_retry = retryable or (lambda _exc: True)

    prev_delay = cfg.base_delay
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result  # type: ignore[return-value]
        except Exception as exc:
            last_exc = exc
            if not should_retry(exc):
                raise
            if attempt >= cfg.max_retries:
                raise

            prev_delay = _next_delay(prev_delay, cfg.base_delay, cfg.max_delay)
            await asyncio.sleep(prev_delay)

    assert last_exc is not None  # noqa: S101
    raise last_exc
