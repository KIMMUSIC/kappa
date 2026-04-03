"""Infrastructure resilience primitives for multi-agent orchestration.

Provides per-key serialisation (SessionLane) and decorrelated jitter
exponential backoff to prevent resource contention and thundering-herd
API storms when multiple workers run concurrently.
"""

from kappa.infra.jitter import jitter_backoff_sync
from kappa.infra.session_lane import SyncSessionLane

__all__ = ["SyncSessionLane", "jitter_backoff_sync"]
