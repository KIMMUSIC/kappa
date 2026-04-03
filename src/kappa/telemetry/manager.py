"""TelemetryManager: observer-pattern JSONL trajectory recorder.

Records structured ``TrajectoryRecord`` entries to a JSONL file
each time a worker completes or is rejected by the Reviewer.

Three mandatory metadata fields per record:
  - ``think``:    worker's explicit reasoning trace (<think> block)
  - ``critique``: reviewer's focused critique text (empty if approved)
  - ``score``:    composite quality score 0.0–1.0

Thread-safe: a ``threading.Lock`` serialises all file I/O so that
concurrent reviewer calls from the orchestrator's ThreadPoolExecutor
never corrupt the log.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kappa.config import TelemetryConfig


@dataclass(frozen=True)
class TrajectoryRecord:
    """Single worker execution trajectory."""

    task_id: str
    worker_goal: str
    think: str
    critique: str
    score: float
    outcome: str  # success | rejected | error
    token_usage: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ for default timestamp
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            )


class TelemetryManager:
    """Observer-pattern JSONL trajectory recorder.

    Usage::

        mgr = TelemetryManager()
        mgr.record(TrajectoryRecord(
            task_id="task-001", worker_goal="Print hello",
            think="I will print hello", critique="", score=0.95,
            outcome="success",
        ))
        print(mgr.summary())

    Args:
        config: Optional ``TelemetryConfig`` with ``enabled`` flag
            and ``log_path``.
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        cfg = config or TelemetryConfig()
        self._enabled = cfg.enabled
        self._log_path = Path(cfg.log_path)
        self._lock = threading.Lock()

        if self._enabled:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, trajectory: TrajectoryRecord) -> None:
        """Append a trajectory record as one JSONL line.

        Thread-safe.  No-op if telemetry is disabled.
        """
        if not self._enabled:
            return

        line = json.dumps(asdict(trajectory), ensure_ascii=False)
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[TrajectoryRecord]:
        """Load every record from the JSONL log file.

        Returns an empty list if the file does not exist or telemetry
        is disabled.
        """
        if not self._enabled or not self._log_path.exists():
            return []

        records: list[TrajectoryRecord] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(TrajectoryRecord(**data))
        return records

    def summary(self) -> dict:
        """Compute aggregate statistics over all recorded trajectories.

        Returns a dict with:
          - ``total``:          number of records
          - ``avg_score``:      mean quality score
          - ``rejection_rate``: fraction of rejected outcomes
          - ``success_count``:  number of successful outcomes
          - ``rejected_count``: number of rejected outcomes
        """
        records = self.read_all()
        if not records:
            return {
                "total": 0,
                "avg_score": 0.0,
                "rejection_rate": 0.0,
                "success_count": 0,
                "rejected_count": 0,
            }

        total = len(records)
        avg_score = sum(r.score for r in records) / total
        rejected = sum(1 for r in records if r.outcome == "rejected")
        success = sum(1 for r in records if r.outcome == "success")

        return {
            "total": total,
            "avg_score": round(avg_score, 4),
            "rejection_rate": round(rejected / total, 4),
            "success_count": success,
            "rejected_count": rejected,
        }
