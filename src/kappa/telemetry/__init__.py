"""Agent-RRM telemetry pipeline for trajectory recording.

Provides an observer-pattern JSONL logger that captures the reasoning
trajectory of each worker execution — think traces, reviewer critiques,
and composite quality scores — for future Agent-RRM training.
"""

from kappa.telemetry.manager import TelemetryManager, TrajectoryRecord

__all__ = ["TelemetryManager", "TrajectoryRecord"]
