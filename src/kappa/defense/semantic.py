"""Semantic loop detector using Jaccard similarity.

Tracks a sliding window of recent agent outputs (think blocks, tool
arguments) and raises ``SemanticLoopException`` when the average
pairwise similarity exceeds a configurable threshold — indicating
the agent is stuck repeating the same futile attempts.
"""

from __future__ import annotations

from collections import deque

from kappa.config import SemanticConfig
from kappa.exceptions import SemanticLoopException


def _tokenize(text: str) -> set[str]:
    """Split text into a set of lowercased word tokens."""
    return set(text.lower().split())


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two texts.

    Returns a value in [0.0, 1.0] where 1.0 means identical token sets.
    Returns 0.0 if both texts are empty.
    """
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


class SemanticLoopDetector:
    """Detects semantic repetition in agent behaviour.

    Maintains a sliding window of the most recent ``window_size`` text
    entries.  After at least ``min_samples`` entries have been recorded,
    ``check()`` computes the average pairwise Jaccard similarity across
    the window and raises ``SemanticLoopException`` if it exceeds
    ``similarity_threshold``.

    Args:
        config: SemanticConfig with window_size, similarity_threshold,
            and min_samples.
    """

    def __init__(self, config: SemanticConfig | None = None) -> None:
        cfg = config or SemanticConfig()
        self._window_size = cfg.window_size
        self._threshold = cfg.similarity_threshold
        self._min_samples = cfg.min_samples
        self._history: deque[str] = deque(maxlen=cfg.window_size)

    @property
    def history(self) -> list[str]:
        """Current entries in the sliding window (oldest first)."""
        return list(self._history)

    def record(self, text: str) -> None:
        """Add a text entry to the sliding window."""
        self._history.append(text)

    def _average_pairwise_similarity(self) -> float:
        """Compute average Jaccard similarity across all pairs in the window."""
        entries = list(self._history)
        n = len(entries)
        if n < 2:
            return 0.0

        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += jaccard_similarity(entries[i], entries[j])
                pairs += 1

        return total / pairs

    def check(self) -> None:
        """Check for semantic repetition and raise if detected.

        Does nothing if fewer than ``min_samples`` entries have been
        recorded (not enough data to judge).

        Raises:
            SemanticLoopException: If the average pairwise similarity
                in the current window exceeds the threshold.
        """
        if len(self._history) < self._min_samples:
            return

        avg_sim = self._average_pairwise_similarity()
        if avg_sim >= self._threshold:
            raise SemanticLoopException(
                f"Semantic loop detected: average similarity {avg_sim:.2f} "
                f">= threshold {self._threshold:.2f} "
                f"over last {len(self._history)} entries.",
                similarity=avg_sim,
            )

    def reset(self) -> None:
        """Clear the history window."""
        self._history.clear()
