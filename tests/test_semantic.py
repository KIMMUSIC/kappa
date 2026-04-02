"""Tests for the semantic infinite loop detector (Task 3).

Covers:
- Jaccard similarity calculation
- Sliding window behaviour
- Threshold detection and SemanticLoopException
- Edge cases (empty text, below min_samples, reset)
- SemanticConfig integration
"""

from __future__ import annotations

import pytest

from kappa.config import SemanticConfig
from kappa.defense.semantic import SemanticLoopDetector, jaccard_similarity
from kappa.exceptions import SemanticLoopException


# ── Jaccard similarity function ────────────────────────────────


class TestJaccardSimilarity:

    def test_identical_texts(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert jaccard_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        # {"hello", "world"} & {"hello", "foo"} = {"hello"}
        # union = {"hello", "world", "foo"} → 1/3
        sim = jaccard_similarity("hello world", "hello foo")
        assert abs(sim - 1 / 3) < 1e-9

    def test_case_insensitive(self):
        assert jaccard_similarity("Hello World", "hello world") == 1.0

    def test_both_empty(self):
        assert jaccard_similarity("", "") == 0.0

    def test_one_empty(self):
        assert jaccard_similarity("hello", "") == 0.0

    def test_duplicate_words_ignored(self):
        # set("a a a".split()) == {"a"}, set("a".split()) == {"a"}
        assert jaccard_similarity("a a a", "a") == 1.0


# ── SemanticLoopDetector ───────────────────────────────────────


class TestDetectorBasics:

    def test_no_exception_below_min_samples(self):
        detector = SemanticLoopDetector(SemanticConfig(min_samples=3))
        detector.record("same text")
        detector.record("same text")
        # Only 2 entries, min_samples=3 → should not raise
        detector.check()  # no exception

    def test_history_tracks_entries(self):
        detector = SemanticLoopDetector()
        detector.record("a")
        detector.record("b")
        assert detector.history == ["a", "b"]

    def test_window_slides(self):
        detector = SemanticLoopDetector(SemanticConfig(window_size=3))
        detector.record("a")
        detector.record("b")
        detector.record("c")
        detector.record("d")  # "a" should be evicted
        assert detector.history == ["b", "c", "d"]

    def test_reset_clears_history(self):
        detector = SemanticLoopDetector()
        detector.record("text")
        detector.reset()
        assert detector.history == []


class TestDetectorLoopDetection:

    def test_identical_texts_trigger_exception(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=5, similarity_threshold=0.85, min_samples=3)
        )
        for _ in range(3):
            detector.record("I need to fix the bug in the parser module")
        with pytest.raises(SemanticLoopException, match="Semantic loop detected"):
            detector.check()

    def test_exception_carries_similarity(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=5, similarity_threshold=0.85, min_samples=3)
        )
        for _ in range(3):
            detector.record("identical repeated thought")
        with pytest.raises(SemanticLoopException) as exc_info:
            detector.check()
        assert exc_info.value.similarity == 1.0

    def test_diverse_texts_no_exception(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=5, similarity_threshold=0.85, min_samples=3)
        )
        detector.record("first I will analyze the problem")
        detector.record("now let me try a different approach using recursion")
        detector.record("the solution requires dynamic programming instead")
        detector.check()  # no exception — all different

    def test_near_threshold_below_does_not_trigger(self):
        config = SemanticConfig(
            window_size=3, similarity_threshold=0.90, min_samples=3
        )
        detector = SemanticLoopDetector(config)
        # These share many words but not 90%
        detector.record("fix the parser error in module")
        detector.record("fix the linter error in module")
        detector.record("fix the sandbox error in module")
        detector.check()  # should not raise — similarity below 0.90

    def test_gradual_drift_eventually_triggers(self):
        config = SemanticConfig(
            window_size=3, similarity_threshold=0.80, min_samples=3
        )
        detector = SemanticLoopDetector(config)
        # First entries are diverse → no trigger
        detector.record("analyze the input data")
        detector.record("write the sorting algorithm")
        detector.record("test the output results")
        detector.check()  # diverse, no trigger

        # Now feed identical entries → old diverse ones slide out
        detector.record("fix the same bug again")
        detector.record("fix the same bug again")
        detector.record("fix the same bug again")
        with pytest.raises(SemanticLoopException):
            detector.check()


class TestDetectorWithToolArgs:

    def test_repeated_tool_calls_detected(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=4, similarity_threshold=0.85, min_samples=3)
        )
        tool_args = '{"name": "read_memory", "kwargs": {"path": "LEARNINGS.md"}}'
        for _ in range(4):
            detector.record(tool_args)
        with pytest.raises(SemanticLoopException):
            detector.check()

    def test_varied_tool_calls_no_trigger(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=4, similarity_threshold=0.85, min_samples=3)
        )
        detector.record('{"name": "read_memory", "kwargs": {"path": "LEARNINGS.md"}}')
        detector.record('{"name": "write_memory", "kwargs": {"path": "notes.md", "content": "hello"}}')
        detector.record('{"name": "read_memory", "kwargs": {"path": "config.yaml"}}')
        detector.check()  # no exception


class TestDetectorEdgeCases:

    def test_empty_strings_recorded(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=3, similarity_threshold=0.85, min_samples=3)
        )
        for _ in range(3):
            detector.record("")
        # Empty strings → jaccard returns 0.0, so no trigger
        detector.check()

    def test_single_word_repetition(self):
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=3, similarity_threshold=0.85, min_samples=3)
        )
        for _ in range(3):
            detector.record("error")
        with pytest.raises(SemanticLoopException):
            detector.check()

    def test_check_before_any_record(self):
        detector = SemanticLoopDetector()
        detector.check()  # no entries, no exception

    def test_reset_then_check(self):
        detector = SemanticLoopDetector(
            SemanticConfig(min_samples=2, similarity_threshold=0.85)
        )
        detector.record("same")
        detector.record("same")
        detector.reset()
        detector.check()  # reset cleared history, no exception


# ── Config integration ──────────────────────────────────────────


class TestSemanticConfig:

    def test_default_values(self):
        config = SemanticConfig()
        assert config.window_size == 5
        assert config.similarity_threshold == 0.85
        assert config.min_samples == 3

    def test_custom_values(self):
        config = SemanticConfig(window_size=10, similarity_threshold=0.70, min_samples=5)
        assert config.window_size == 10
        assert config.similarity_threshold == 0.70
        assert config.min_samples == 5
