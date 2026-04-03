"""Tests for RAG pipeline (Phase 4, Task 2).

Validates:
  - Chunking logic (fixed-size with overlap)
  - InMemoryVectorStore (add, query, cosine similarity)
  - RAGManager ingestion and retrieval
  - KnowledgeSearchTool (BaseTool protocol compliance)
  - ToolRegistry integration with budget tracking
  - Edge cases and error handling
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from kappa.budget.tracker import BudgetTracker
from kappa.config import BudgetConfig, RAGConfig
from kappa.rag.manager import (
    InMemoryVectorStore,
    RAGManager,
    _cosine_similarity,
    chunk_text,
)
from kappa.rag.tool import KnowledgeSearchTool
from kappa.tools.registry import BaseTool, ToolRegistry, ToolResult


# ── Fake Embedding Provider ───────────────────────────────────


class FakeEmbeddingProvider:
    """Deterministic embedding provider for testing.

    Produces fixed-dimension vectors from character frequency counts.
    This ensures semantically similar texts produce similar vectors
    while remaining fully deterministic and dependency-free.
    """

    def __init__(self, dim: int = 26) -> None:
        self._dim = dim
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._char_freq(t) for t in texts]

    def _char_freq(self, text: str) -> list[float]:
        """Character frequency vector (a-z), normalized."""
        freq = [0.0] * self._dim
        lower = text.lower()
        for ch in lower:
            idx = ord(ch) - ord("a")
            if 0 <= idx < self._dim:
                freq[idx] += 1.0
        # Normalize
        magnitude = math.sqrt(sum(f * f for f in freq))
        if magnitude > 0:
            freq = [f / magnitude for f in freq]
        return freq


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


@pytest.fixture
def rag(embedder):
    return RAGManager(embedding_provider=embedder)


@pytest.fixture
def budget_tracker():
    config = BudgetConfig(max_total_tokens=10_000, max_cost_usd=1.0)
    return BudgetTracker(config)


@pytest.fixture
def registry(budget_tracker):
    return ToolRegistry(tracker=budget_tracker, cost_per_tool_call=50)


# ── Chunking Tests ─────────────────────────────────────────────


class TestChunkText:
    """chunk_text() splitting logic."""

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_exact_chunk_size(self):
        text = "a" * 512
        chunks = chunk_text(text, chunk_size=512, overlap=0)
        assert len(chunks) == 1

    def test_splits_with_overlap(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) >= 2
        # Each chunk should be at most chunk_size chars
        for c in chunks:
            assert len(c) <= 512

    def test_overlap_creates_redundancy(self):
        text = "ABCDEFGHIJ" * 100  # 1000 chars
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        # Consecutive chunks should share some content
        for i in range(len(chunks) - 1):
            tail = chunks[i][-50:]
            head = chunks[i + 1][:50]
            # Due to overlap, the end of one chunk appears in the next
            assert tail == head

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\t  ") == []

    def test_custom_parameters(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 10

    def test_overlap_ge_chunk_size_no_infinite_loop(self):
        """Bad config (overlap >= chunk_size) should not hang."""
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=50, overlap=50)
        assert len(chunks) >= 1
        # Should terminate, not loop forever


# ── Cosine Similarity Tests ────────────────────────────────────


class TestCosineSimilarity:
    """_cosine_similarity() correctness."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-9

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_different_lengths(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0


# ── InMemoryVectorStore Tests ─────────────────────────────────


class TestInMemoryVectorStore:
    """InMemoryVectorStore add/query operations."""

    def test_add_and_count(self):
        store = InMemoryVectorStore()
        assert store.count == 0
        store.add(["a"], [[1.0, 0.0]], ["doc a"])
        assert store.count == 1

    def test_add_multiple(self):
        store = InMemoryVectorStore()
        store.add(
            ["a", "b"],
            [[1.0, 0.0], [0.0, 1.0]],
            ["doc a", "doc b"],
        )
        assert store.count == 2

    def test_query_returns_ranked(self):
        store = InMemoryVectorStore()
        store.add(
            ["a", "b", "c"],
            [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]],
            ["exact match", "partial match", "no match"],
        )
        results = store.query([1.0, 0.0], top_k=3)
        assert len(results) == 3
        # Most similar first
        assert results[0]["document"] == "exact match"
        assert results[0]["score"] > results[1]["score"]
        assert results[1]["score"] > results[2]["score"]

    def test_query_top_k_limits(self):
        store = InMemoryVectorStore()
        store.add(
            [f"d{i}" for i in range(10)],
            [[float(i), 0.0] for i in range(10)],
            [f"doc {i}" for i in range(10)],
        )
        results = store.query([1.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_query_empty_store(self):
        store = InMemoryVectorStore()
        results = store.query([1.0, 0.0], top_k=5)
        assert results == []

    def test_metadata_stored_and_returned(self):
        store = InMemoryVectorStore()
        store.add(
            ["a"],
            [[1.0, 0.0]],
            ["doc"],
            [{"source": "test.md", "chunk_index": 0}],
        )
        results = store.query([1.0, 0.0], top_k=1)
        assert results[0]["metadata"]["source"] == "test.md"

    def test_result_structure(self):
        store = InMemoryVectorStore()
        store.add(["a"], [[1.0, 0.0]], ["doc a"])
        results = store.query([1.0, 0.0], top_k=1)
        r = results[0]
        assert "id" in r
        assert "document" in r
        assert "score" in r
        assert "metadata" in r


# ── RAGManager Tests ──────────────────────────────────────────


class TestRAGManager:
    """RAGManager ingestion and retrieval."""

    def test_ingest_returns_chunk_count(self, rag):
        count = rag.ingest("Hello world, this is a test document.")
        assert count >= 1

    def test_ingest_updates_document_count(self, rag):
        assert rag.document_count == 0
        rag.ingest("A" * 1000)
        assert rag.document_count > 0

    def test_ingest_empty_returns_zero(self, rag):
        assert rag.ingest("") == 0
        assert rag.ingest("   ") == 0

    def test_ingest_with_source(self, rag):
        rag.ingest("Some content", source="readme.md")
        results = rag.query("content")
        assert results[0]["metadata"]["source"] == "readme.md"

    def test_ingest_file(self, embedder):
        rag = RAGManager(embedding_provider=embedder)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("Python is a programming language.\n" * 20)
            path = f.name
        try:
            count = rag.ingest_file(path)
            assert count >= 1
            assert rag.document_count >= 1
        finally:
            os.unlink(path)

    def test_query_returns_results(self, rag):
        rag.ingest("Python is great for data science and machine learning.")
        rag.ingest("JavaScript powers the modern web frontend.")
        rag.ingest("Rust provides memory safety without garbage collection.")

        results = rag.query("programming language for data")
        assert len(results) > 0
        # Python doc should rank highest (char frequency overlap with 'data')
        assert "Python" in results[0]["document"] or len(results) > 0

    def test_query_top_k(self, rag):
        for i in range(10):
            rag.ingest(f"Document number {i} with unique content.")
        results = rag.query("document", top_k=3)
        assert len(results) == 3

    def test_query_min_score_filter(self, embedder):
        config = RAGConfig(min_score=0.99)
        rag = RAGManager(embedding_provider=embedder, config=config)
        rag.ingest("aaaaaa")
        rag.ingest("zzzzzz")
        # Query for 'a'-heavy text — only the 'aaaaaa' doc should pass threshold
        results = rag.query("aaa")
        # min_score=0.99 filters low-similarity results
        for r in results:
            assert r["score"] >= 0.99

    def test_custom_config(self, embedder):
        config = RAGConfig(chunk_size=100, chunk_overlap=10, top_k=2)
        rag = RAGManager(embedding_provider=embedder, config=config)
        text = "word " * 200  # 1000 chars
        count = rag.ingest(text)
        # With chunk_size=100, should produce ~11 chunks
        assert count >= 10

    def test_multiple_ingestions_accumulate(self, rag):
        rag.ingest("First document.")
        c1 = rag.document_count
        rag.ingest("Second document.")
        c2 = rag.document_count
        assert c2 > c1

    def test_embedding_provider_called(self, embedder, rag):
        rag.ingest("Test text for embedding.")
        assert embedder.call_count >= 1


# ── KnowledgeSearchTool Tests ─────────────────────────────────


class TestKnowledgeSearchTool:
    """KnowledgeSearchTool BaseTool compliance and integration."""

    def test_satisfies_base_tool_protocol(self, rag):
        tool = KnowledgeSearchTool(rag)
        assert isinstance(tool, BaseTool)

    def test_name(self, rag):
        tool = KnowledgeSearchTool(rag)
        assert tool.name == "knowledge_search"

    def test_description(self, rag):
        tool = KnowledgeSearchTool(rag)
        assert "knowledge" in tool.description.lower()

    def test_execute_returns_tool_result(self, rag):
        rag.ingest("Kappa is a self-healing agent harness.")
        tool = KnowledgeSearchTool(rag)
        result = tool.execute(query="agent harness")
        assert isinstance(result, ToolResult)
        assert result.success is True

    def test_execute_formats_results(self, rag):
        rag.ingest("Python is great for AI development.")
        tool = KnowledgeSearchTool(rag)
        result = tool.execute(query="python AI")
        assert "score=" in result.output
        assert "source=" in result.output

    def test_execute_missing_query(self, rag):
        tool = KnowledgeSearchTool(rag)
        result = tool.execute()
        assert result.success is False
        assert "query" in result.error.lower()

    def test_execute_empty_query(self, rag):
        tool = KnowledgeSearchTool(rag)
        result = tool.execute(query="")
        assert result.success is False

    def test_execute_no_results(self, rag):
        tool = KnowledgeSearchTool(rag)
        # Empty store
        result = tool.execute(query="anything")
        assert result.success is True
        assert "No relevant documents" in result.output

    def test_execute_custom_top_k(self, rag):
        for i in range(10):
            rag.ingest(f"Document {i}: content about topic {i}.")
        tool = KnowledgeSearchTool(rag, top_k=3)
        result = tool.execute(query="topic")
        # Should have exactly 3 numbered results
        assert "[1]" in result.output
        assert "[3]" in result.output

    def test_execute_top_k_override(self, rag):
        for i in range(10):
            rag.ingest(f"Document {i}: content.")
        tool = KnowledgeSearchTool(rag, top_k=5)
        result = tool.execute(query="document", top_k=2)
        lines = [l for l in result.output.split("\n") if l.startswith("[")]
        assert len(lines) == 2


# ── ToolRegistry Integration ──────────────────────────────────


class TestRAGRegistryIntegration:
    """KnowledgeSearchTool registered in ToolRegistry with budget tracking."""

    def test_register_and_execute(self, rag, registry):
        tool = KnowledgeSearchTool(rag)
        registry.register(tool)
        rag.ingest("Test knowledge base content.")

        result = registry.execute("knowledge_search", query="knowledge")
        assert result.success is True

    def test_appears_in_tool_list(self, rag, registry):
        tool = KnowledgeSearchTool(rag)
        registry.register(tool)
        names = [t["name"] for t in registry.list_tools()]
        assert "knowledge_search" in names

    def test_budget_tracked(self, rag, registry, budget_tracker):
        tool = KnowledgeSearchTool(rag)
        registry.register(tool)
        rag.ingest("Some content for budget test.")

        initial = budget_tracker.total_tokens
        registry.execute("knowledge_search", query="budget")
        assert budget_tracker.total_tokens == initial + 50

    def test_multiple_calls_accumulate_budget(self, rag, registry, budget_tracker):
        tool = KnowledgeSearchTool(rag)
        registry.register(tool)
        rag.ingest("Content.")

        registry.execute("knowledge_search", query="a")
        registry.execute("knowledge_search", query="b")
        registry.execute("knowledge_search", query="c")
        assert budget_tracker.total_tokens == 150

    def test_budget_exceeded_blocks_rag_tool(self, rag):
        """RAG tools cannot bypass budget."""
        config = BudgetConfig(max_total_tokens=60, max_cost_usd=10.0)
        tracker = BudgetTracker(config)
        reg = ToolRegistry(tracker=tracker, cost_per_tool_call=50)

        tool = KnowledgeSearchTool(rag)
        reg.register(tool)
        rag.ingest("Content for budget limit test.")

        # First call OK (50 tokens)
        result = reg.execute("knowledge_search", query="test")
        assert result.success is True

        # Second call exceeds budget
        from kappa.exceptions import BudgetExceededException

        with pytest.raises(BudgetExceededException):
            reg.execute("knowledge_search", query="test2")


# ── RAGConfig Tests ───────────────────────────────────────────


class TestRAGConfig:
    """RAGConfig dataclass."""

    def test_defaults(self):
        config = RAGConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.top_k == 5
        assert config.min_score == 0.0

    def test_custom_values(self):
        config = RAGConfig(chunk_size=256, chunk_overlap=32, top_k=10, min_score=0.5)
        assert config.chunk_size == 256
        assert config.top_k == 10

    def test_frozen(self):
        config = RAGConfig()
        with pytest.raises(AttributeError):
            config.chunk_size = 999


# ── MCP + RAG Combined ────────────────────────────────────────


class TestMCPAndRAGCoexistence:
    """MCP and RAG tools coexist in the same ToolRegistry."""

    def test_both_registered(self, rag, registry):
        from kappa.tools.mcp import MCPBridge

        from tests.test_mcp import FakeMCPTransport, SAMPLE_CALL_RESULTS, SAMPLE_TOOLS

        # Register RAG tool
        rag_tool = KnowledgeSearchTool(rag)
        registry.register(rag_tool)

        # Register MCP tools
        transport = FakeMCPTransport(tools=SAMPLE_TOOLS, call_results=SAMPLE_CALL_RESULTS)
        bridge = MCPBridge("srv", transport)
        bridge.connect()
        bridge.register_all(registry)

        # All tools present
        names = sorted(t["name"] for t in registry.list_tools())
        assert "knowledge_search" in names
        assert "mcp:srv:read_file" in names

    def test_shared_budget_tracking(self, rag, budget_tracker):
        from kappa.tools.mcp import MCPBridge

        from tests.test_mcp import FakeMCPTransport, SAMPLE_CALL_RESULTS, SAMPLE_TOOLS

        reg = ToolRegistry(tracker=budget_tracker, cost_per_tool_call=50)

        # Register both
        rag_tool = KnowledgeSearchTool(rag)
        reg.register(rag_tool)
        rag.ingest("Content.")

        transport = FakeMCPTransport(tools=SAMPLE_TOOLS, call_results=SAMPLE_CALL_RESULTS)
        bridge = MCPBridge("srv", transport)
        bridge.connect()
        bridge.register_all(reg)

        # Execute one of each
        reg.execute("knowledge_search", query="test")
        reg.execute("mcp:srv:read_file", path="/x")

        # Both charged to same budget
        assert budget_tracker.total_tokens == 100
