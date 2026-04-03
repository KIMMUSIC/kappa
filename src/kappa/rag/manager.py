"""RAG manager — document ingestion and semantic retrieval.

Chunks documents, embeds them via a pluggable ``EmbeddingProvider``,
stores vectors in a pluggable ``VectorStore``, and answers queries
with ranked context snippets.

All components are protocol-based for testability and extensibility.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kappa.config import RAGConfig


# ── Protocols ──────────────────────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces dense vector embeddings from text."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores and retrieves vectors by similarity."""

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert documents with their embeddings."""
        ...

    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar documents.

        Each result dict contains ``id``, ``document``, ``score``,
        and optional ``metadata``.
        """
        ...

    @property
    def count(self) -> int:
        """Number of documents stored."""
        ...


# ── In-Memory Vector Store ─────────────────────────────────────


@dataclass
class _VectorEntry:
    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity.

    Suitable for testing, prototyping, and small-to-medium corpora.
    For production workloads, swap in ChromaDB or FAISS via the
    ``VectorStore`` protocol.
    """

    def __init__(self) -> None:
        self._entries: list[_VectorEntry] = []

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metas = metadatas or [{} for _ in ids]
        for doc_id, emb, doc, meta in zip(ids, embeddings, documents, metas):
            self._entries.append(
                _VectorEntry(id=doc_id, embedding=emb, document=doc, metadata=meta)
            )

    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, _VectorEntry]] = []
        for entry in self._entries:
            sim = _cosine_similarity(embedding, entry.embedding)
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, entry in scored[:top_k]:
            results.append({
                "id": entry.id,
                "document": entry.document,
                "score": score,
                "metadata": entry.metadata,
            })
        return results

    @property
    def count(self) -> int:
        return len(self._entries)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Chunker ────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into fixed-size character chunks with overlap.

    Args:
        text: Source text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.  Empty input returns an empty list.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if overlap >= chunk_size:
            # Prevent infinite loop from bad config
            start = end

    return chunks


# ── RAG Manager ────────────────────────────────────────────────


class RAGManager:
    """Orchestrates document ingestion and semantic retrieval.

    Args:
        embedding_provider: Produces vector embeddings from text.
        store: Vector store backend.  Defaults to ``InMemoryVectorStore``.
        config: RAG pipeline configuration.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: VectorStore | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._embedder = embedding_provider
        self._store: VectorStore = store or InMemoryVectorStore()
        self._config = config or RAGConfig()

    @property
    def document_count(self) -> int:
        """Number of chunks currently indexed."""
        return self._store.count

    def ingest(self, text: str, source: str = "") -> int:
        """Chunk, embed, and index a document.

        Args:
            text: Raw document content.
            source: Optional source identifier (filename, URL, etc.).

        Returns:
            Number of chunks indexed.
        """
        chunks = chunk_text(
            text,
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )
        if not chunks:
            return 0

        # Generate deterministic IDs
        ids = [
            hashlib.sha256(f"{source}:{i}:{c[:64]}".encode()).hexdigest()[:16]
            for i, c in enumerate(chunks)
        ]

        embeddings = self._embedder.embed(chunks)

        metadatas = [
            {"source": source, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self._store.add(ids, embeddings, chunks, metadatas)
        return len(chunks)

    def ingest_file(self, path: str, encoding: str = "utf-8") -> int:
        """Read a file and ingest its content.

        Args:
            path: Path to the file to ingest.
            encoding: File encoding.

        Returns:
            Number of chunks indexed.
        """
        with open(path, encoding=encoding) as f:
            text = f.read()
        return self.ingest(text, source=path)

    def query(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Search the knowledge base for relevant chunks.

        Args:
            question: Natural language query.
            top_k: Number of results (defaults to ``RAGConfig.top_k``).

        Returns:
            Ranked list of result dicts with ``id``, ``document``,
            ``score``, and ``metadata``.
        """
        k = top_k if top_k is not None else self._config.top_k
        query_emb = self._embedder.embed([question])[0]
        results = self._store.query(query_emb, top_k=k)

        # Filter by minimum score
        if self._config.min_score > 0.0:
            results = [r for r in results if r["score"] >= self._config.min_score]

        return results
