"""KnowledgeSearchTool — BaseTool wrapper for RAG queries.

Integrates the RAG pipeline into the ``ToolRegistry`` so that
agents and reviewers can query domain knowledge during execution.
"""

from __future__ import annotations

from typing import Any

from kappa.rag.manager import RAGManager
from kappa.tools.registry import ToolResult


class KnowledgeSearchTool:
    """Search the domain knowledge base via RAG.

    Implements the ``BaseTool`` protocol for ``ToolRegistry`` integration.

    Args:
        rag_manager: The ``RAGManager`` to query.
        top_k: Default number of results per query (overridable at call time).
    """

    def __init__(self, rag_manager: RAGManager, top_k: int = 5) -> None:
        self._rag = rag_manager
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return (
            "Search the domain knowledge base for relevant information. "
            "Pass a 'query' string to retrieve the most relevant document chunks."
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a knowledge search.

        Keyword Args:
            query: The search query string (required).
            top_k: Optional override for number of results.

        Returns:
            ``ToolResult`` with formatted search results or an error.
        """
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(
                success=False,
                output="",
                error="Missing required argument: 'query'",
            )

        top_k = kwargs.get("top_k", self._top_k)

        try:
            results = self._rag.query(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output="",
                error=f"RAG query failed: {exc}",
            )

        if not results:
            return ToolResult(
                success=True,
                output="No relevant documents found.",
            )

        # Format results as readable text
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            score = r.get("score", 0.0)
            source = r.get("metadata", {}).get("source", "unknown")
            text = r.get("document", "")
            parts.append(
                f"[{i}] (score={score:.3f}, source={source})\n{text}"
            )

        return ToolResult(success=True, output="\n\n".join(parts))
