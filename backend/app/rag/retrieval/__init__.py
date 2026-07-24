"""
Private retrieval components for AutoMind's project-isolated RAG pipeline.
"""

from app.rag.retrieval.semantic_search_service import (
    SemanticSearchError,
    SemanticSearchService,
)

__all__ = (
    "SemanticSearchError",
    "SemanticSearchService",
)