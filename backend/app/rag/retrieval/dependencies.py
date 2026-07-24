from __future__ import annotations

from functools import lru_cache

from app.rag.embeddings.local_provider import (
    local_embedding_provider,
)
from app.rag.vectorstore.chroma_store import (
    chroma_vector_store,
)

from app.rag.retrieval.semantic_search_service import (
    SemanticSearchService,
)
from app.rag.retrieval.retrieval_engine import (
    RetrievalEngine,
)
from app.rag.retrieval.context_builder import (
    ContextBuilder,
    ContextBuilderConfig,
)
from app.rag.retrieval.citation_engine import (
    CitationEngine,
)
from app.rag.retrieval.private_rag_service import (
    PrivateRAGService,
)


@lru_cache(maxsize=1)
def get_semantic_search_service() -> SemanticSearchService:
    """
    Singleton semantic-search service.

    Uses the shared local embedding provider and the shared ChromaDB
    vector store.
    """

    return SemanticSearchService(
        embedding_provider=local_embedding_provider,
        vector_store=chroma_vector_store,
    )


@lru_cache(maxsize=1)
def get_retrieval_engine() -> RetrievalEngine:
    """
    Singleton retrieval engine.
    """

    return RetrievalEngine(
        semantic_search_service=get_semantic_search_service(),
    )


@lru_cache(maxsize=1)
def get_context_builder() -> ContextBuilder:
    """
    Singleton context builder.
    """

    return ContextBuilder(
        config=ContextBuilderConfig(),
    )


@lru_cache(maxsize=1)
def get_citation_engine() -> CitationEngine:
    """
    Singleton citation engine.
    """

    return CitationEngine()


@lru_cache(maxsize=1)
def get_private_rag_service() -> PrivateRAGService:
    """
    Main dependency exposed to the API layer.

    Future supervisor agents, LangGraph workflows,
    evaluation pipelines, and chat services should
    all resolve the Private RAG service through this
    provider instead of constructing their own instance.
    """

    return PrivateRAGService(
        retrieval_engine=get_retrieval_engine(),
        context_builder=get_context_builder(),
        citation_engine=get_citation_engine(),
    )