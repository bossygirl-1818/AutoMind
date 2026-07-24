from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.rag.retrieval.citation_engine import (
    CitationBundle,
    CitationEngine,
    CitationEngineError,
)
from app.rag.retrieval.context_builder import (
    BuiltContext,
    ContextBuilder,
    ContextBuilderError,
)
from app.rag.retrieval.retrieval_engine import (
    RetrievalEngine,
    RetrievalEngineError,
    RetrievalEngineRequest,
    RetrievalEngineResult,
)


class PrivateRAGServiceError(Exception):
    """
    Raised when AutoMind cannot prepare private RAG evidence safely.
    """


@dataclass(frozen=True, slots=True)
class PrivateRAGRequest:
    """
    Internal request used to prepare project-specific RAG evidence.

    This request is separate from the public API contract because it controls
    trusted retrieval policies selected by AutoMind services and agents.
    """

    query: str

    top_k: int = 8

    score_threshold: float | None = None

    document_ids: tuple[UUID, ...] | None = None

    max_chunks_per_document: int | None = 4

    candidate_multiplier: int = 3

    deduplicate_content: bool = True

    include_metadata: bool = True

    def __post_init__(self) -> None:
        normalized_query = " ".join(
            str(self.query).split()
        )

        if len(normalized_query) < 2:
            raise ValueError(
                "Private RAG query must contain meaningful text."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.top_k > 50:
            raise ValueError(
                "top_k cannot exceed 50."
            )

        if (
            self.score_threshold is not None
            and not 0.0 <= self.score_threshold <= 1.0
        ):
            raise ValueError(
                "score_threshold must be between 0 and 1."
            )

        if (
            self.max_chunks_per_document is not None
            and self.max_chunks_per_document < 1
        ):
            raise ValueError(
                "max_chunks_per_document must be greater than zero."
            )

        if self.candidate_multiplier < 1:
            raise ValueError(
                "candidate_multiplier must be greater than zero."
            )

        if self.candidate_multiplier > 10:
            raise ValueError(
                "candidate_multiplier cannot exceed 10."
            )

        normalized_document_ids = (
            tuple(dict.fromkeys(self.document_ids))
            if self.document_ids
            else None
        )

        object.__setattr__(
            self,
            "query",
            normalized_query,
        )

        object.__setattr__(
            self,
            "document_ids",
            normalized_document_ids,
        )


@dataclass(frozen=True, slots=True)
class PrivateRAGResult:
    """
    Complete RAG evidence package prepared for an LLM or agent.

    This object contains:

        - curated retrieval results,
        - bounded engineering context,
        - authoritative citation registry.

    No LLM call is made in Phase 8. This result becomes the input to the
    future answer-generation and multi-agent orchestration phases.
    """

    project_id: UUID

    query: str

    collection_name: str

    retrieval: RetrievalEngineResult

    context: BuiltContext

    citations: CitationBundle

    def __post_init__(self) -> None:
        normalized_query = self.query.strip()
        normalized_collection_name = self.collection_name.strip()

        if not normalized_query:
            raise ValueError(
                "Private RAG result query cannot be empty."
            )

        if not normalized_collection_name:
            raise ValueError(
                "Private RAG collection name cannot be empty."
            )

        if self.retrieval.project_id != self.project_id:
            raise ValueError(
                "Retrieval result project does not match "
                "the Private RAG project."
            )

        if self.context.project_id != self.project_id:
            raise ValueError(
                "Built context project does not match "
                "the Private RAG project."
            )

        if self.citations.project_id != self.project_id:
            raise ValueError(
                "Citation bundle project does not match "
                "the Private RAG project."
            )

        if self.retrieval.query != normalized_query:
            raise ValueError(
                "Retrieval query does not match "
                "the Private RAG query."
            )

        if self.context.query != normalized_query:
            raise ValueError(
                "Context query does not match "
                "the Private RAG query."
            )

        if self.citations.query != normalized_query:
            raise ValueError(
                "Citation query does not match "
                "the Private RAG query."
            )

        if (
            self.context.included_chunk_count
            != self.citations.count
        ):
            raise ValueError(
                "Citation count must match the number of chunks "
                "included in the LLM context."
            )

        object.__setattr__(
            self,
            "query",
            normalized_query,
        )

        object.__setattr__(
            self,
            "collection_name",
            normalized_collection_name,
        )

    @property
    def has_context(self) -> bool:
        return not self.context.is_empty_retrieval

    @property
    def retrieved_chunk_count(self) -> int:
        return self.retrieval.chunk_count

    @property
    def included_chunk_count(self) -> int:
        return self.context.included_chunk_count

    @property
    def citation_count(self) -> int:
        return self.citations.count


class PrivateRAGService:
    """
    Coordinates the complete Phase 8 Private RAG evidence pipeline.

    Workflow:

        authenticated project query
            -> semantic retrieval
            -> retrieval curation
            -> bounded context construction
            -> citation registry generation

    This service intentionally does not:

        - authenticate users,
        - authorize project membership,
        - validate database document ownership,
        - call an LLM,
        - persist chat messages.

    Those responsibilities belong to the API, secure-workspace, chat, and
    future LLM orchestration layers.
    """

    def __init__(
        self,
        *,
        retrieval_engine: RetrievalEngine,
        context_builder: ContextBuilder,
        citation_engine: CitationEngine,
    ) -> None:
        self._retrieval_engine = retrieval_engine
        self._context_builder = context_builder
        self._citation_engine = citation_engine

    def prepare(
        self,
        *,
        project_id: UUID,
        collection_name: str,
        request: PrivateRAGRequest,
    ) -> PrivateRAGResult:
        """
        Prepare private, project-isolated evidence for one engineering query.

        The caller must complete authentication, project authorization, and
        optional document ownership validation before invoking this method.
        """

        normalized_collection_name = (
            self._normalize_collection_name(
                collection_name
            )
        )

        retrieval_request = RetrievalEngineRequest(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_ids=request.document_ids,
            max_chunks_per_document=(
                request.max_chunks_per_document
            ),
            candidate_multiplier=request.candidate_multiplier,
            deduplicate_content=request.deduplicate_content,
            include_metadata=request.include_metadata,
        )

        try:
            retrieval_result = (
                self._retrieval_engine.retrieve(
                    project_id=project_id,
                    collection_name=normalized_collection_name,
                    request=retrieval_request,
                )
            )

            built_context = self._context_builder.build(
                retrieval_result
            )

            citation_bundle = self._citation_engine.build(
                built_context
            )

            self._validate_pipeline_consistency(
                retrieval_result=retrieval_result,
                built_context=built_context,
                citation_bundle=citation_bundle,
            )

        except RetrievalEngineError as exc:
            raise PrivateRAGServiceError(
                "Private RAG retrieval could not be completed."
            ) from exc

        except ContextBuilderError as exc:
            raise PrivateRAGServiceError(
                "Retrieved evidence could not be converted "
                "into LLM context."
            ) from exc

        except CitationEngineError as exc:
            raise PrivateRAGServiceError(
                "The citation registry could not be generated."
            ) from exc

        except PrivateRAGServiceError:
            raise

        except Exception as exc:
            raise PrivateRAGServiceError(
                "Private RAG evidence preparation failed."
            ) from exc

        return PrivateRAGResult(
            project_id=project_id,
            query=request.query,
            collection_name=normalized_collection_name,
            retrieval=retrieval_result,
            context=built_context,
            citations=citation_bundle,
        )

    @staticmethod
    def _validate_pipeline_consistency(
        *,
        retrieval_result: RetrievalEngineResult,
        built_context: BuiltContext,
        citation_bundle: CitationBundle,
    ) -> None:
        """
        Verify consistency across retrieval, context, and citation layers.

        These checks prevent downstream agents from receiving mismatched
        evidence packages due to programming errors or corrupted state.
        """

        if (
            retrieval_result.project_id
            != built_context.project_id
        ):
            raise PrivateRAGServiceError(
                "Retrieval and context project identifiers do not match."
            )

        if (
            built_context.project_id
            != citation_bundle.project_id
        ):
            raise PrivateRAGServiceError(
                "Context and citation project identifiers do not match."
            )

        if retrieval_result.query != built_context.query:
            raise PrivateRAGServiceError(
                "Retrieval and context queries do not match."
            )

        if built_context.query != citation_bundle.query:
            raise PrivateRAGServiceError(
                "Context and citation queries do not match."
            )

        if (
            built_context.included_chunk_count
            != citation_bundle.count
        ):
            raise PrivateRAGServiceError(
                "Citation count does not match the number of "
                "context sources."
            )

        context_chunk_ids = tuple(
            source.chunk_id
            for source in built_context.sources
        )

        citation_chunk_ids = tuple(
            citation.chunk_id
            for citation in citation_bundle.citations
        )

        if context_chunk_ids != citation_chunk_ids:
            raise PrivateRAGServiceError(
                "Citation order does not match the context source order."
            )

    @staticmethod
    def _normalize_collection_name(
        collection_name: str,
    ) -> str:
        normalized = str(collection_name).strip().lower()

        if not normalized:
            raise ValueError(
                "Collection name cannot be empty."
            )

        if len(normalized) < 3:
            raise ValueError(
                "Collection name must contain at least "
                "three characters."
            )

        if len(normalized) > 512:
            raise ValueError(
                "Collection name cannot exceed 512 characters."
            )

        return normalized