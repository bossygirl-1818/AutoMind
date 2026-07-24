from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID

from app.rag.retrieval.semantic_search_service import (
    SemanticSearchError,
    SemanticSearchService,
)
from app.schemas.retrieval import (
    RetrievedChunk,
    SemanticSearchRequest,
)


class RetrievalEngineError(Exception):
    """
    Raised when the retrieval engine cannot produce a safe result set.
    """


@dataclass(frozen=True, slots=True)
class RetrievalEngineRequest:
    """
    Internal retrieval configuration used by AutoMind's RAG pipeline.

    This is intentionally separate from the public API schema. The API
    represents client-controlled input, while this model represents trusted
    retrieval behavior selected by AutoMind services and agents.
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
        normalized_query = " ".join(str(self.query).split())

        if len(normalized_query) < 2:
            raise ValueError(
                "Retrieval query must contain meaningful text."
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
class RetrievalEngineResult:
    """
    Internal result returned to context and citation builders.
    """

    project_id: UUID

    query: str

    collection_name: str

    chunks: tuple[RetrievedChunk, ...]

    requested_top_k: int

    candidate_count: int

    retrieval_time_ms: float

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError(
                "Retrieval result query cannot be empty."
            )

        if not self.collection_name.strip():
            raise ValueError(
                "Retrieval result collection name cannot be empty."
            )

        if self.requested_top_k < 1:
            raise ValueError(
                "requested_top_k must be greater than zero."
            )

        if self.candidate_count < 0:
            raise ValueError(
                "candidate_count cannot be negative."
            )

        if self.retrieval_time_ms < 0:
            raise ValueError(
                "retrieval_time_ms cannot be negative."
            )

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class RetrievalEngine:
    """
    Produces a high-quality retrieval set for AutoMind's Private RAG layer.

    The semantic-search service performs direct vector search. This engine
    applies retrieval policy above that low-level operation:

        1. Over-fetch candidate chunks.
        2. Remove duplicate chunk IDs.
        3. Optionally remove duplicate content.
        4. Limit excessive chunks from one document.
        5. Reassign deterministic final ranks.
        6. Return only the requested number of chunks.

    This prevents one large or repetitive engineering document from
    dominating the LLM context.
    """

    MAX_VECTOR_CANDIDATES = 50

    def __init__(
        self,
        *,
        semantic_search_service: SemanticSearchService,
    ) -> None:
        self._semantic_search_service = semantic_search_service

    def retrieve(
        self,
        *,
        project_id: UUID,
        collection_name: str,
        request: RetrievalEngineRequest,
    ) -> RetrievalEngineResult:
        """
        Retrieve and curate project-isolated engineering chunks.

        Project authorization and document ownership checks must be completed
        by the calling application service or API route before this method is
        invoked.
        """

        candidate_top_k = self._calculate_candidate_top_k(request)

        semantic_request = SemanticSearchRequest(
            query=request.query,
            top_k=candidate_top_k,
            score_threshold=request.score_threshold,
            document_ids=(
                list(request.document_ids)
                if request.document_ids
                else None
            ),
            include_metadata=request.include_metadata,
        )

        try:
            semantic_response = self._semantic_search_service.search(
                project_id=project_id,
                collection_name=collection_name,
                request=semantic_request,
            )

            curated_chunks = self._curate_chunks(
                chunks=semantic_response.results,
                top_k=request.top_k,
                max_chunks_per_document=(
                    request.max_chunks_per_document
                ),
                deduplicate_content=request.deduplicate_content,
            )

        except SemanticSearchError as exc:
            raise RetrievalEngineError(
                "The retrieval engine could not complete semantic search."
            ) from exc

        except RetrievalEngineError:
            raise

        except Exception as exc:
            raise RetrievalEngineError(
                "The retrieval engine could not produce a result set."
            ) from exc

        return RetrievalEngineResult(
            project_id=project_id,
            query=semantic_response.query,
            collection_name=semantic_response.collection_name,
            chunks=tuple(curated_chunks),
            requested_top_k=request.top_k,
            candidate_count=semantic_response.result_count,
            retrieval_time_ms=semantic_response.retrieval_time_ms,
        )

    def _calculate_candidate_top_k(
        self,
        request: RetrievalEngineRequest,
    ) -> int:
        """
        Calculate how many raw vector candidates should be requested.

        Over-fetching is necessary because deduplication and per-document
        limits may remove several raw results before final context assembly.
        """

        candidate_top_k = (
            request.top_k
            * request.candidate_multiplier
        )

        return min(
            max(candidate_top_k, request.top_k),
            self.MAX_VECTOR_CANDIDATES,
        )

    def _curate_chunks(
        self,
        *,
        chunks: list[RetrievedChunk],
        top_k: int,
        max_chunks_per_document: int | None,
        deduplicate_content: bool,
    ) -> list[RetrievedChunk]:
        """
        Apply deterministic retrieval-quality controls.

        Chunks remain in the semantic search order. This engine does not
        invent a new relevance score or perform heuristic score manipulation.
        """

        selected_chunks: list[RetrievedChunk] = []

        seen_chunk_ids: set[str] = set()

        seen_content_hashes: set[str] = set()

        document_chunk_counts: dict[UUID, int] = {}

        for chunk in chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue

            content_hash = self._content_hash(chunk.content)

            if (
                deduplicate_content
                and content_hash in seen_content_hashes
            ):
                continue

            current_document_count = document_chunk_counts.get(
                chunk.document_id,
                0,
            )

            if (
                max_chunks_per_document is not None
                and current_document_count
                >= max_chunks_per_document
            ):
                continue

            reranked_chunk = chunk.model_copy(
                update={
                    "rank": len(selected_chunks) + 1,
                }
            )

            selected_chunks.append(reranked_chunk)

            seen_chunk_ids.add(chunk.chunk_id)

            if deduplicate_content:
                seen_content_hashes.add(content_hash)

            document_chunk_counts[chunk.document_id] = (
                current_document_count + 1
            )

            if len(selected_chunks) >= top_k:
                break

        return selected_chunks

    @staticmethod
    def _content_hash(content: str) -> str:
        """
        Generate a stable fingerprint for duplicate-content detection.

        Whitespace is normalized before hashing so visually equivalent chunks
        are treated as duplicates even if their spacing differs.
        """

        normalized_content = " ".join(
            str(content).split()
        ).casefold()

        return sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()