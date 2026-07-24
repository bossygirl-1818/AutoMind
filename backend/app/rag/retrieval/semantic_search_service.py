from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import UUID

from app.rag.embeddings import BaseEmbeddingProvider
from app.rag.vectorstore import (
    BaseVectorStore,
    SearchResult,
    VectorStoreError,
)
from app.schemas.retrieval import (
    RetrievedChunk,
    RetrievedChunkMetadata,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


class SemanticSearchError(Exception):
    """
    Raised when project-scoped semantic retrieval cannot complete safely.
    """


class SemanticSearchService:
    """
    Performs private, project-isolated semantic search.

    Responsibilities:

        1. Validate the project and collection identifiers.
        2. Generate a private local query embedding.
        3. Build project-scoped ChromaDB filters.
        4. Perform vector similarity search.
        5. Normalize vector distances into relevance scores.
        6. Convert raw vector-store results into API-safe schemas.

    This service is independent of FastAPI and SQLAlchemy. It can therefore
    be reused later by API routes, LangGraph agents, background jobs, and
    the full Private RAG orchestration pipeline.
    """

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def search(
        self,
        *,
        project_id: UUID,
        collection_name: str,
        request: SemanticSearchRequest,
    ) -> SemanticSearchResponse:
        """
        Search one project-isolated vector collection.

        The caller must authorize the authenticated user's access to the
        project before invoking this method.

        Project isolation is enforced twice:

            1. The caller supplies the project's dedicated collection.
            2. Every ChromaDB query includes a project_id metadata filter.

        This defense-in-depth approach prevents accidental cross-project
        retrieval even if an incorrect collection is supplied.
        """

        normalized_collection_name = self._normalize_collection_name(
            collection_name
        )

        started_at = perf_counter()

        try:
            self._ensure_collection_exists(normalized_collection_name)

            query_embedding = self._embedding_provider.embed_query(
                request.query
            )

            where_filter = self._build_where_filter(
                project_id=project_id,
                document_ids=request.document_ids,
            )

            raw_results = self._vector_store.search(
                collection_name=normalized_collection_name,
                embedding=query_embedding,
                top_k=request.top_k,
                where=where_filter,
            )

            retrieved_chunks = self._build_retrieved_chunks(
                results=raw_results,
                expected_project_id=project_id,
                score_threshold=request.score_threshold,
                include_metadata=request.include_metadata,
            )

        except SemanticSearchError:
            raise

        except VectorStoreError as exc:
            raise SemanticSearchError(
                "The semantic-search vector store could not complete "
                "the retrieval request."
            ) from exc

        except Exception as exc:
            raise SemanticSearchError(
                "Semantic search could not be completed."
            ) from exc

        retrieval_time_ms = round(
            (perf_counter() - started_at) * 1000,
            3,
        )

        return SemanticSearchResponse(
            project_id=project_id,
            query=request.query,
            results=retrieved_chunks,
            result_count=len(retrieved_chunks),
            collection_name=normalized_collection_name,
            retrieval_time_ms=retrieval_time_ms,
            retrieved_at=datetime.now(timezone.utc),
        )

    def _ensure_collection_exists(
        self,
        collection_name: str,
    ) -> None:
        """
        Fail explicitly when the project collection has not been created.

        An absent collection usually means that the project has no indexed
        documents yet, or that indexing did not complete successfully.
        """

        if not self._vector_store.collection_exists(collection_name):
            raise SemanticSearchError(
                f"Vector collection '{collection_name}' does not exist. "
                "The project may not contain indexed documents yet."
            )

    @staticmethod
    def _build_where_filter(
        *,
        project_id: UUID,
        document_ids: list[UUID] | None,
    ) -> dict[str, Any]:
        """
        Build a ChromaDB-compatible metadata filter.

        Every search includes project_id, even though projects use isolated
        collections. Optional document filtering is combined using `$and`.
        """

        project_filter: dict[str, Any] = {
            "project_id": str(project_id),
        }

        if not document_ids:
            return project_filter

        document_filter: dict[str, Any] = {
            "document_id": {
                "$in": [
                    str(document_id)
                    for document_id in document_ids
                ]
            }
        }

        return {
            "$and": [
                project_filter,
                document_filter,
            ]
        }

    def _build_retrieved_chunks(
        self,
        *,
        results: list[SearchResult],
        expected_project_id: UUID,
        score_threshold: float | None,
        include_metadata: bool,
    ) -> list[RetrievedChunk]:
        """
        Convert raw vector-store output into validated retrieval results.

        Results are returned in the vector store's relevance order. Rank is
        assigned only after invalid or below-threshold results are removed.
        """

        retrieved_chunks: list[RetrievedChunk] = []

        for result in results:
            self._validate_result_project(
                result=result,
                expected_project_id=expected_project_id,
            )

            relevance_score = self._distance_to_relevance_score(
                result.score
            )

            if (
                score_threshold is not None
                and relevance_score < score_threshold
            ):
                continue

            document_id = self._parse_document_id(result)

            metadata = (
                self._build_metadata(result.metadata)
                if include_metadata
                else None
            )

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=result.id,
                    document_id=document_id,
                    content=result.document,
                    rank=len(retrieved_chunks) + 1,
                    distance=result.score,
                    relevance_score=relevance_score,
                    metadata=metadata,
                )
            )

        return retrieved_chunks

    @staticmethod
    def _validate_result_project(
        *,
        result: SearchResult,
        expected_project_id: UUID,
    ) -> None:
        """
        Reject any result whose metadata does not match the requested project.

        This validation protects against malformed legacy records, incorrect
        collection selection, and metadata corruption.
        """

        stored_project_id = str(
            result.metadata.get("project_id", "")
        ).strip()

        if not stored_project_id:
            raise SemanticSearchError(
                f"Retrieved chunk '{result.id}' is missing project metadata."
            )

        if stored_project_id != str(expected_project_id):
            raise SemanticSearchError(
                "Cross-project retrieval was blocked because a retrieved "
                "chunk did not belong to the requested project."
            )

    @staticmethod
    def _parse_document_id(
        result: SearchResult,
    ) -> UUID:
        raw_document_id = str(
            result.metadata.get("document_id", "")
        ).strip()

        if not raw_document_id:
            raise SemanticSearchError(
                f"Retrieved chunk '{result.id}' is missing document metadata."
            )

        try:
            return UUID(raw_document_id)

        except ValueError as exc:
            raise SemanticSearchError(
                f"Retrieved chunk '{result.id}' contains an invalid "
                "document identifier."
            ) from exc

    @classmethod
    def _build_metadata(
        cls,
        raw_metadata: dict[str, Any],
    ) -> RetrievedChunkMetadata:
        """
        Map Phase 7 indexing metadata into the public retrieval schema.

        Phase 7 stores:
            source_filename -> API original_filename
            heading         -> API section_title
        """

        known_keys = {
            "project_id",
            "document_id",
            "chunk_index",
            "page_number",
            "heading",
            "source_filename",
            "file_extension",
            "document_type",
        }

        source_filename = cls._optional_string(
            raw_metadata.get("source_filename")
        )

        heading = cls._optional_string(
            raw_metadata.get("heading")
        )

        page_number = cls._optional_positive_integer(
            raw_metadata.get("page_number")
        )

        chunk_index = cls._optional_non_negative_integer(
            raw_metadata.get("chunk_index")
        )

        source_reference = cls._build_source_reference(
            page_number=page_number,
            heading=heading,
        )

        additional = {
            str(key): value
            for key, value in raw_metadata.items()
            if key not in known_keys
        }

        return RetrievedChunkMetadata(
            original_filename=source_filename,
            document_type=cls._optional_string(
                raw_metadata.get("document_type")
            ),
            file_extension=cls._optional_string(
                raw_metadata.get("file_extension")
            ),
            chunk_index=chunk_index,
            page_number=page_number,
            section_title=heading,
            source_reference=source_reference,
            additional=additional,
        )

    @staticmethod
    def _build_source_reference(
        *,
        page_number: int | None,
        heading: str | None,
    ) -> str | None:
        reference_parts: list[str] = []

        if page_number is not None:
            reference_parts.append(f"Page {page_number}")

        if heading:
            reference_parts.append(f"Section: {heading}")

        if not reference_parts:
            return None

        return " | ".join(reference_parts)

    @staticmethod
    def _distance_to_relevance_score(
        distance: float,
    ) -> float:
        """
        Convert a non-negative vector distance into a normalized score.

        ChromaDB returns smaller values for more similar vectors. The
        transformation below is monotonic and vector-metric independent:

            relevance = 1 / (1 + distance)

        Therefore:
            distance 0.0 -> relevance 1.0
            distance 1.0 -> relevance 0.5
            distance 3.0 -> relevance 0.25

        This avoids incorrectly assuming that every configured collection
        uses cosine distance.
        """

        normalized_distance = max(float(distance), 0.0)
        relevance_score = 1.0 / (1.0 + normalized_distance)

        return round(
            min(max(relevance_score, 0.0), 1.0),
            6,
        )

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()

        return normalized or None

    @staticmethod
    def _optional_positive_integer(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            parsed_value = int(value)
        except (TypeError, ValueError):
            return None

        return parsed_value if parsed_value >= 1 else None

    @staticmethod
    def _optional_non_negative_integer(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            parsed_value = int(value)
        except (TypeError, ValueError):
            return None

        return parsed_value if parsed_value >= 0 else None

    @staticmethod
    def _normalize_collection_name(
        collection_name: str,
    ) -> str:
        normalized = str(collection_name).strip().lower()

        if not normalized:
            raise ValueError("Collection name cannot be empty.")

        if len(normalized) < 3:
            raise ValueError(
                "Collection name must contain at least three characters."
            )

        if len(normalized) > 512:
            raise ValueError(
                "Collection name cannot exceed 512 characters."
            )

        return normalized