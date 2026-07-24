from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PrivateRAGQueryRequest(BaseModel):
    """
    Public request contract for project-isolated Private RAG retrieval.

    The project identifier comes from the protected API path and is never
    accepted from the request body.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="Natural-language engineering question.",
        examples=[
            "What safety requirements apply when the steering sensor fails?"
        ],
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description=(
            "Maximum number of curated chunks to include in retrieval."
        ),
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional minimum normalized semantic relevance score."
        ),
    )

    document_ids: list[UUID] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Optional allowlist restricting retrieval to specific "
            "documents inside the project."
        ),
    )

    @field_validator("query")
    @classmethod
    def normalize_query(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(value.split())

        if len(normalized) < 2:
            raise ValueError(
                "Query must contain meaningful text."
            )

        return normalized

    @field_validator("document_ids")
    @classmethod
    def normalize_document_ids(
        cls,
        value: list[UUID] | None,
    ) -> list[UUID] | None:
        if not value:
            return None

        return list(dict.fromkeys(value))


class RAGRetrievedChunkResponse(BaseModel):
    """
    One curated chunk returned by the retrieval engine.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    chunk_id: str

    document_id: UUID

    rank: int = Field(
        ...,
        ge=1,
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    distance: float | None = None

    content: str

    source_filename: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    section_title: str | None = None

    source_reference: str | None = None


class RAGContextSourceResponse(BaseModel):
    """
    Traceability metadata for a chunk included in the LLM context.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_index: int = Field(
        ...,
        ge=1,
    )

    source_label: str

    chunk_id: str

    document_id: UUID

    retrieval_rank: int = Field(
        ...,
        ge=1,
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    source_filename: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    section_title: str | None = None

    source_reference: str | None = None

    included_characters: int = Field(
        ...,
        ge=1,
    )

    was_truncated: bool


class RAGCitationResponse(BaseModel):
    """
    Public citation record corresponding to one source supplied to the LLM.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    citation_id: str

    source_index: int = Field(
        ...,
        ge=1,
    )

    chunk_id: str

    document_id: UUID

    retrieval_rank: int = Field(
        ...,
        ge=1,
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    display_label: str

    source_filename: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    section_title: str | None = None

    source_reference: str | None = None

    was_truncated: bool


class PrivateRAGQueryResponse(BaseModel):
    """
    Complete API response for one project-isolated Private RAG request.

    This response contains evidence preparation only. No LLM-generated answer
    is produced during Phase 8.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    project_id: UUID

    query: str

    collection_name: str

    has_context: bool

    retrieved_chunks: list[RAGRetrievedChunkResponse] = Field(
        default_factory=list,
    )

    retrieved_chunk_count: int = Field(
        ...,
        ge=0,
    )

    candidate_count: int = Field(
        ...,
        ge=0,
    )

    retrieval_time_ms: float = Field(
        ...,
        ge=0.0,
    )

    context_text: str

    context_sources: list[RAGContextSourceResponse] = Field(
        default_factory=list,
    )

    context_character_count: int = Field(
        ...,
        ge=1,
    )

    included_chunk_count: int = Field(
        ...,
        ge=0,
    )

    truncated_chunk_count: int = Field(
        ...,
        ge=0,
    )

    context_was_limited: bool

    citations: list[RAGCitationResponse] = Field(
        default_factory=list,
    )

    citation_count: int = Field(
        ...,
        ge=0,
    )

    prepared_at: datetime