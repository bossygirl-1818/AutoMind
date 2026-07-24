from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SemanticSearchRequest(BaseModel):
    """
    Request payload for project-isolated semantic search.

    The project ID is intentionally not included here. It must be obtained
    from the API path and validated against the authenticated user's access
    permissions before the retrieval service is called.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="Natural-language engineering query.",
        examples=[
            "What safety requirements apply when the steering sensor fails?"
        ],
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve.",
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional normalized relevance threshold. Results below this "
            "threshold are excluded."
        ),
    )

    document_ids: list[UUID] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Optional document allowlist. Retrieval will be restricted to "
            "these documents after project ownership validation."
        ),
    )

    include_metadata: bool = Field(
        default=True,
        description="Whether chunk metadata should be included in the response.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized_query = " ".join(value.split())

        if len(normalized_query) < 2:
            raise ValueError("Query must contain meaningful text.")

        return normalized_query

    @field_validator("document_ids")
    @classmethod
    def remove_duplicate_document_ids(
        cls,
        value: list[UUID] | None,
    ) -> list[UUID] | None:
        if value is None:
            return None

        # dict preserves insertion order while removing duplicates.
        unique_document_ids = list(dict.fromkeys(value))

        if not unique_document_ids:
            return None

        return unique_document_ids


class RetrievedChunkMetadata(BaseModel):
    """
    Structured metadata attached to an indexed document chunk.

    Known metadata is represented explicitly while provider-specific or
    extractor-specific metadata remains available through `additional`.
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    original_filename: str | None = None

    document_type: str | None = None

    file_extension: str | None = None

    chunk_index: int | None = Field(
        default=None,
        ge=0,
    )

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    section_title: str | None = None

    source_reference: str | None = Field(
        default=None,
        description=(
            "Human-readable source location such as a page, section, "
            "heading, or paragraph range."
        ),
    )

    additional: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional sanitized metadata stored with the chunk.",
    )


class RetrievedChunk(BaseModel):
    """
    A single semantic retrieval result.

    `distance` preserves the raw vector-store result, while
    `relevance_score` provides a normalized value suitable for APIs,
    thresholding, ranking, and later context-building stages.
    """

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    chunk_id: str = Field(
        ...,
        min_length=1,
    )

    document_id: UUID

    content: str = Field(
        ...,
        min_length=1,
    )

    rank: int = Field(
        ...,
        ge=1,
    )

    distance: float | None = Field(
        default=None,
        description="Raw distance returned by the vector database.",
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized semantic relevance score.",
    )

    metadata: RetrievedChunkMetadata | None = None


class SemanticSearchResponse(BaseModel):
    """
    Response returned by the semantic search API.

    Timing and collection information are included for observability and
    debugging without exposing private embedding vectors.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    project_id: UUID

    query: str

    results: list[RetrievedChunk] = Field(
        default_factory=list,
    )

    result_count: int = Field(
        ...,
        ge=0,
    )

    collection_name: str = Field(
        ...,
        min_length=1,
    )

    retrieval_time_ms: float = Field(
        ...,
        ge=0.0,
    )

    retrieved_at: datetime