from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GroundedAnswerQueryRequest(BaseModel):
    """
    Public request contract for one project-grounded engineering answer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="Engineering question to answer using project evidence.",
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    document_ids: list[UUID] | None = Field(
        default=None,
        max_length=100,
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


class UsedCitationResponse(BaseModel):
    """
    Citation actually referenced by the generated answer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    citation_id: str

    source_index: int = Field(
        ...,
        ge=1,
    )

    document_id: UUID

    chunk_id: str

    display_label: str

    source_filename: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    section_title: str | None = None

    source_reference: str | None = None

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class GroundedAnswerQueryResponse(BaseModel):
    """
    Public grounded-answer response returned by AutoMind.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    project_id: UUID

    query: str

    answer: str

    has_context: bool

    provider_name: str

    model_name: str

    finish_reason: str | None = None

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    llm_latency_ms: float = Field(
        ...,
        ge=0.0,
    )

    retrieval_time_ms: float = Field(
        ...,
        ge=0.0,
    )

    retrieved_chunk_count: int = Field(
        ...,
        ge=0,
    )

    included_chunk_count: int = Field(
        ...,
        ge=0,
    )

    used_citation_ids: list[str] = Field(
        default_factory=list,
    )

    used_citations: list[UsedCitationResponse] = Field(
        default_factory=list,
    )

    citation_count: int = Field(
        ...,
        ge=0,
    )

    generated_at: datetime