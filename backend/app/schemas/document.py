from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import (
    DocumentStatus,
    DocumentType,
)


class DocumentCreate(BaseModel):
    project_id: UUID
    document_type: DocumentType = DocumentType.OTHER


class DocumentResponse(BaseModel):
    id: UUID
    project_id: UUID
    uploaded_by_id: UUID

    original_filename: str
    stored_filename: str
    file_extension: str
    mime_type: str
    size_bytes: int
    storage_path: str
    sha256_hash: str

    document_type: DocumentType
    status: DocumentStatus

    version_number: int
    is_latest: bool

    processing_error: str | None

    processing_started_at: datetime | None
    processing_completed_at: datetime | None

    extracted_character_count: int | None
    extracted_section_count: int | None
    generated_chunk_count: int | None

    embedding_model: str | None
    embedding_provider: str | None
    embedding_dimensions: int | None

    chunking_strategy: str | None
    vector_collection_name: str | None

    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None

    model_config = ConfigDict(
        from_attributes=True,
    )


class DocumentUpdate(BaseModel):
    document_type: DocumentType | None = None
    status: DocumentStatus | None = None

    processing_error: str | None = Field(
        default=None,
        max_length=5000,
    )