from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    REQUIREMENT = "requirement"
    SAFETY = "safety"
    TEST = "test"
    ARCHITECTURE = "architecture"
    SPECIFICATION = "specification"
    SOURCE_CODE = "source_code"
    VEHICLE_LOG = "vehicle_log"
    REPORT = "report"
    OTHER = "other"


class Document(Base):
    __tablename__ = "documents"

    __table_args__ = (
        Index(
            "ix_documents_project_status",
            "project_id",
            "status",
        ),
        Index(
            "ix_documents_project_created_at",
            "project_id",
            "created_at",
        ),
        Index(
            "ix_documents_uploaded_by_id",
            "uploaded_by_id",
        ),
        Index(
            "ix_documents_sha256_hash",
            "sha256_hash",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "projects.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    stored_filename: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    file_extension: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    mime_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    sha256_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    document_type: Mapped[DocumentType] = mapped_column(
        SqlEnum(
            DocumentType,
            name="document_type",
        ),
        default=DocumentType.OTHER,
        nullable=False,
    )

    status: Mapped[DocumentStatus] = mapped_column(
        SqlEnum(
            DocumentStatus,
            name="document_status",
        ),
        default=DocumentStatus.UPLOADED,
        nullable=False,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    is_latest: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    extracted_character_count: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    extracted_section_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    generated_chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    embedding_dimensions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    chunking_strategy: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    vector_collection_name: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    project = relationship(
        "Project",
        foreign_keys=[project_id],
        lazy="joined",
    )

    uploaded_by = relationship(
        "User",
        foreign_keys=[uploaded_by_id],
        lazy="joined",
    )