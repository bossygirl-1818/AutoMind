from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import (
    Document,
    DocumentStatus,
    DocumentType,
)
from app.models.user import User
from app.rag.embeddings import local_embedding_provider
from app.rag.indexing import (
    DocumentIndexingError,
    DocumentIndexingResult,
    document_indexing_pipeline,
)


def get_document_by_hash_for_project(
    db: Session,
    project_id: UUID,
    sha256_hash: str,
) -> Document | None:
    """
    Retrieve an existing latest document in a project using its SHA-256 hash.

    This supports duplicate-file detection within the same project workspace.
    """

    return (
        db.query(Document)
        .filter(
            Document.project_id == project_id,
            Document.sha256_hash == sha256_hash,
            Document.is_latest.is_(True),
        )
        .first()
    )


def create_document_record(
    db: Session,
    *,
    project_id: UUID,
    current_user: User,
    original_filename: str,
    stored_filename: str,
    file_extension: str,
    mime_type: str,
    size_bytes: int,
    storage_path: Path,
    sha256_hash: str,
    document_type: DocumentType,
) -> Document:
    """
    Create the PostgreSQL metadata record for an uploaded document.

    The physical file must already be stored successfully before this
    function is called.

    AI indexing is handled separately so database persistence and document
    intelligence remain independently testable.
    """

    document = Document(
        project_id=project_id,
        uploaded_by_id=current_user.id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_extension=file_extension,
        mime_type=mime_type,
        size_bytes=size_bytes,
        storage_path=str(storage_path),
        sha256_hash=sha256_hash,
        document_type=document_type,
        status=DocumentStatus.UPLOADED,
    )

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()
        raise

    return document


def mark_document_processing(
    db: Session,
    document: Document,
) -> Document:
    """
    Mark a document as actively being processed by the Phase 7 pipeline.

    Previous processing results and errors are cleared so controlled
    reprocessing does not expose stale metadata.
    """

    document.status = DocumentStatus.PROCESSING
    document.processing_started_at = datetime.utcnow()
    document.processing_completed_at = None
    document.processing_error = None

    document.extracted_character_count = None
    document.extracted_section_count = None
    document.generated_chunk_count = None

    document.embedding_model = None
    document.embedding_provider = None
    document.embedding_dimensions = None
    document.chunking_strategy = None
    document.vector_collection_name = None

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()
        raise

    return document


def mark_document_indexed(
    db: Session,
    document: Document,
    indexing_result: DocumentIndexingResult,
) -> Document:
    """
    Persist successful document-intelligence processing metadata.
    """

    document.status = DocumentStatus.INDEXED
    document.processing_completed_at = datetime.utcnow()
    document.processing_error = None

    document.extracted_character_count = (
        indexing_result.extracted_characters
    )
    document.extracted_section_count = (
        indexing_result.extracted_sections
    )
    document.generated_chunk_count = indexing_result.generated_chunks

    document.embedding_model = local_embedding_provider.model_name
    document.embedding_provider = local_embedding_provider.provider_name
    document.embedding_dimensions = local_embedding_provider.dimensions

    document.chunking_strategy = indexing_result.chunking_strategy
    document.vector_collection_name = indexing_result.collection_name

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()
        raise

    return document


def mark_document_failed(
    db: Session,
    document: Document,
    error: Exception | str,
) -> Document:
    """
    Mark a document as failed while preserving its uploaded source file.

    Internal exception details are stored in a bounded form for diagnostics.
    They should not be returned directly to API clients.
    """

    error_message = str(error).strip() or "Unknown document processing error."

    document.status = DocumentStatus.FAILED
    document.processing_completed_at = datetime.utcnow()
    document.processing_error = error_message[:4000]

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()
        raise

    return document


def index_document_record(
    document: Document,
) -> DocumentIndexingResult:
    """
    Process and index an existing AutoMind document record.

    Workflow:

    1. Text extraction
    2. Technical-text cleaning
    3. Engineering-aware chunking
    4. Local embedding generation
    5. Persistent ChromaDB indexing
    """

    storage_path = Path(document.storage_path).expanduser().resolve()

    if not storage_path.exists():
        raise DocumentIndexingError(
            f"Stored file for document '{document.id}' does not exist."
        )

    if not storage_path.is_file():
        raise DocumentIndexingError(
            f"Storage path for document '{document.id}' is not a file."
        )

    collection_name = build_project_collection_name(document.project_id)

    return document_indexing_pipeline.index_document(
        file_path=storage_path,
        document_id=str(document.id),
        project_id=str(document.project_id),
        collection_name=collection_name,
        source_filename=document.original_filename,
        document_hash=document.sha256_hash,
    )


def build_project_collection_name(
    project_id: UUID | str,
) -> str:
    """
    Build a deterministic ChromaDB collection name for a project.

    Each project receives an isolated vector collection, preventing semantic
    retrieval across unrelated engineering workspaces.
    """

    normalized_project_id = str(project_id).strip().lower()

    if not normalized_project_id:
        raise ValueError("Project ID cannot be empty.")

    return f"automind-project-{normalized_project_id}"


def get_documents_for_project(
    db: Session,
    project_id: UUID,
) -> list[Document]:
    """
    Retrieve all non-archived documents belonging to a project workspace.

    Documents are returned newest first.
    """

    return (
        db.query(Document)
        .filter(
            Document.project_id == project_id,
            Document.status != DocumentStatus.ARCHIVED,
        )
        .order_by(Document.created_at.desc())
        .all()
    )