from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps.auth import get_current_user
from app.database.session import get_db
from app.models.document import DocumentType
from app.models.project import ProjectStatus
from app.models.user import User
from app.rag.indexing import DocumentIndexingError
from app.schemas.document import DocumentResponse
from app.services.document_service import (
    create_document_record,
    get_document_by_hash_for_project,
    get_documents_for_project,
    index_document_record,
    mark_document_failed,
    mark_document_indexed,
    mark_document_processing,
)
from app.services.project_service import get_project_by_id_for_user
from app.utils.file_storage import (
    EmptyFileError,
    FileSizeLimitExceededError,
    FileStorageError,
    store_uploaded_file,
)
from app.utils.file_validation import (
    FileTypeMismatchError,
    FileValidationError,
    MissingFilenameError,
    UnsupportedFileExtensionError,
    UnsupportedMimeTypeError,
    validate_uploaded_file,
)


router = APIRouter(
    prefix="/projects/{project_id}/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    responses={
        status.HTTP_201_CREATED: {
            "description": (
                "Document uploaded and processed. The returned status indicates "
                "whether indexing succeeded or failed."
            ),
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "The uploaded file is empty or invalid.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project workspace not found.",
        },
        status.HTTP_409_CONFLICT: {
            "description": "The same document already exists in the project.",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "The uploaded file exceeds the configured size limit.",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "description": "The uploaded file type is not supported.",
        },
    },
)
async def upload_project_document(
    project_id: UUID,
    document_type: DocumentType = Form(DocumentType.OTHER),
    upload_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """
    Upload and index an engineering document inside a secure project workspace.

    Workflow:

    1. Validate project ownership and status
    2. Validate the uploaded file
    3. Store the file securely
    4. Prevent duplicate documents using SHA-256
    5. Create the PostgreSQL document record
    6. Mark the document as processing
    7. Extract, clean, chunk, embed, and index the document
    8. Mark the document as indexed or failed

    A document whose AI processing fails remains securely stored and is marked
    as FAILED so it can be inspected or reprocessed later.
    """

    project = get_project_by_id_for_user(
        db=db,
        project_id=project_id,
        current_user=current_user,
    )

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project workspace not found.",
        )

    if project.status == ProjectStatus.ARCHIVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Documents cannot be uploaded to an archived project.",
        )

    try:
        validate_uploaded_file(upload_file)

    except (
        UnsupportedFileExtensionError,
        UnsupportedMimeTypeError,
        FileTypeMismatchError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    except (MissingFilenameError, FileValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        stored_file = await store_uploaded_file(
            upload_file=upload_file,
            project_id=project_id,
        )

    except FileSizeLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc

    except EmptyFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored.",
        ) from exc

    existing_document = get_document_by_hash_for_project(
        db=db,
        project_id=project_id,
        sha256_hash=stored_file.sha256_hash,
    )

    if existing_document is not None:
        Path(stored_file.storage_path).unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An identical document already exists "
                "in this project workspace."
            ),
        )

    try:
        document = create_document_record(
            db=db,
            project_id=project_id,
            current_user=current_user,
            original_filename=stored_file.original_filename,
            stored_filename=stored_file.stored_filename,
            file_extension=stored_file.file_extension,
            mime_type=stored_file.mime_type,
            size_bytes=stored_file.size_bytes,
            storage_path=stored_file.storage_path,
            sha256_hash=stored_file.sha256_hash,
            document_type=document_type,
        )

    except SQLAlchemyError as exc:
        db.rollback()
        Path(stored_file.storage_path).unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document metadata could not be saved.",
        ) from exc

    try:
        document = mark_document_processing(
            db=db,
            document=document,
        )

        indexing_result = await run_in_threadpool(
            index_document_record,
            document,
        )

        document = mark_document_indexed(
            db=db,
            document=document,
            indexing_result=indexing_result,
        )

    except DocumentIndexingError as exc:
        document = mark_document_failed(
            db=db,
            document=document,
            error=exc,
        )

    except SQLAlchemyError as exc:
        db.rollback()

        try:
            document = mark_document_failed(
                db=db,
                document=document,
                error="Document processing metadata could not be saved.",
            )
        except SQLAlchemyError:
            db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing state could not be saved.",
        ) from exc

    except Exception as exc:
        document = mark_document_failed(
            db=db,
            document=document,
            error=exc,
        )

    return document


@router.get(
    "",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
    summary="List all documents in a project workspace",
    responses={
        status.HTTP_200_OK: {
            "description": "Project documents returned successfully.",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication credentials are missing or invalid.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Project workspace not found.",
        },
    },
)
def list_project_documents(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DocumentResponse]:
    """
    Return all active documents belonging to a project workspace.

    Only the project owner may retrieve the documents.
    """

    project = get_project_by_id_for_user(
        db=db,
        project_id=project_id,
        current_user=current_user,
    )

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project workspace not found.",
        )

    return get_documents_for_project(
        db=db,
        project_id=project_id,
    )