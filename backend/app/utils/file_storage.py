import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.core.config import settings


CHUNK_SIZE_BYTES = 1024 * 1024


class FileStorageError(Exception):
    """
    Base exception for document-storage failures.
    """


class FileSizeLimitExceededError(FileStorageError):
    """
    Raised when an uploaded file exceeds the configured size limit.
    """


class EmptyFileError(FileStorageError):
    """
    Raised when an uploaded file contains no data.
    """


@dataclass(frozen=True)
class StoredFileResult:
    """
    Metadata generated after a file is stored successfully.
    """

    original_filename: str
    stored_filename: str
    file_extension: str
    mime_type: str
    size_bytes: int
    storage_path: Path
    sha256_hash: str


def sanitize_original_filename(filename: str | None) -> str:
    """
    Remove directory components from a client-provided filename.

    This prevents path-traversal values such as:
    ../../sensitive-file.txt
    """

    if not filename:
        return "unnamed-document"

    safe_filename = Path(filename).name.strip()

    return safe_filename or "unnamed-document"


def build_project_upload_directory(project_id: UUID) -> Path:
    """
    Create and return the storage directory for a project workspace.
    """

    upload_root = Path(settings.UPLOAD_DIR).resolve()
    project_directory = upload_root / str(project_id)

    project_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return project_directory


async def store_uploaded_file(
    *,
    upload_file: UploadFile,
    project_id: UUID,
) -> StoredFileResult:
    """
    Store an uploaded document securely using chunked file processing.

    The function:

    1. Sanitizes the original filename.
    2. Creates a project-specific directory.
    3. Generates a collision-resistant stored filename.
    4. Streams the file to disk in chunks.
    5. Enforces the configured upload-size limit.
    6. Calculates the SHA-256 hash during writing.
    7. Removes incomplete files when storage fails.
    """

    original_filename = sanitize_original_filename(
        upload_file.filename,
    )

    file_extension = Path(original_filename).suffix.lower()

    stored_filename = f"{uuid.uuid4().hex}{file_extension}"

    project_directory = build_project_upload_directory(
        project_id,
    )

    storage_path = project_directory / stored_filename

    sha256_hasher = hashlib.sha256()
    total_size = 0

    try:
        with storage_path.open("wb") as destination:
            while chunk := await upload_file.read(CHUNK_SIZE_BYTES):
                total_size += len(chunk)

                if total_size > settings.max_upload_size_bytes:
                    raise FileSizeLimitExceededError(
                        "Uploaded file exceeds the configured "
                        f"{settings.MAX_UPLOAD_SIZE_MB} MB limit."
                    )

                sha256_hasher.update(chunk)
                destination.write(chunk)

        if total_size == 0:
            raise EmptyFileError(
                "Uploaded file is empty."
            )

        return StoredFileResult(
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_extension=file_extension,
            mime_type=(
                upload_file.content_type
                or "application/octet-stream"
            ),
            size_bytes=total_size,
            storage_path=storage_path,
            sha256_hash=sha256_hasher.hexdigest(),
        )

    except Exception:
        if storage_path.exists():
            storage_path.unlink()

        raise

    finally:
        await upload_file.close()