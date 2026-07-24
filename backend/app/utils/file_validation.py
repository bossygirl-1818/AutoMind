from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


class FileValidationError(Exception):
    """
    Base exception for uploaded-file validation failures.
    """


class MissingFilenameError(FileValidationError):
    """
    Raised when an uploaded file does not contain a valid filename.
    """


class UnsupportedFileExtensionError(FileValidationError):
    """
    Raised when the uploaded file extension is not permitted.
    """


class UnsupportedMimeTypeError(FileValidationError):
    """
    Raised when the uploaded file MIME type is not permitted.
    """


class FileTypeMismatchError(FileValidationError):
    """
    Raised when the file extension and MIME type do not match.
    """


EXTENSION_MIME_TYPE_MAP: dict[str, set[str]] = {
    ".pdf": {
        "application/pdf",
    },
    ".docx": {
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document",
    },
    ".txt": {
        "text/plain",
    },
    ".md": {
        "text/markdown",
        "text/plain",
    },
}


def normalize_mime_type(mime_type: str | None) -> str:
    """
    Normalize a MIME type provided by the upload client.

    MIME types can occasionally include parameters such as:

    text/plain; charset=utf-8

    Only the primary MIME type is used for validation.
    """

    if not mime_type:
        return "application/octet-stream"

    return mime_type.split(";", maxsplit=1)[0].strip().lower()


def get_file_extension(filename: str) -> str:
    """
    Extract and normalize the file extension from a filename.
    """

    return Path(filename).suffix.lower()


def validate_uploaded_file(upload_file: UploadFile) -> None:
    """
    Validate an uploaded document before it is written to storage.

    Validation includes:

    1. Confirming that the filename exists.
    2. Confirming that the extension is allowed.
    3. Confirming that the MIME type is allowed.
    4. Confirming that the extension and MIME type agree.
    """

    filename = (
        Path(upload_file.filename).name.strip()
        if upload_file.filename
        else ""
    )

    if not filename:
        raise MissingFilenameError(
            "The uploaded file must have a valid filename."
        )

    file_extension = get_file_extension(filename)

    if not file_extension:
        raise UnsupportedFileExtensionError(
            "The uploaded file must include a supported file extension."
        )

    if file_extension not in settings.allowed_document_extensions:
        allowed_extensions = ", ".join(
            sorted(settings.allowed_document_extensions)
        )

        raise UnsupportedFileExtensionError(
            f"Unsupported file extension '{file_extension}'. "
            f"Allowed extensions: {allowed_extensions}."
        )

    normalized_mime_type = normalize_mime_type(
        upload_file.content_type,
    )

    if normalized_mime_type not in settings.allowed_document_mime_types:
        allowed_mime_types = ", ".join(
            sorted(settings.allowed_document_mime_types)
        )

        raise UnsupportedMimeTypeError(
            f"Unsupported MIME type '{normalized_mime_type}'. "
            f"Allowed MIME types: {allowed_mime_types}."
        )

    expected_mime_types = EXTENSION_MIME_TYPE_MAP.get(
        file_extension,
        set(),
    )

    if (
        expected_mime_types
        and normalized_mime_type not in expected_mime_types
    ):
        expected_values = ", ".join(
            sorted(expected_mime_types)
        )

        raise FileTypeMismatchError(
            f"The file extension '{file_extension}' does not match "
            f"the supplied MIME type '{normalized_mime_type}'. "
            f"Expected MIME type: {expected_values}."
        )