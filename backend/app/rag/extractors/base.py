from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DocumentExtractionError(Exception):
    """Base exception for document text-extraction failures."""


class UnsupportedDocumentTypeError(DocumentExtractionError):
    """Raised when an extractor does not support the supplied file type."""


class EmptyDocumentError(DocumentExtractionError):
    """Raised when no usable text can be extracted from a document."""


@dataclass(frozen=True, slots=True)
class ExtractedSection:
    """
    Represents one traceable unit extracted from a source document.

    A section may correspond to a PDF page, DOCX paragraph group,
    Markdown section, or a complete plain-text document.
    """

    text: str
    page_number: int | None = None
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned_text = self.text.strip()

        if not cleaned_text:
            raise ValueError("Extracted section text cannot be empty.")

        if self.page_number is not None and self.page_number < 1:
            raise ValueError("Page number must be greater than or equal to 1.")

        object.__setattr__(self, "text", cleaned_text)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """
    Standard output returned by every AutoMind document extractor.

    This common structure allows cleaning, chunking, citation generation,
    and indexing services to remain independent of the original file type.
    """

    source_path: Path
    file_type: str
    extraction_method: str
    sections: tuple[ExtractedSection, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_file_type = self.file_type.lower().lstrip(".").strip()

        if not normalized_file_type:
            raise ValueError("File type cannot be empty.")

        if not self.extraction_method.strip():
            raise ValueError("Extraction method cannot be empty.")

        if not self.sections:
            raise EmptyDocumentError(
                f"No usable text was extracted from '{self.source_path.name}'."
            )

        object.__setattr__(self, "file_type", normalized_file_type)
        object.__setattr__(
            self,
            "extraction_method",
            self.extraction_method.strip(),
        )
        object.__setattr__(self, "source_path", self.source_path.resolve())

    @property
    def full_text(self) -> str:
        """Return all extracted sections in their original order."""

        return "\n\n".join(section.text for section in self.sections)

    @property
    def total_characters(self) -> int:
        """Return the total number of extracted text characters."""

        return sum(len(section.text) for section in self.sections)

    @property
    def section_count(self) -> int:
        """Return the total number of extracted sections."""

        return len(self.sections)


class BaseDocumentExtractor(ABC):
    """
    Abstract contract implemented by every AutoMind document extractor.

    Concrete extractors are responsible only for reading their supported
    document format and returning a normalized ExtractionResult.
    """

    supported_extensions: frozenset[str] = frozenset()

    def validate_file(self, file_path: Path) -> Path:
        """
        Validate the source file before extraction.

        Returns the normalized absolute path when validation succeeds.
        """

        normalized_path = file_path.expanduser().resolve()

        if not normalized_path.exists():
            raise DocumentExtractionError(
                f"Document file does not exist: '{normalized_path.name}'."
            )

        if not normalized_path.is_file():
            raise DocumentExtractionError(
                f"Document path is not a file: '{normalized_path.name}'."
            )

        extension = normalized_path.suffix.lower()

        if extension not in self.supported_extensions:
            supported = ", ".join(sorted(self.supported_extensions))

            raise UnsupportedDocumentTypeError(
                f"Unsupported file type '{extension or 'unknown'}'. "
                f"Supported types: {supported}."
            )

        return normalized_path

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractionResult:
        """
        Extract structured text and metadata from a document.

        Implementations must raise DocumentExtractionError or one of its
        subclasses when extraction cannot be completed safely.
        """

        raise NotImplementedError