from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from app.rag.extractors.base import (
    BaseDocumentExtractor,
    UnsupportedDocumentTypeError,
)
from app.rag.extractors.docx_extractor import DocxDocumentExtractor
from app.rag.extractors.markdown_extractor import MarkdownDocumentExtractor
from app.rag.extractors.pdf_extractor import PdfDocumentExtractor
from app.rag.extractors.txt_extractor import TxtDocumentExtractor


ExtractorType: TypeAlias = type[BaseDocumentExtractor]


class DocumentExtractorRegistry:
    """
    Central registry for AutoMind document extractors.

    The registry maps supported file extensions to extractor classes and
    creates the correct extractor without exposing format-specific logic
    to the document-processing pipeline.
    """

    def __init__(self) -> None:
        self._extractors: dict[str, ExtractorType] = {}

        self.register(TxtDocumentExtractor)
        self.register(MarkdownDocumentExtractor)
        self.register(PdfDocumentExtractor)
        self.register(DocxDocumentExtractor)

    def register(
        self,
        extractor_class: ExtractorType,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register an extractor class for all its supported extensions.

        Args:
            extractor_class:
                Concrete extractor class to register.

            overwrite:
                Whether an existing extension mapping may be replaced.

        Raises:
            ValueError:
                If the extractor declares no supported extensions or an
                extension has already been registered.
        """

        extensions = extractor_class.supported_extensions

        if not extensions:
            raise ValueError(
                f"Extractor '{extractor_class.__name__}' does not declare "
                "any supported file extensions."
            )

        for extension in extensions:
            normalized_extension = self._normalize_extension(extension)

            if (
                normalized_extension in self._extractors
                and not overwrite
            ):
                existing_extractor = self._extractors[normalized_extension]

                raise ValueError(
                    f"Extension '{normalized_extension}' is already registered "
                    f"to '{existing_extractor.__name__}'."
                )

            self._extractors[normalized_extension] = extractor_class

    def get_extractor(
        self,
        file_path: str | Path,
    ) -> BaseDocumentExtractor:
        """
        Return an extractor instance for the supplied document path.

        Raises:
            UnsupportedDocumentTypeError:
                If no extractor is registered for the file extension.
        """

        path = Path(file_path)
        extension = self._normalize_extension(path.suffix)

        extractor_class = self._extractors.get(extension)

        if extractor_class is None:
            supported_extensions = ", ".join(
                sorted(self._extractors.keys())
            )

            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{extension or 'unknown'}'. "
                f"Supported types: {supported_extensions}."
            )

        return extractor_class()

    def supports(
        self,
        file_path_or_extension: str | Path,
    ) -> bool:
        """
        Return True when the file extension has a registered extractor.
        """

        value = str(file_path_or_extension)
        extension = Path(value).suffix or value
        normalized_extension = self._normalize_extension(extension)

        return normalized_extension in self._extractors

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """
        Return all registered extensions in deterministic order.
        """

        return tuple(sorted(self._extractors.keys()))

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        """
        Normalize extensions into lowercase dot-prefixed format.

        Examples:
            PDF  -> .pdf
            .DOCX -> .docx
        """

        normalized = extension.strip().lower()

        if not normalized:
            return ""

        if not normalized.startswith("."):
            normalized = f".{normalized}"

        return normalized


document_extractor_registry = DocumentExtractorRegistry()