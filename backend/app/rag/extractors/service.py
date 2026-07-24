from __future__ import annotations

from pathlib import Path

from app.rag.extractors.base import ExtractionResult
from app.rag.extractors.registry import document_extractor_registry


class DocumentExtractionService:
    """
    High-level service responsible for document extraction.

    The rest of AutoMind should never directly instantiate
    PDF/DOCX/TXT extractors. This service provides a single
    entry point for document extraction.
    """

    def extract(
        self,
        file_path: str | Path,
    ) -> ExtractionResult:
        """
        Extract text from any supported document.

        Parameters
        ----------
        file_path:
            Path to the uploaded document.

        Returns
        -------
        ExtractionResult
            Standardized extraction result.
        """

        path = Path(file_path)

        extractor = document_extractor_registry.get_extractor(path)

        return extractor.extract(path)


document_extraction_service = DocumentExtractionService()