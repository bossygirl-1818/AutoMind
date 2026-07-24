from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from app.rag.extractors.base import (
    BaseDocumentExtractor,
    DocumentExtractionError,
    EmptyDocumentError,
    ExtractedSection,
    ExtractionResult,
)


class PdfDocumentExtractor(BaseDocumentExtractor):
    """
    Extract text from PDF documents using PyMuPDF.

    Each PDF page is returned as a separate ExtractedSection so that
    page-level citations and engineering traceability can be preserved.
    """

    supported_extensions = frozenset({".pdf"})

    def extract(self, file_path: Path) -> ExtractionResult:
        path = self.validate_file(file_path)

        try:
            document = fitz.open(path)
        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to open PDF document '{path.name}'."
            ) from exc

        try:
            if document.needs_pass:
                raise DocumentExtractionError(
                    f"PDF document '{path.name}' is password protected."
                )

            sections: list[ExtractedSection] = []
            total_pages = document.page_count

            for page_index in range(total_pages):
                page = document.load_page(page_index)

                try:
                    page_text = page.get_text("text", sort=True)
                except Exception as exc:
                    raise DocumentExtractionError(
                        f"Failed to extract page {page_index + 1} "
                        f"from PDF document '{path.name}'."
                    ) from exc

                cleaned_page_text = page_text.strip()

                if not cleaned_page_text:
                    continue

                page_metadata = self._build_page_metadata(
                    page=page,
                    page_index=page_index,
                )

                sections.append(
                    ExtractedSection(
                        text=cleaned_page_text,
                        page_number=page_index + 1,
                        metadata=page_metadata,
                    )
                )

            if not sections:
                raise EmptyDocumentError(
                    f"No readable text was found in PDF document '{path.name}'. "
                    "The document may be scanned or image-based."
                )

            document_metadata = self._build_document_metadata(
                document=document,
                total_pages=total_pages,
            )

            return ExtractionResult(
                source_path=path,
                file_type="pdf",
                extraction_method="PyMuPDFTextExtractor",
                sections=tuple(sections),
                metadata=document_metadata,
            )

        except DocumentExtractionError:
            raise

        except Exception as exc:
            raise DocumentExtractionError(
                f"Unexpected PDF extraction failure for '{path.name}'."
            ) from exc

        finally:
            document.close()

    @staticmethod
    def _build_page_metadata(
        page: fitz.Page,
        page_index: int,
    ) -> dict[str, Any]:
        """
        Build metadata for an individual PDF page.

        Width and height are useful later for layout-aware processing,
        table extraction, and scanned-document diagnostics.
        """

        page_rect = page.rect

        return {
            "page_index": page_index,
            "page_width": float(page_rect.width),
            "page_height": float(page_rect.height),
            "rotation": int(page.rotation),
            "section_type": "pdf_page",
        }

    @staticmethod
    def _build_document_metadata(
        document: fitz.Document,
        total_pages: int,
    ) -> dict[str, Any]:
        """
        Extract safe PDF-level metadata.

        Empty metadata values are excluded to avoid storing noisy records.
        """

        raw_metadata = document.metadata or {}

        safe_metadata = {
            key: value.strip()
            for key, value in raw_metadata.items()
            if isinstance(value, str) and value.strip()
        }

        return {
            "page_count": total_pages,
            "pdf_metadata": safe_metadata,
            "is_encrypted": bool(document.is_encrypted),
            "format": "PDF",
        }