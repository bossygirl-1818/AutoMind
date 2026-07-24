from __future__ import annotations

from pathlib import Path

from app.rag.extractors.base import (
    BaseDocumentExtractor,
    DocumentExtractionError,
    ExtractedSection,
    ExtractionResult,
)


class TxtDocumentExtractor(BaseDocumentExtractor):
    """
    Extract plain text documents.

    UTF-8 is attempted first.
    If decoding fails, UTF-8 with replacement is used so that
    corrupted bytes do not stop indexing.
    """

    supported_extensions = frozenset({".txt"})

    def extract(self, file_path: Path) -> ExtractionResult:
        path = self.validate_file(file_path)

        try:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

        except Exception as exc:
            raise DocumentExtractionError(
                f"Failed to read text document '{path.name}'."
            ) from exc

        return ExtractionResult(
            source_path=path,
            file_type="txt",
            extraction_method="PlainTextExtractor",
            sections=(
                ExtractedSection(
                    text=text,
                ),
            ),
            metadata={
                "encoding": "utf-8",
            },
        )