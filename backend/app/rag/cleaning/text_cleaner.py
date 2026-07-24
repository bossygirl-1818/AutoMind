from __future__ import annotations

import re

from app.rag.extractors.base import (
    ExtractedSection,
    ExtractionResult,
)


class TextCleaner:
    """
    Cleans extracted engineering documents while preserving
    technical meaning.

    The cleaner intentionally avoids aggressive normalization.
    Requirement IDs, CAN identifiers, AUTOSAR names,
    ISO references, code snippets, and engineering symbols
    should remain unchanged whenever possible.
    """

    _multiple_spaces = re.compile(r"[ \t]{2,}")
    _multiple_blank_lines = re.compile(r"\n{3,}")

    def clean(
        self,
        extraction: ExtractionResult,
    ) -> ExtractionResult:
        """
        Clean every extracted section.

        Returns a new immutable ExtractionResult.
        """

        cleaned_sections = tuple(
            self._clean_section(section)
            for section in extraction.sections
        )

        return ExtractionResult(
            source_path=extraction.source_path,
            file_type=extraction.file_type,
            extraction_method=extraction.extraction_method,
            sections=cleaned_sections,
            metadata=extraction.metadata,
        )

    def _clean_section(
        self,
        section: ExtractedSection,
    ) -> ExtractedSection:

        text = section.text

        text = self._normalize_newlines(text)
        text = self._remove_null_bytes(text)
        text = self._collapse_spaces(text)
        text = self._collapse_blank_lines(text)

        return ExtractedSection(
            text=text.strip(),
            page_number=section.page_number,
            heading=section.heading,
            metadata=section.metadata,
        )

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """
        Convert CRLF / CR into LF.
        """

        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_null_bytes(text: str) -> str:
        """
        Remove NULL characters occasionally found
        in extracted PDF text.
        """

        return text.replace("\x00", "")

    def _collapse_spaces(self, text: str) -> str:
        """
        Collapse repeated spaces while preserving indentation.
        """

        return self._multiple_spaces.sub(" ", text)

    def _collapse_blank_lines(self, text: str) -> str:
        """
        Prevent huge vertical gaps without
        destroying paragraph boundaries.
        """

        return self._multiple_blank_lines.sub("\n\n", text)


text_cleaner = TextCleaner()