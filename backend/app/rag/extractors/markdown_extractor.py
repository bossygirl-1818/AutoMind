from __future__ import annotations

import re
from pathlib import Path

from app.rag.extractors.base import (
    BaseDocumentExtractor,
    DocumentExtractionError,
    ExtractedSection,
    ExtractionResult,
)


class MarkdownDocumentExtractor(BaseDocumentExtractor):
    """
    Extract Markdown documents while preserving heading-based structure.

    Each heading and its following content become a traceable section.
    Content before the first heading is preserved as an introductory section.
    """

    supported_extensions = frozenset({".md", ".markdown"})

    _heading_pattern = re.compile(
        r"^(?P<level>#{1,6})[ \t]+(?P<title>.+?)\s*$",
        flags=re.MULTILINE,
    )

    def extract(self, file_path: Path) -> ExtractionResult:
        path = self.validate_file(file_path)

        try:
            try:
                text = path.read_text(encoding="utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                encoding = "utf-8-replacement"

        except OSError as exc:
            raise DocumentExtractionError(
                f"Failed to read Markdown document '{path.name}'."
            ) from exc

        sections = self._extract_sections(text)

        return ExtractionResult(
            source_path=path,
            file_type=path.suffix,
            extraction_method="MarkdownHeadingExtractor",
            sections=tuple(sections),
            metadata={
                "encoding": encoding,
                "heading_count": sum(
                    1 for section in sections if section.heading is not None
                ),
            },
        )

    def _extract_sections(self, text: str) -> list[ExtractedSection]:
        """
        Split Markdown content using ATX headings such as #, ##, and ###.

        The original heading line is included in the extracted section text
        so technical context is not lost during downstream chunking.
        """

        matches = list(self._heading_pattern.finditer(text))
        sections: list[ExtractedSection] = []

        if not matches:
            sections.append(
                ExtractedSection(
                    text=text,
                    metadata={
                        "section_type": "document",
                        "heading_level": None,
                    },
                )
            )
            return sections

        introductory_text = text[: matches[0].start()].strip()

        if introductory_text:
            sections.append(
                ExtractedSection(
                    text=introductory_text,
                    metadata={
                        "section_type": "introduction",
                        "heading_level": None,
                    },
                )
            )

        for index, match in enumerate(matches):
            section_start = match.start()
            section_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            section_text = text[section_start:section_end].strip()
            heading = match.group("title").strip()
            heading_level = len(match.group("level"))

            sections.append(
                ExtractedSection(
                    text=section_text,
                    heading=heading,
                    metadata={
                        "section_type": "heading",
                        "heading_level": heading_level,
                    },
                )
            )

        return sections