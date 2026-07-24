from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.rag.retrieval.context_builder import BuiltContext, ContextSource


class CitationEngineError(Exception):
    """
    Raised when citation records cannot be generated safely.
    """


@dataclass(frozen=True, slots=True)
class Citation:
    """
    Represents one source citation available to the LLM and API layers.

    Citation labels are deterministic and correspond directly to the
    source labels inserted by the context builder:

        [SOURCE 1]
        [SOURCE 2]
        [SOURCE 3]
    """

    citation_id: str

    source_index: int

    chunk_id: str

    document_id: UUID

    retrieval_rank: int

    relevance_score: float

    source_filename: str | None

    page_number: int | None

    section_title: str | None

    source_reference: str | None

    display_label: str

    was_truncated: bool

    def __post_init__(self) -> None:
        normalized_citation_id = self.citation_id.strip()
        normalized_chunk_id = self.chunk_id.strip()
        normalized_display_label = self.display_label.strip()

        if not normalized_citation_id:
            raise ValueError("Citation ID cannot be empty.")

        if self.source_index < 1:
            raise ValueError(
                "Citation source_index must be greater than zero."
            )

        if not normalized_chunk_id:
            raise ValueError("Citation chunk ID cannot be empty.")

        if self.retrieval_rank < 1:
            raise ValueError(
                "Citation retrieval rank must be greater than zero."
            )

        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(
                "Citation relevance score must be between 0 and 1."
            )

        if not normalized_display_label:
            raise ValueError("Citation display label cannot be empty.")

        object.__setattr__(
            self,
            "citation_id",
            normalized_citation_id,
        )
        object.__setattr__(
            self,
            "chunk_id",
            normalized_chunk_id,
        )
        object.__setattr__(
            self,
            "display_label",
            normalized_display_label,
        )


@dataclass(frozen=True, slots=True)
class CitationBundle:
    """
    Complete citation registry for one built RAG context.

    The ordered citation tuple preserves the same source order used inside
    the LLM context. The index allows fast lookup when generated answers
    reference a label such as [SOURCE 2].
    """

    project_id: UUID

    query: str

    citations: tuple[Citation, ...]

    citation_index: dict[str, Citation]

    def __post_init__(self) -> None:
        normalized_query = self.query.strip()

        if not normalized_query:
            raise ValueError(
                "Citation bundle query cannot be empty."
            )

        if len(self.citations) != len(self.citation_index):
            raise ValueError(
                "Citation index must contain exactly one entry "
                "for every citation."
            )

        for citation in self.citations:
            indexed_citation = self.citation_index.get(
                citation.citation_id
            )

            if indexed_citation != citation:
                raise ValueError(
                    f"Citation index entry '{citation.citation_id}' "
                    "does not match the citation registry."
                )

        object.__setattr__(
            self,
            "query",
            normalized_query,
        )

    @property
    def count(self) -> int:
        return len(self.citations)

    @property
    def is_empty(self) -> bool:
        return not self.citations

    def get(
        self,
        citation_id: str,
    ) -> Citation | None:
        """
        Return one citation using a normalized citation identifier.
        """

        normalized_citation_id = str(citation_id).strip().upper()

        return self.citation_index.get(
            normalized_citation_id
        )


class CitationEngine:
    """
    Generates deterministic citations from a source-aware BuiltContext.

    Responsibilities:

        1. Preserve the context source order.
        2. Generate stable citation identifiers.
        3. Build human-readable source labels.
        4. Retain document, page, section, and chunk traceability.
        5. Provide constant-time citation lookup.

    This engine does not inspect or modify an LLM-generated answer. It builds
    the authoritative citation registry that later answer-generation and
    citation-validation stages will use.
    """

    CITATION_PREFIX = "SOURCE"

    def build(
        self,
        context: BuiltContext,
    ) -> CitationBundle:
        """
        Generate citations for every source included in the built context.
        """

        try:
            citations = tuple(
                self._build_citation(source)
                for source in context.sources
            )

            citation_index = {
                citation.citation_id: citation
                for citation in citations
            }

            if len(citation_index) != len(citations):
                raise CitationEngineError(
                    "Duplicate citation identifiers were generated."
                )

        except CitationEngineError:
            raise

        except Exception as exc:
            raise CitationEngineError(
                "The citation registry could not be generated."
            ) from exc

        return CitationBundle(
            project_id=context.project_id,
            query=context.query,
            citations=citations,
            citation_index=citation_index,
        )

    def _build_citation(
        self,
        source: ContextSource,
    ) -> Citation:
        citation_id = self._build_citation_id(
            source.context_index
        )

        return Citation(
            citation_id=citation_id,
            source_index=source.context_index,
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            retrieval_rank=source.rank,
            relevance_score=source.relevance_score,
            source_filename=source.source_filename,
            page_number=source.page_number,
            section_title=source.section_title,
            source_reference=source.source_reference,
            display_label=self._build_display_label(source),
            was_truncated=source.was_truncated,
        )

    def _build_citation_id(
        self,
        source_index: int,
    ) -> str:
        if source_index < 1:
            raise CitationEngineError(
                "Citation source index must be greater than zero."
            )

        return f"{self.CITATION_PREFIX} {source_index}"

    @staticmethod
    def _build_display_label(
        source: ContextSource,
    ) -> str:
        """
        Build a readable citation label without exposing storage paths.

        Examples:

            requirements.pdf, Page 12
            safety_analysis.docx, Section: Sensor Failure
            Document 0b7d..., Page 3
        """

        label_parts: list[str] = []

        if source.source_filename:
            label_parts.append(source.source_filename)
        else:
            label_parts.append(
                f"Document {str(source.document_id)[:8]}"
            )

        if source.page_number is not None:
            label_parts.append(
                f"Page {source.page_number}"
            )

        if source.section_title:
            label_parts.append(
                f"Section: {source.section_title}"
            )

        elif (
            source.source_reference
            and source.page_number is None
        ):
            label_parts.append(
                source.source_reference
            )

        return ", ".join(label_parts)