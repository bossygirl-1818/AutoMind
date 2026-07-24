from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """
    Represents one traceable text chunk produced from an extracted document.

    Every chunk retains enough metadata to support semantic retrieval,
    source citations, project isolation, re-indexing, and auditability.
    """

    chunk_id: str
    text: str
    chunk_index: int
    document_id: str
    project_id: str
    source_filename: str
    page_number: int | None = None
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_chunk_id = self.chunk_id.strip()
        normalized_text = self.text.strip()
        normalized_document_id = self.document_id.strip()
        normalized_project_id = self.project_id.strip()
        normalized_source_filename = self.source_filename.strip()

        if not normalized_chunk_id:
            raise ValueError("Chunk ID cannot be empty.")

        if not normalized_text:
            raise ValueError("Chunk text cannot be empty.")

        if self.chunk_index < 0:
            raise ValueError("Chunk index cannot be negative.")

        if not normalized_document_id:
            raise ValueError("Document ID cannot be empty.")

        if not normalized_project_id:
            raise ValueError("Project ID cannot be empty.")

        if not normalized_source_filename:
            raise ValueError("Source filename cannot be empty.")

        if self.page_number is not None and self.page_number < 1:
            raise ValueError("Page number must be greater than or equal to 1.")

        object.__setattr__(self, "chunk_id", normalized_chunk_id)
        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "document_id", normalized_document_id)
        object.__setattr__(self, "project_id", normalized_project_id)
        object.__setattr__(
            self,
            "source_filename",
            normalized_source_filename,
        )

    @property
    def character_count(self) -> int:
        """Return the number of characters in the chunk."""

        return len(self.text)


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """
    Immutable result returned by the AutoMind chunking pipeline.
    """

    document_id: str
    project_id: str
    chunks: tuple[DocumentChunk, ...]
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not self.project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        if not self.chunks:
            raise ValueError("Chunking result must contain at least one chunk.")

        if not self.chunking_strategy.strip():
            raise ValueError("Chunking strategy cannot be empty.")

        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than the chunk size."
            )

    @property
    def chunk_count(self) -> int:
        """Return the total number of generated chunks."""

        return len(self.chunks)

    @property
    def total_characters(self) -> int:
        """Return the total number of characters across all chunks."""

        return sum(chunk.character_count for chunk in self.chunks)