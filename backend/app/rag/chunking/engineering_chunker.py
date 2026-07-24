from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.chunking.models import (
    ChunkingResult,
    DocumentChunk,
)
from app.rag.extractors.base import ExtractionResult


class EngineeringDocumentChunker:
    """
    Split extracted engineering documents into traceable semantic chunks.

    The chunker processes each extracted section independently so that
    page numbers, headings, and section metadata are not mixed across
    unrelated document regions.
    """

    STRATEGY_NAME = "engineering-recursive-v1"

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_CHUNK_OVERLAP = 120
    DEFAULT_MINIMUM_CHUNK_LENGTH = 100

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        minimum_chunk_length: int = DEFAULT_MINIMUM_CHUNK_LENGTH,
    ) -> None:
        self._validate_configuration(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            minimum_chunk_length=minimum_chunk_length,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.minimum_chunk_length = minimum_chunk_length

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            keep_separator=True,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    def chunk(
        self,
        extraction: ExtractionResult,
        *,
        document_id: str,
        project_id: str,
        source_filename: str | None = None,
        document_hash: str | None = None,
    ) -> ChunkingResult:
        """
        Generate traceable chunks from a cleaned extraction result.

        Args:
            extraction:
                Cleaned document extraction result.

            document_id:
                Database identifier of the source document.

            project_id:
                Database identifier of the owning engineering project.

            source_filename:
                Original uploaded filename. Defaults to the extraction
                source path filename.

            document_hash:
                SHA-256 hash of the original uploaded document.

        Returns:
            ChunkingResult containing immutable DocumentChunk records.
        """

        normalized_document_id = self._normalize_required_identifier(
            document_id,
            field_name="document_id",
        )
        normalized_project_id = self._normalize_required_identifier(
            project_id,
            field_name="project_id",
        )

        filename = (
            source_filename.strip()
            if source_filename and source_filename.strip()
            else extraction.source_path.name
        )

        generated_chunks: list[DocumentChunk] = []
        global_chunk_index = 0

        for section_index, section in enumerate(extraction.sections):
            split_texts = self._split_section(section.text)

            for section_chunk_index, chunk_text in enumerate(split_texts):
                normalized_text = chunk_text.strip()

                if not normalized_text:
                    continue

                chunk_metadata = self._build_chunk_metadata(
                    extraction=extraction,
                    section_metadata=section.metadata,
                    section_index=section_index,
                    section_chunk_index=section_chunk_index,
                    document_hash=document_hash,
                )

                chunk_id = self._generate_chunk_id(
                    project_id=normalized_project_id,
                    document_id=normalized_document_id,
                    chunk_index=global_chunk_index,
                    chunk_text=normalized_text,
                )

                generated_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=normalized_text,
                        chunk_index=global_chunk_index,
                        document_id=normalized_document_id,
                        project_id=normalized_project_id,
                        source_filename=filename,
                        page_number=section.page_number,
                        heading=section.heading,
                        metadata=chunk_metadata,
                    )
                )

                global_chunk_index += 1

        if not generated_chunks:
            raise ValueError(
                f"No usable chunks were generated from "
                f"'{extraction.source_path.name}'."
            )

        return ChunkingResult(
            document_id=normalized_document_id,
            project_id=normalized_project_id,
            chunks=tuple(generated_chunks),
            chunking_strategy=self.STRATEGY_NAME,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={
                "source_filename": filename,
                "source_file_type": extraction.file_type,
                "source_path_name": Path(extraction.source_path).name,
                "extraction_method": extraction.extraction_method,
                "minimum_chunk_length": self.minimum_chunk_length,
                "section_count": extraction.section_count,
                "document_hash": document_hash,
            },
        )

    def _split_section(self, text: str) -> list[str]:
        """
        Split one extracted section without crossing its metadata boundary.

        Very small final chunks are merged into the previous chunk when
        possible to avoid weak, context-poor embeddings.
        """

        raw_chunks = [
            chunk.strip()
            for chunk in self._splitter.split_text(text)
            if chunk.strip()
        ]

        if len(raw_chunks) <= 1:
            return raw_chunks

        merged_chunks: list[str] = []

        for chunk in raw_chunks:
            if (
                len(chunk) < self.minimum_chunk_length
                and merged_chunks
            ):
                merged_chunks[-1] = (
                    f"{merged_chunks[-1].rstrip()}\n\n{chunk}"
                )
            else:
                merged_chunks.append(chunk)

        return merged_chunks

    def _build_chunk_metadata(
        self,
        *,
        extraction: ExtractionResult,
        section_metadata: dict[str, Any],
        section_index: int,
        section_chunk_index: int,
        document_hash: str | None,
    ) -> dict[str, Any]:
        """
        Build metadata required for retrieval, citations, and re-indexing.
        """

        metadata: dict[str, Any] = {
            "chunking_strategy": self.STRATEGY_NAME,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "section_index": section_index,
            "section_chunk_index": section_chunk_index,
            "file_type": extraction.file_type,
            "extraction_method": extraction.extraction_method,
        }

        metadata.update(section_metadata)

        if document_hash:
            metadata["document_hash"] = document_hash.strip()

        return metadata

    @staticmethod
    def _generate_chunk_id(
        *,
        project_id: str,
        document_id: str,
        chunk_index: int,
        chunk_text: str,
    ) -> str:
        """
        Generate a deterministic SHA-256 chunk identifier.

        Reprocessing unchanged content with the same identifiers and chunking
        strategy produces the same chunk ID, which helps prevent duplicates.
        """

        identity = (
            f"{EngineeringDocumentChunker.STRATEGY_NAME}|"
            f"{project_id}|"
            f"{document_id}|"
            f"{chunk_index}|"
            f"{chunk_text}"
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _normalize_required_identifier(
        value: str,
        *,
        field_name: str,
    ) -> str:
        normalized = str(value).strip()

        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")

        return normalized

    @staticmethod
    def _validate_configuration(
        *,
        chunk_size: int,
        chunk_overlap: int,
        minimum_chunk_length: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than the chunk size."
            )

        if minimum_chunk_length <= 0:
            raise ValueError(
                "Minimum chunk length must be greater than zero."
            )

        if minimum_chunk_length >= chunk_size:
            raise ValueError(
                "Minimum chunk length must be smaller than the chunk size."
            )


engineering_document_chunker = EngineeringDocumentChunker()