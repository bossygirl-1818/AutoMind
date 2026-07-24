from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.chunking import (
    ChunkingResult,
    EngineeringDocumentChunker,
    engineering_document_chunker,
)
from app.rag.cleaning.text_cleaner import TextCleaner, text_cleaner
from app.rag.extractors import (
    DocumentExtractionService,
    ExtractionResult,
    document_extraction_service,
)
from app.rag.indexing.vector_indexing_service import VectorIndexingService


class DocumentIndexingError(Exception):
    """Raised when the document indexing pipeline cannot complete safely."""


@dataclass(frozen=True, slots=True)
class DocumentIndexingResult:
    """
    Summary returned after a document has been successfully indexed.
    """

    project_id: str
    document_id: str
    source_filename: str
    collection_name: str
    extracted_sections: int
    extracted_characters: int
    generated_chunks: int
    chunking_strategy: str

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        if not self.document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not self.source_filename.strip():
            raise ValueError("Source filename cannot be empty.")

        if not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        if self.extracted_sections <= 0:
            raise ValueError(
                "Extracted section count must be greater than zero."
            )

        if self.extracted_characters <= 0:
            raise ValueError(
                "Extracted character count must be greater than zero."
            )

        if self.generated_chunks <= 0:
            raise ValueError(
                "Generated chunk count must be greater than zero."
            )


class DocumentIndexingPipeline:
    """
    Orchestrates the complete AutoMind document-intelligence workflow.

    Pipeline:

        source file
            -> extraction
            -> text cleaning
            -> engineering chunking
            -> embedding generation
            -> vector storage

    The pipeline depends on abstractions and services rather than directly
    depending on PyMuPDF, Sentence Transformers, or ChromaDB.
    """

    def __init__(
        self,
        *,
        extraction_service: DocumentExtractionService,
        cleaner: TextCleaner,
        chunker: EngineeringDocumentChunker,
        vector_indexing_service: VectorIndexingService,
    ) -> None:
        self._extraction_service = extraction_service
        self._cleaner = cleaner
        self._chunker = chunker
        self._vector_indexing_service = vector_indexing_service

    def index_document(
        self,
        *,
        file_path: str | Path,
        document_id: str,
        project_id: str,
        collection_name: str,
        source_filename: str | None = None,
        document_hash: str | None = None,
    ) -> DocumentIndexingResult:
        """
        Process and index one uploaded AutoMind document.

        Args:
            file_path:
                Secure storage path of the uploaded document.

            document_id:
                Database identifier of the source document.

            project_id:
                Database identifier of the owning project.

            collection_name:
                ChromaDB collection where vectors will be stored.

            source_filename:
                Original uploaded filename shown in citations.

            document_hash:
                SHA-256 hash previously calculated during upload.

        Returns:
            DocumentIndexingResult containing processing statistics.
        """

        path = Path(file_path).expanduser().resolve()

        normalized_document_id = self._normalize_identifier(
            document_id,
            field_name="document_id",
        )
        normalized_project_id = self._normalize_identifier(
            project_id,
            field_name="project_id",
        )
        normalized_collection_name = self._normalize_collection_name(
            collection_name
        )

        filename = (
            source_filename.strip()
            if source_filename and source_filename.strip()
            else path.name
        )

        try:
            extraction = self._extract_document(path)
            cleaned_extraction = self._clean_document(extraction)

            chunking_result = self._chunk_document(
                extraction=cleaned_extraction,
                document_id=normalized_document_id,
                project_id=normalized_project_id,
                source_filename=filename,
                document_hash=document_hash,
            )

            self._index_chunks(
                collection_name=normalized_collection_name,
                chunking_result=chunking_result,
            )

        except DocumentIndexingError:
            raise

        except Exception as exc:
            raise DocumentIndexingError(
                f"Failed to index document '{filename}'."
            ) from exc

        return DocumentIndexingResult(
            project_id=normalized_project_id,
            document_id=normalized_document_id,
            source_filename=filename,
            collection_name=normalized_collection_name,
            extracted_sections=cleaned_extraction.section_count,
            extracted_characters=cleaned_extraction.total_characters,
            generated_chunks=chunking_result.chunk_count,
            chunking_strategy=chunking_result.chunking_strategy,
        )

    def _extract_document(
        self,
        file_path: Path,
    ) -> ExtractionResult:
        """Extract structured text from the source document."""

        try:
            return self._extraction_service.extract(file_path)
        except Exception as exc:
            raise DocumentIndexingError(
                f"Text extraction failed for '{file_path.name}'."
            ) from exc

    def _clean_document(
        self,
        extraction: ExtractionResult,
    ) -> ExtractionResult:
        """Clean extracted text without altering technical meaning."""

        try:
            return self._cleaner.clean(extraction)
        except Exception as exc:
            raise DocumentIndexingError(
                f"Text cleaning failed for "
                f"'{extraction.source_path.name}'."
            ) from exc

    def _chunk_document(
        self,
        *,
        extraction: ExtractionResult,
        document_id: str,
        project_id: str,
        source_filename: str,
        document_hash: str | None,
    ) -> ChunkingResult:
        """Create traceable engineering chunks."""

        try:
            return self._chunker.chunk(
                extraction,
                document_id=document_id,
                project_id=project_id,
                source_filename=source_filename,
                document_hash=document_hash,
            )
        except Exception as exc:
            raise DocumentIndexingError(
                f"Chunk generation failed for '{source_filename}'."
            ) from exc

    def _index_chunks(
        self,
        *,
        collection_name: str,
        chunking_result: ChunkingResult,
    ) -> None:
        """Generate embeddings and persist them in the vector store."""

        try:
            self._vector_indexing_service.index_chunks(
                collection_name=collection_name,
                chunking_result=chunking_result,
            )
        except Exception as exc:
            raise DocumentIndexingError(
                f"Vector indexing failed for document "
                f"'{chunking_result.document_id}'."
            ) from exc

    @staticmethod
    def _normalize_identifier(
        value: str,
        *,
        field_name: str,
    ) -> str:
        normalized = str(value).strip()

        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")

        return normalized

    @staticmethod
    def _normalize_collection_name(
        collection_name: str,
    ) -> str:
        """
        Validate the collection name before sending it to ChromaDB.

        Project-specific collection names should be deterministic and should
        contain only characters supported by the vector-store naming rules.
        """

        normalized = str(collection_name).strip().lower()

        if not normalized:
            raise ValueError("Collection name cannot be empty.")

        if len(normalized) < 3:
            raise ValueError(
                "Collection name must contain at least three characters."
            )

        if len(normalized) > 512:
            raise ValueError(
                "Collection name cannot exceed 512 characters."
            )

        return normalized


def build_document_indexing_pipeline(
    vector_indexing_service: VectorIndexingService,
) -> DocumentIndexingPipeline:
    """
    Build the document-indexing pipeline using AutoMind's default services.

    VectorIndexingService is supplied explicitly because it requires the
    configured embedding provider and vector-store implementation.
    """

    return DocumentIndexingPipeline(
        extraction_service=document_extraction_service,
        cleaner=text_cleaner,
        chunker=engineering_document_chunker,
        vector_indexing_service=vector_indexing_service,
    )