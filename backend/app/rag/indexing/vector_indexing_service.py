from __future__ import annotations

from app.rag.chunking.models import ChunkingResult
from app.rag.embeddings import BaseEmbeddingProvider
from app.rag.vectorstore import (
    BaseVectorStore,
    VectorRecord,
)


class VectorIndexingService:
    """
    Converts document chunks into embeddings and stores them
    inside the configured vector database.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def index_chunks(
        self,
        *,
        collection_name: str,
        chunking_result: ChunkingResult,
    ) -> None:
        """
        Generate embeddings for every chunk and
        store them inside the vector database.
        """

        texts = [
            chunk.text
            for chunk in chunking_result.chunks
        ]

        embedding_result = (
            self._embedding_provider.embed_documents(texts)
        )

        records: list[VectorRecord] = []

        for chunk, embedding in zip(
            chunking_result.chunks,
            embedding_result.vectors,
        ):
            metadata = {
                **chunk.metadata,
                "project_id": chunk.project_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "heading": chunk.heading,
                "source_filename": chunk.source_filename,
                "embedding_model": embedding_result.model_name,
                "embedding_provider": embedding_result.provider_name,
            }

            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    embedding=embedding,
                    document=chunk.text,
                    metadata=metadata,
                )
            )

        self._vector_store.upsert(
            collection_name=collection_name,
            records=records,
        )