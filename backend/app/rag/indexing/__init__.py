from app.rag.embeddings import local_embedding_provider
from app.rag.indexing.document_indexing_pipeline import (
    DocumentIndexingError,
    DocumentIndexingPipeline,
    DocumentIndexingResult,
    build_document_indexing_pipeline,
)
from app.rag.indexing.vector_indexing_service import VectorIndexingService
from app.rag.vectorstore import chroma_vector_store


vector_indexing_service = VectorIndexingService(
    embedding_provider=local_embedding_provider,
    vector_store=chroma_vector_store,
)

document_indexing_pipeline = build_document_indexing_pipeline(
    vector_indexing_service=vector_indexing_service,
)


__all__ = [
    "DocumentIndexingError",
    "DocumentIndexingPipeline",
    "DocumentIndexingResult",
    "VectorIndexingService",
    "build_document_indexing_pipeline",
    "document_indexing_pipeline",
    "vector_indexing_service",
]