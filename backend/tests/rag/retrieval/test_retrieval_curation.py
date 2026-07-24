from uuid import uuid4

from app.rag.retrieval.retrieval_engine import RetrievalEngine
from app.schemas.retrieval import RetrievedChunk


def create_chunk(content, document_id, rank=1):
    return RetrievedChunk(
        chunk_id=str(uuid4()),
        document_id=document_id,
        content=content,
        rank=rank,
        distance=0.1,
        relevance_score=0.9,
        metadata=None,
    )


def test_duplicate_content_is_removed():
    engine = RetrievalEngine.__new__(RetrievalEngine)

    document_id = uuid4()

    chunks = [
        create_chunk(
            "ADAS vehicle architecture",
            document_id,
        ),
        create_chunk(
            "ADAS vehicle architecture",
            document_id,
        ),
    ]

    result = engine._curate_chunks(
        chunks=chunks,
        top_k=5,
        max_chunks_per_document=None,
        deduplicate_content=True,
    )

    assert len(result) == 1


def test_rank_is_reassigned():
    engine = RetrievalEngine.__new__(RetrievalEngine)

    chunks = [
        create_chunk("chunk one", uuid4()),
        create_chunk("chunk two", uuid4()),
    ]

    result = engine._curate_chunks(
        chunks=chunks,
        top_k=5,
        max_chunks_per_document=None,
        deduplicate_content=True,
    )

    assert result[0].rank == 1
    assert result[1].rank == 2