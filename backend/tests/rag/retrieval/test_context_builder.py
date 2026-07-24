from uuid import uuid4

from app.rag.retrieval.context_builder import (
    ContextBuilder,
    ContextBuilderConfig,
)
from app.rag.retrieval.retrieval_engine import (
    RetrievalEngineResult,
)
from app.schemas.retrieval import RetrievedChunk


def create_chunk(content):
    return RetrievedChunk(
        chunk_id=str(uuid4()),
        document_id=uuid4(),
        content=content,
        rank=1,
        distance=0.1,
        relevance_score=0.9,
        metadata=None,
    )


def test_context_builder_creates_context():
    builder = ContextBuilder()

    retrieval_result = RetrievalEngineResult(
        project_id=uuid4(),
        query="vehicle safety architecture",
        collection_name="project_test",
        chunks=(
            create_chunk(
                "The vehicle uses sensor fusion architecture."
            ),
        ),
        requested_top_k=5,
        candidate_count=1,
        retrieval_time_ms=10,
    )

    context = builder.build(retrieval_result)

    assert context.included_chunk_count == 1
    assert "sensor fusion architecture" in context.text


def test_context_builder_limits_large_chunks():
    builder = ContextBuilder(
        config=ContextBuilderConfig(
            max_chunk_characters=50,
            max_context_characters=500,
        )
    )

    retrieval_result = RetrievalEngineResult(
        project_id=uuid4(),
        query="ADAS",
        collection_name="project_test",
        chunks=(
            create_chunk(
                "A" * 200
            ),
        ),
        requested_top_k=5,
        candidate_count=1,
        retrieval_time_ms=10,
    )

    context = builder.build(retrieval_result)

    assert context.truncated_chunk_count == 1