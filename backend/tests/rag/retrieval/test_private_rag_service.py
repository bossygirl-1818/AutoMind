from uuid import uuid4

from app.rag.retrieval.private_rag_service import (
    PrivateRAGRequest,
    PrivateRAGService,
)
from app.rag.retrieval.retrieval_engine import (
    RetrievalEngineResult,
)
from app.rag.retrieval.context_builder import (
    BuiltContext,
)
from app.rag.retrieval.citation_engine import (
    CitationBundle,
)


class FakeRetrievalEngine:
    def retrieve(
        self,
        *,
        project_id,
        collection_name,
        request,
    ):
        return RetrievalEngineResult(
            project_id=project_id,
            query=request.query,
            collection_name=collection_name,
            chunks=(),
            requested_top_k=request.top_k,
            candidate_count=0,
            retrieval_time_ms=5.0,
        )


class FakeContextBuilder:
    def build(
        self,
        retrieval_result,
    ):
        return BuiltContext(
            project_id=retrieval_result.project_id,
            query=retrieval_result.query,
            text="No relevant context",
            sources=(),
            total_characters=20,
            available_chunk_count=0,
            included_chunk_count=0,
            truncated_chunk_count=0,
            context_was_limited=False,
        )


class FakeCitationEngine:
    def build(
        self,
        context,
    ):
        return CitationBundle(
            project_id=context.project_id,
            query=context.query,
            citations=(),
            citation_index={},
        )


def test_private_rag_service_pipeline():
    service = PrivateRAGService(
        retrieval_engine=FakeRetrievalEngine(),
        context_builder=FakeContextBuilder(),
        citation_engine=FakeCitationEngine(),
    )

    project_id = uuid4()

    result = service.prepare(
        project_id=project_id,
        collection_name="test_collection",
        request=PrivateRAGRequest(
            query="vehicle architecture",
        ),
    )

    assert result.project_id == project_id
    assert result.query == "vehicle architecture"
    assert result.collection_name == "test_collection"


def test_private_rag_service_empty_context():
    service = PrivateRAGService(
        retrieval_engine=FakeRetrievalEngine(),
        context_builder=FakeContextBuilder(),
        citation_engine=FakeCitationEngine(),
    )

    result = service.prepare(
        project_id=uuid4(),
        collection_name="automind",
        request=PrivateRAGRequest(
            query="test query",
        ),
    )

    assert result.has_context is False
    assert result.retrieved_chunk_count == 0
    assert result.citation_count == 0