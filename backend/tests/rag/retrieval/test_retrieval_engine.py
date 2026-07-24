from uuid import uuid4

from app.rag.retrieval.retrieval_engine import (
    RetrievalEngine,
    RetrievalEngineRequest,
)


def test_retrieval_request_validation():
    request = RetrievalEngineRequest(
        query="vehicle architecture",
        top_k=5,
    )

    assert request.query == "vehicle architecture"
    assert request.top_k == 5


def test_retrieval_request_rejects_empty_query():
    try:
        RetrievalEngineRequest(
            query=" ",
        )
        assert False
    except ValueError:
        assert True