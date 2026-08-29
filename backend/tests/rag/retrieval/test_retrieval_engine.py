from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.rag.retrieval.retrieval_engine import (
    RetrievalEngine,
    RetrievalEngineRequest,
    RetrievalEngineResult,
)
from app.schemas.retrieval import RetrievedChunk


class TestRetrievalEngineRequestValidation:
    def test_valid_request_is_accepted(self) -> None:
        request = RetrievalEngineRequest(
            query="  vehicle communication failure  ",
        )

        assert request.query == "vehicle communication failure"
        assert request.top_k == 8
        assert request.score_threshold is None
        assert request.max_chunks_per_document == 4
        assert request.candidate_multiplier == 3
        assert request.deduplicate_content is True
        assert request.include_metadata is True

    @pytest.mark.parametrize(
        "query",
        [
            "",
            " ",
            "a",
        ],
    )
    def test_query_must_contain_meaningful_text(
        self,
        query: str,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="Retrieval query must contain meaningful text.",
        ):
            RetrievalEngineRequest(query=query)

    def test_top_k_cannot_be_zero(self) -> None:
        with pytest.raises(
            ValueError,
            match="top_k must be greater than zero.",
        ):
            RetrievalEngineRequest(
                query="vehicle communication",
                top_k=0,
            )

    def test_top_k_cannot_exceed_fifty(self) -> None:
        with pytest.raises(
            ValueError,
            match="top_k cannot exceed 50.",
        ):
            RetrievalEngineRequest(
                query="vehicle communication",
                top_k=51,
            )

    @pytest.mark.parametrize(
        "score_threshold",
        [
            -0.1,
            1.1,
        ],
    )
    def test_score_threshold_must_be_between_zero_and_one(
        self,
        score_threshold: float,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="score_threshold must be between 0 and 1.",
        ):
            RetrievalEngineRequest(
                query="vehicle communication",
                score_threshold=score_threshold,
            )

    def test_max_chunks_per_document_must_be_positive(self) -> None:
        with pytest.raises(
            ValueError,
            match="max_chunks_per_document must be greater than zero.",
        ):
            RetrievalEngineRequest(
                query="vehicle communication",
                max_chunks_per_document=0,
            )

    def test_candidate_multiplier_must_be_positive(self) -> None:
        with pytest.raises(
            ValueError,
            match="candidate_multiplier must be greater than zero.",
        ):
            RetrievalEngineRequest(
                query="vehicle communication",
                candidate_multiplier=0,
            )

    def test_candidate_multiplier_cannot_exceed_ten(self) -> None:
        with pytest.raises(
            ValueError,
            match="candidate_multiplier cannot exceed 10.",
        ):
            RetrievalEngineRequest(
                query="vehicle communication",
                candidate_multiplier=11,
            )

    def test_document_ids_are_deduplicated(self) -> None:
        document_id = uuid4()

        request = RetrievalEngineRequest(
            query="vehicle communication",
            document_ids=(
                document_id,
                document_id,
            ),
        )

        assert request.document_ids == (document_id,)


class TestRetrievalEngineResultValidation:
    def test_result_properties(self) -> None:
        project_id = uuid4()

        result = RetrievalEngineResult(
            project_id=project_id,
            query="vehicle communication",
            collection_name="project_collection",
            chunks=(),
            requested_top_k=8,
            candidate_count=0,
            retrieval_time_ms=12.5,
        )

        assert result.project_id == project_id
        assert result.query == "vehicle communication"
        assert result.collection_name == "project_collection"
        assert result.chunk_count == 0
        assert result.is_empty is True

    def test_result_rejects_empty_query(self) -> None:
        with pytest.raises(
            ValueError,
            match="Retrieval result query cannot be empty.",
        ):
            RetrievalEngineResult(
                project_id=uuid4(),
                query="",
                collection_name="project_collection",
                chunks=(),
                requested_top_k=8,
                candidate_count=0,
                retrieval_time_ms=10.0,
            )

    def test_result_rejects_empty_collection_name(self) -> None:
        with pytest.raises(
            ValueError,
            match="Retrieval result collection name cannot be empty.",
        ):
            RetrievalEngineResult(
                project_id=uuid4(),
                query="vehicle communication",
                collection_name="",
                chunks=(),
                requested_top_k=8,
                candidate_count=0,
                retrieval_time_ms=10.0,
            )

    def test_result_rejects_negative_candidate_count(self) -> None:
        with pytest.raises(
            ValueError,
            match="candidate_count cannot be negative.",
        ):
            RetrievalEngineResult(
                project_id=uuid4(),
                query="vehicle communication",
                collection_name="project_collection",
                chunks=(),
                requested_top_k=8,
                candidate_count=-1,
                retrieval_time_ms=10.0,
            )

    def test_result_rejects_negative_retrieval_time(self) -> None:
        with pytest.raises(
            ValueError,
            match="retrieval_time_ms cannot be negative.",
        ):
            RetrievalEngineResult(
                project_id=uuid4(),
                query="vehicle communication",
                collection_name="project_collection",
                chunks=(),
                requested_top_k=8,
                candidate_count=0,
                retrieval_time_ms=-1.0,
            )
class TestRetrievalEngineCandidateCalculation:
    def test_candidate_top_k_uses_multiplier(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        request = RetrievalEngineRequest(
            query="vehicle communication",
            top_k=8,
            candidate_multiplier=3,
        )

        assert engine._calculate_candidate_top_k(request) == 24

    def test_candidate_top_k_cannot_be_below_requested_top_k(
        self,
    ) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        request = RetrievalEngineRequest(
            query="vehicle communication",
            top_k=8,
            candidate_multiplier=1,
        )

        assert engine._calculate_candidate_top_k(request) == 8

    def test_candidate_top_k_is_capped_at_fifty(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        request = RetrievalEngineRequest(
            query="vehicle communication",
            top_k=50,
            candidate_multiplier=10,
        )

        assert engine._calculate_candidate_top_k(request) == 50

def make_chunk(
    *,
    chunk_id: str,
    document_id,
    content: str,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        rank=rank,
        distance=0.2,
        relevance_score=0.9,
    )


class TestRetrievalEngineDeduplication:
    def test_duplicate_chunk_ids_are_removed(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_id = uuid4()

        chunks = [
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=1,
            ),
            make_chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                content="Digital twin predicts vehicle movement.",
                rank=2,
            ),
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=3,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=None,
            deduplicate_content=True,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "chunk-1",
            "chunk-2",
        ]

    def test_duplicate_content_is_removed(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_id = uuid4()

        chunks = [
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=1,
            ),
            make_chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=2,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=None,
            deduplicate_content=True,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "chunk-1",
        ]

    def test_duplicate_content_is_case_and_whitespace_insensitive(
        self,
    ) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_id = uuid4()

        chunks = [
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=1,
            ),
            make_chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                content="  VEHICLE   communication   uses V2X.  ",
                rank=2,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=None,
            deduplicate_content=True,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "chunk-1",
        ]

    def test_content_deduplication_can_be_disabled(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_id = uuid4()

        chunks = [
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=1,
            ),
            make_chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                content="Vehicle communication uses V2X.",
                rank=2,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=None,
            deduplicate_content=False,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "chunk-1",
            "chunk-2",
        ]

class TestRetrievalEngineDocumentLimits:
    def test_max_chunks_per_document_is_enforced(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_a = uuid4()
        document_b = uuid4()

        chunks = [
            make_chunk(
                chunk_id="a-1",
                document_id=document_a,
                content="Vehicle communication requirement one.",
                rank=1,
            ),
            make_chunk(
                chunk_id="a-2",
                document_id=document_a,
                content="Vehicle communication requirement two.",
                rank=2,
            ),
            make_chunk(
                chunk_id="a-3",
                document_id=document_a,
                content="Vehicle communication requirement three.",
                rank=3,
            ),
            make_chunk(
                chunk_id="b-1",
                document_id=document_b,
                content="Digital twin requirement.",
                rank=4,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=2,
            deduplicate_content=True,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "a-1",
            "a-2",
            "b-1",
        ]

    def test_chunks_from_different_documents_are_allowed(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_a = uuid4()
        document_b = uuid4()

        chunks = [
            make_chunk(
                chunk_id="a-1",
                document_id=document_a,
                content="Requirement from document A.",
                rank=1,
            ),
            make_chunk(
                chunk_id="b-1",
                document_id=document_b,
                content="Requirement from document B.",
                rank=2,
            ),
            make_chunk(
                chunk_id="a-2",
                document_id=document_a,
                content="Another requirement from document A.",
                rank=3,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=2,
            deduplicate_content=True,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "a-1",
            "b-1",
            "a-2",
        ]

    def test_none_allows_unlimited_chunks_per_document(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_id = uuid4()

        chunks = [
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Requirement one.",
                rank=1,
            ),
            make_chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                content="Requirement two.",
                rank=2,
            ),
            make_chunk(
                chunk_id="chunk-3",
                document_id=document_id,
                content="Requirement three.",
                rank=3,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=10,
            max_chunks_per_document=None,
            deduplicate_content=True,
        )

        assert [chunk.chunk_id for chunk in result] == [
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ]

    def test_final_ranks_are_reassigned_after_filtering(self) -> None:
        engine = RetrievalEngine(
            semantic_search_service=Mock(),
        )

        document_id = uuid4()

        chunks = [
            make_chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                content="Requirement one.",
                rank=10,
            ),
            make_chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                content="Requirement two.",
                rank=20,
            ),
            make_chunk(
                chunk_id="chunk-3",
                document_id=document_id,
                content="Requirement three.",
                rank=30,
            ),
        ]

        result = engine._curate_chunks(
            chunks=chunks,
            top_k=2,
            max_chunks_per_document=None,
            deduplicate_content=True,
        )

        assert [chunk.rank for chunk in result] == [1, 2]