from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps.auth import get_current_user
from app.database.session import get_db
from app.models.document import Document, DocumentStatus
from app.models.project import ProjectStatus
from app.models.user import User
from app.rag.retrieval.dependencies import get_private_rag_service
from app.rag.retrieval.private_rag_service import (
    PrivateRAGRequest,
    PrivateRAGResult,
    PrivateRAGService,
    PrivateRAGServiceError,
)
from app.schemas.private_rag import (
    PrivateRAGQueryRequest,
    PrivateRAGQueryResponse,
    RAGCitationResponse,
    RAGContextSourceResponse,
    RAGRetrievedChunkResponse,
)
from app.services.document_service import build_project_collection_name
from app.services.project_service import get_project_by_id_for_user


router = APIRouter(
    prefix="/projects/{project_id}/retrieval",
    tags=["Private RAG"],
)


@router.post(
    "/search",
    response_model=PrivateRAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve private engineering context",
    responses={
        status.HTTP_200_OK: {
            "description": (
                "Project-isolated engineering evidence retrieved successfully."
            ),
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": (
                "Authentication credentials are missing or invalid."
            ),
        },
        status.HTTP_404_NOT_FOUND: {
            "description": (
                "Project workspace or requested document was not found."
            ),
        },
        status.HTTP_409_CONFLICT: {
            "description": (
                "The project is archived or contains no indexed documents."
            ),
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": (
                "The retrieval request or requested document selection "
                "cannot be processed."
            ),
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": (
                "Private RAG evidence preparation failed."
            ),
        },
    },
)
async def search_project_knowledge(
    project_id: UUID,
    request_data: PrivateRAGQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    private_rag_service: PrivateRAGService = Depends(
        get_private_rag_service
    ),
) -> PrivateRAGQueryResponse:
    """
    Retrieve project-isolated engineering evidence for an authenticated user.

    Workflow:

    1. Authenticate the caller.
    2. Verify project ownership.
    3. Reject archived projects.
    4. Verify that the project contains indexed documents.
    5. Validate any document-specific retrieval allowlist.
    6. Build the deterministic project ChromaDB collection name.
    7. Execute the blocking retrieval pipeline in a worker thread.
    8. Convert the internal Private RAG result into an API-safe response.

    This endpoint prepares retrieval evidence, bounded LLM context, and
    citations. It does not call an LLM or generate a final answer.
    """

    project = get_project_by_id_for_user(
        db=db,
        project_id=project_id,
        current_user=current_user,
    )

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project workspace not found.",
        )

    if project.status == ProjectStatus.ARCHIVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Private retrieval cannot be performed "
                "for an archived project."
            ),
        )

    indexed_document_count = (
        db.query(Document)
        .filter(
            Document.project_id == project_id,
            Document.status == DocumentStatus.INDEXED,
            Document.is_latest.is_(True),
        )
        .count()
    )

    if indexed_document_count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The project does not contain any indexed documents. "
                "Upload and successfully index a document before searching."
            ),
        )

    validated_document_ids = _validate_document_selection(
        db=db,
        project_id=project_id,
        requested_document_ids=request_data.document_ids,
    )

    collection_name = build_project_collection_name(
        project_id
    )

    rag_request = PrivateRAGRequest(
        query=request_data.query,
        top_k=request_data.top_k,
        score_threshold=request_data.score_threshold,
        document_ids=validated_document_ids,
    )

    try:
        rag_result = await run_in_threadpool(
            private_rag_service.prepare,
            project_id=project_id,
            collection_name=collection_name,
            request=rag_request,
        )

    except PrivateRAGServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Private RAG evidence preparation could not be completed."
            ),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred during private retrieval."
            ),
        ) from exc

    return _build_private_rag_response(
        result=rag_result,
        prepared_at=datetime.now(timezone.utc),
    )


def _validate_document_selection(
    *,
    db: Session,
    project_id: UUID,
    requested_document_ids: list[UUID] | None,
) -> tuple[UUID, ...] | None:
    """
    Validate a client-supplied document retrieval allowlist.

    Every selected document must:

        - belong to the requested project,
        - represent the latest document version,
        - have completed indexing successfully.

    A generic not-found response is used for inaccessible or foreign
    documents so the endpoint does not reveal cross-project metadata.
    """

    if not requested_document_ids:
        return None

    unique_document_ids = tuple(
        dict.fromkeys(requested_document_ids)
    )

    documents = (
        db.query(Document)
        .filter(
            Document.id.in_(unique_document_ids),
            Document.project_id == project_id,
            Document.is_latest.is_(True),
        )
        .all()
    )

    documents_by_id = {
        document.id: document
        for document in documents
    }

    missing_document_ids = [
        document_id
        for document_id in unique_document_ids
        if document_id not in documents_by_id
    ]

    if missing_document_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "One or more requested documents were not found "
                "in this project workspace."
            ),
        )

    non_indexed_documents = [
        document
        for document in documents
        if document.status != DocumentStatus.INDEXED
    ]

    if non_indexed_documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "All selected documents must be successfully indexed "
                "before they can be used for semantic retrieval."
            ),
        )

    return unique_document_ids

def _build_private_rag_response(
    *,
    result: PrivateRAGResult,
    prepared_at: datetime,
) -> PrivateRAGQueryResponse:
    """
    Convert the internal PrivateRAGResult into a stable API response.

    Internal dataclasses are mapped explicitly so implementation details
    such as citation lookup dictionaries and service objects are never
    exposed through the HTTP boundary.
    """

    retrieved_chunks = [
        _build_retrieved_chunk_response(chunk)
        for chunk in result.retrieval.chunks
    ]

    context_sources = [
        _build_context_source_response(source)
        for source in result.context.sources
    ]

    citations = [
        _build_citation_response(citation)
        for citation in result.citations.citations
    ]

    return PrivateRAGQueryResponse(
        project_id=result.project_id,
        query=result.query,
        collection_name=result.collection_name,
        has_context=result.has_context,
        retrieved_chunks=retrieved_chunks,
        retrieved_chunk_count=result.retrieved_chunk_count,
        candidate_count=result.retrieval.candidate_count,
        retrieval_time_ms=result.retrieval.retrieval_time_ms,
        context_text=result.context.text,
        context_sources=context_sources,
        context_character_count=result.context.total_characters,
        included_chunk_count=result.included_chunk_count,
        truncated_chunk_count=result.context.truncated_chunk_count,
        context_was_limited=result.context.context_was_limited,
        citations=citations,
        citation_count=result.citation_count,
        prepared_at=prepared_at,
    )


def _build_retrieved_chunk_response(
    chunk,
) -> RAGRetrievedChunkResponse:
    """
    Map one internal retrieved chunk into its public representation.
    """

    metadata = chunk.metadata

    return RAGRetrievedChunkResponse(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        rank=chunk.rank,
        relevance_score=chunk.relevance_score,
        distance=chunk.distance,
        content=chunk.content,
        source_filename=(
            metadata.original_filename
            if metadata is not None
            else None
        ),
        page_number=(
            metadata.page_number
            if metadata is not None
            else None
        ),
        section_title=(
            metadata.section_title
            if metadata is not None
            else None
        ),
        source_reference=(
            metadata.source_reference
            if metadata is not None
            else None
        ),
    )


def _build_context_source_response(
    source,
) -> RAGContextSourceResponse:
    """
    Map one context source into a public traceability record.
    """

    return RAGContextSourceResponse(
        source_index=source.context_index,
        source_label=f"SOURCE {source.context_index}",
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        retrieval_rank=source.rank,
        relevance_score=source.relevance_score,
        source_filename=source.source_filename,
        page_number=source.page_number,
        section_title=source.section_title,
        source_reference=source.source_reference,
        included_characters=source.included_characters,
        was_truncated=source.was_truncated,
    )


def _build_citation_response(
    citation,
) -> RAGCitationResponse:
    """
    Map one authoritative citation into its API-safe representation.
    """

    return RAGCitationResponse(
        citation_id=citation.citation_id,
        source_index=citation.source_index,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        retrieval_rank=citation.retrieval_rank,
        relevance_score=citation.relevance_score,
        display_label=citation.display_label,
        source_filename=citation.source_filename,
        page_number=citation.page_number,
        section_title=citation.section_title,
        source_reference=citation.source_reference,
        was_truncated=citation.was_truncated,
    )