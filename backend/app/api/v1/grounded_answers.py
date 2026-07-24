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
from app.core.logging import logger
from app.database.session import get_db
from app.llm.dependencies import get_grounded_answer_service
from app.llm.grounded_answer_service import (
    GroundedAnswerRequest,
    GroundedAnswerResult,
    GroundedAnswerService,
    GroundedAnswerServiceError,
)
from app.models.document import Document, DocumentStatus
from app.models.project import ProjectStatus
from app.models.user import User
from app.schemas.grounded_answer import (
    GroundedAnswerQueryRequest,
    GroundedAnswerQueryResponse,
    UsedCitationResponse,
)
from app.services.document_service import build_project_collection_name
from app.services.project_service import get_project_by_id_for_user


router = APIRouter(
    prefix="/projects/{project_id}/answers",
    tags=["Grounded Answers"],
)


@router.post(
    "",
    response_model=GroundedAnswerQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a grounded engineering answer",
    responses={
        status.HTTP_200_OK: {
            "description": (
                "Grounded engineering answer generated successfully."
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
                "The request or selected documents cannot be processed."
            ),
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": (
                "The local language-model service is unavailable "
                "or answer generation failed."
            ),
        },
    },
)
async def generate_grounded_answer(
    project_id: UUID,
    request_data: GroundedAnswerQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    grounded_answer_service: GroundedAnswerService = Depends(
        get_grounded_answer_service
    ),
) -> GroundedAnswerQueryResponse:
    """
    Generate an engineering answer grounded in private project documents.

    Workflow:

    1. Authenticate the caller.
    2. Verify project ownership.
    3. Reject archived projects.
    4. Verify that indexed project documents exist.
    5. Validate any document-specific retrieval allowlist.
    6. Retrieve project-isolated evidence.
    7. Build an injection-resistant prompt.
    8. Generate an answer using the configured local LLM.
    9. Validate generated citation labels.
    10. Return an API-safe grounded answer.
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
                "Grounded answers cannot be generated "
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
                "Upload and successfully index a document before "
                "requesting a grounded answer."
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

    grounded_request = GroundedAnswerRequest(
        query=request_data.query,
        top_k=request_data.top_k,
        score_threshold=request_data.score_threshold,
        document_ids=validated_document_ids,
    )

    try:
        result = await run_in_threadpool(
            grounded_answer_service.answer,
            project_id=project_id,
            collection_name=collection_name,
            request=grounded_request,
        )

    except GroundedAnswerServiceError as exc:
        logger.exception(
            "Grounded answer generation failed for project %s: %s",
            project_id,
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The grounded engineering answer could not be generated. "
                "Check the backend logs for the underlying generation error."
            ),
        ) from exc

    except ValueError as exc:
        logger.warning(
            "Invalid grounded-answer request for project %s: %s",
            project_id,
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected grounded-answer failure for project %s: %s",
            project_id,
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred while generating "
                "the grounded answer."
            ),
        ) from exc

    return _build_grounded_answer_response(
        result=result,
        generated_at=datetime.now(timezone.utc),
    )


def _validate_document_selection(
    *,
    db: Session,
    project_id: UUID,
    requested_document_ids: list[UUID] | None,
) -> tuple[UUID, ...] | None:
    """
    Validate a document allowlist before retrieval.

    Every requested document must belong to the project, represent the latest
    version, and have completed indexing successfully.
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

    if any(
        document.status != DocumentStatus.INDEXED
        for document in documents
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "All selected documents must be successfully indexed "
                "before grounded answer generation."
            ),
        )

    return unique_document_ids


def _build_grounded_answer_response(
    *,
    result: GroundedAnswerResult,
    generated_at: datetime,
) -> GroundedAnswerQueryResponse:
    """
    Convert the internal grounded-answer result into an API-safe response.
    """

    used_citation_ids = result.used_citation_ids

    used_citations = [
        _build_used_citation_response(
            result=result,
            citation_id=citation_id,
        )
        for citation_id in used_citation_ids
    ]

    usage = result.generation.usage

    return GroundedAnswerQueryResponse(
        project_id=result.project_id,
        query=result.query,
        answer=result.answer,
        has_context=result.has_context,
        provider_name=result.provider_name,
        model_name=result.model_name,
        finish_reason=result.generation.finish_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        llm_latency_ms=result.generation.latency_ms,
        retrieval_time_ms=(
            result.rag_result.retrieval.retrieval_time_ms
        ),
        retrieved_chunk_count=(
            result.rag_result.retrieved_chunk_count
        ),
        included_chunk_count=(
            result.rag_result.included_chunk_count
        ),
        used_citation_ids=list(used_citation_ids),
        used_citations=used_citations,
        citation_count=len(used_citations),
        generated_at=generated_at,
    )


def _build_used_citation_response(
    *,
    result: GroundedAnswerResult,
    citation_id: str,
) -> UsedCitationResponse:
    """
    Resolve one citation used by the generated answer.
    """

    citation = result.rag_result.citations.get(
        citation_id
    )

    if citation is None:
        raise ValueError(
            f"Used citation '{citation_id}' was not found "
            "in the authoritative citation registry."
        )

    return UsedCitationResponse(
        citation_id=citation.citation_id,
        source_index=citation.source_index,
        document_id=citation.document_id,
        chunk_id=citation.chunk_id,
        display_label=citation.display_label,
        source_filename=citation.source_filename,
        page_number=citation.page_number,
        section_title=citation.section_title,
        source_reference=citation.source_reference,
        relevance_score=citation.relevance_score,
    )