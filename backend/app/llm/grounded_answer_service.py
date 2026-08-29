from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.llm import (
    BaseLLMProvider,
    LLMGenerationError,
    LLMGenerationRequest,
    LLMGenerationResult,
)
from app.llm.citation_validator import (
    CitationValidationError,
    CitationValidationResult,
    CitationValidator,
)
from app.llm.prompt_builder import (
    EngineeringPrompt,
    EngineeringPromptBuilder,
    PromptBuilderError,
)
from app.rag.retrieval.private_rag_service import (
    PrivateRAGRequest,
    PrivateRAGResult,
    PrivateRAGService,
    PrivateRAGServiceError,
)


class GroundedAnswerServiceError(Exception):
    """
    Raised when AutoMind cannot generate a safe grounded answer.
    """


@dataclass(frozen=True, slots=True)
class GroundedAnswerRequest:
    """
    Internal request for one project-grounded engineering answer.

    Retrieval and generation controls are trusted application settings.
    API clients should not receive unrestricted access to provider-specific
    generation parameters.
    """

    query: str

    top_k: int = 3

    score_threshold: float | None = None

    document_ids: tuple[UUID, ...] | None = None

    temperature: float = 0.1

    max_output_tokens: int = 400

    require_citations: bool = True

    def __post_init__(self) -> None:
        normalized_query = " ".join(
            str(self.query).split()
        )

        if len(normalized_query) < 2:
            raise ValueError(
                "Grounded answer query must contain meaningful text."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.top_k > 20:
            raise ValueError(
                "top_k cannot exceed 20."
            )

        if (
            self.score_threshold is not None
            and not 0.0 <= self.score_threshold <= 1.0
        ):
            raise ValueError(
                "score_threshold must be between 0 and 1."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if self.max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )

        if self.max_output_tokens > 4096:
            raise ValueError(
                "max_output_tokens cannot exceed 4096."
            )

        normalized_document_ids = (
            tuple(dict.fromkeys(self.document_ids))
            if self.document_ids
            else None
        )

        object.__setattr__(
            self,
            "query",
            normalized_query,
        )

        object.__setattr__(
            self,
            "document_ids",
            normalized_document_ids,
        )


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    """
    Complete grounded-answer result returned by AutoMind.

    The result preserves the entire evidence chain:

        query
            -> Private RAG evidence
            -> engineering prompt
            -> LLM generation
            -> citation validation
    """

    project_id: UUID

    query: str

    answer: str

    rag_result: PrivateRAGResult

    prompt: EngineeringPrompt

    generation: LLMGenerationResult

    citation_validation: CitationValidationResult

    def __post_init__(self) -> None:
        normalized_query = self.query.strip()
        normalized_answer = self.answer.strip()

        if not normalized_query:
            raise ValueError(
                "Grounded answer result query cannot be empty."
            )

        if not normalized_answer:
            raise ValueError(
                "Grounded answer result answer cannot be empty."
            )

        if self.rag_result.project_id != self.project_id:
            raise ValueError(
                "RAG result project does not match "
                "the grounded answer project."
            )

        if self.prompt.project_id != self.project_id:
            raise ValueError(
                "Prompt project does not match "
                "the grounded answer project."
            )

        if self.rag_result.query != normalized_query:
            raise ValueError(
                "RAG result query does not match "
                "the grounded answer query."
            )

        if self.prompt.query != normalized_query:
            raise ValueError(
                "Prompt query does not match "
                "the grounded answer query."
            )

        if self.generation.content != normalized_answer:
            raise ValueError(
                "Generated content does not match "
                "the grounded answer text."
            )

        if self.citation_validation.answer != normalized_answer:
            raise ValueError(
                "Citation validation answer does not match "
                "the grounded answer text."
            )

        if not self.citation_validation.is_valid:
            raise ValueError(
                "Grounded answer cannot contain invalid citations."
            )

        object.__setattr__(
            self,
            "query",
            normalized_query,
        )

        object.__setattr__(
            self,
            "answer",
            normalized_answer,
        )

    @property
    def has_context(self) -> bool:
        return self.rag_result.has_context

    @property
    def used_citation_ids(self) -> tuple[str, ...]:
        return self.citation_validation.used_citation_ids

    @property
    def model_name(self) -> str:
        return self.generation.model_name

    @property
    def provider_name(self) -> str:
        return self.generation.provider_name


class GroundedAnswerService:
    """
    Coordinates AutoMind's complete grounded answer workflow.

    Workflow:

        1. Retrieve project-isolated evidence.
        2. Build an injection-resistant engineering prompt.
        3. Generate a response through the configured LLM provider.
        4. Validate all generated citations.
        5. Return a traceable grounded-answer result.

    This service remains independent of FastAPI, SQLAlchemy, and chat
    persistence so it can later be reused by APIs, LangGraph agents,
    evaluation pipelines, and background workers.
    """

    def __init__(
        self,
        *,
        private_rag_service: PrivateRAGService,
        prompt_builder: EngineeringPromptBuilder,
        llm_provider: BaseLLMProvider,
        citation_validator: CitationValidator,
    ) -> None:
        self._private_rag_service = private_rag_service
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._citation_validator = citation_validator

    def answer(
        self,
        *,
        project_id: UUID,
        collection_name: str,
        request: GroundedAnswerRequest,
    ) -> GroundedAnswerResult:
        """
        Generate one project-grounded engineering answer.

        Authentication, project authorization, and document ownership
        validation must be completed by the calling API or application layer.
        """

        rag_request = PrivateRAGRequest(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_ids=request.document_ids,
        )

        try:
            rag_result = self._private_rag_service.prepare(
                project_id=project_id,
                collection_name=collection_name,
                request=rag_request,
            )

            prompt = self._prompt_builder.build(
                rag_result
            )

            generation_request = LLMGenerationRequest(
                messages=prompt.messages,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                response_format="text",
                request_metadata={
                    "project_id": str(project_id),
                    "query": request.query,
                    "has_context": rag_result.has_context,
                },
            )

            generation_result = self._llm_provider.generate(
                generation_request
            )

            citation_validation = (
                self._citation_validator.ensure_valid(
                    answer=generation_result.content,
                    available_citation_ids=(
                        prompt.available_citation_ids
                    ),
                    require_citations=(
                        request.require_citations
                        and rag_result.has_context
                    ),
                )
            )

        except PrivateRAGServiceError as exc:
            raise GroundedAnswerServiceError(
                "Project evidence retrieval failed."
            ) from exc

        except PromptBuilderError as exc:
            raise GroundedAnswerServiceError(
                "The engineering prompt could not be created."
            ) from exc

        except LLMGenerationError as exc:
            raise GroundedAnswerServiceError(
                "The configured language model could not generate "
                "an engineering response."
            ) from exc

        except CitationValidationError as exc:
            raise GroundedAnswerServiceError(
                "The generated answer failed citation validation."
            ) from exc

        except GroundedAnswerServiceError:
            raise

        except Exception as exc:
            raise GroundedAnswerServiceError(
                "Grounded answer generation failed."
            ) from exc

        return GroundedAnswerResult(
            project_id=project_id,
            query=request.query,
            answer=generation_result.content,
            rag_result=rag_result,
            prompt=prompt,
            generation=generation_result,
            citation_validation=citation_validation,
        )