from __future__ import annotations

from functools import lru_cache

from app.llm.citation_validator import CitationValidator
from app.llm.grounded_answer_service import GroundedAnswerService
from app.llm.ollama_provider import OllamaProvider
from app.llm.prompt_builder import EngineeringPromptBuilder
from app.rag.retrieval.dependencies import get_private_rag_service


@lru_cache(maxsize=1)
def get_ollama_provider() -> OllamaProvider:
    """
    Return the singleton local Ollama provider.

    The provider is stateless and reuses the same configuration across
    requests. Ollama itself manages model loading and inference lifecycle.
    """

    return OllamaProvider()


@lru_cache(maxsize=1)
def get_engineering_prompt_builder() -> EngineeringPromptBuilder:
    """
    Return the singleton engineering prompt builder.
    """

    return EngineeringPromptBuilder()


@lru_cache(maxsize=1)
def get_citation_validator() -> CitationValidator:
    """
    Return the singleton citation validator.
    """

    return CitationValidator()


@lru_cache(maxsize=1)
def get_grounded_answer_service() -> GroundedAnswerService:
    """
    Return the main grounded-answer service used by AutoMind APIs and agents.

    The service combines:

        - project-isolated Private RAG,
        - engineering prompt construction,
        - local Ollama inference,
        - citation validation.
    """

    return GroundedAnswerService(
        private_rag_service=get_private_rag_service(),
        prompt_builder=get_engineering_prompt_builder(),
        llm_provider=get_ollama_provider(),
        citation_validator=get_citation_validator(),
    )