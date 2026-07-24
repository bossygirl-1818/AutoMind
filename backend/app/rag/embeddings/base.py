from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


class EmbeddingGenerationError(Exception):
    """Raised when an embedding provider cannot generate embeddings."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """
    Represents embeddings generated for one or more text inputs.
    """

    vectors: tuple[tuple[float, ...], ...]
    model_name: str
    dimensions: int
    provider_name: str

    def __post_init__(self) -> None:
        normalized_model_name = self.model_name.strip()
        normalized_provider_name = self.provider_name.strip()

        if not self.vectors:
            raise ValueError("Embedding result must contain at least one vector.")

        if not normalized_model_name:
            raise ValueError("Embedding model name cannot be empty.")

        if not normalized_provider_name:
            raise ValueError("Embedding provider name cannot be empty.")

        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")

        for index, vector in enumerate(self.vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding vector at index {index} has dimension "
                    f"{len(vector)}, expected {self.dimensions}."
                )

        object.__setattr__(self, "model_name", normalized_model_name)
        object.__setattr__(self, "provider_name", normalized_provider_name)

    @property
    def count(self) -> int:
        """Return the number of generated embeddings."""

        return len(self.vectors)


class BaseEmbeddingProvider(ABC):
    """
    Abstract contract for all AutoMind embedding providers.

    Implementations may use local models, OpenAI, Azure OpenAI,
    Hugging Face endpoints, or enterprise-hosted models.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the stable provider identifier."""

        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured embedding model name."""

        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding-vector dimension."""

        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingResult:
        """
        Generate embeddings for document chunks.
        """

        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        """
        Generate one embedding for a semantic-search query.
        """

        raise NotImplementedError

    @staticmethod
    def validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
        """
        Validate and normalize document text inputs.
        """

        normalized_texts = tuple(
            str(text).strip()
            for text in texts
            if str(text).strip()
        )

        if not normalized_texts:
            raise ValueError(
                "At least one non-empty text input is required."
            )

        return normalized_texts

    @staticmethod
    def validate_query(text: str) -> str:
        """
        Validate and normalize a semantic-search query.
        """

        normalized_text = str(text).strip()

        if not normalized_text:
            raise ValueError("Embedding query cannot be empty.")

        return normalized_text