from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


class VectorStoreError(Exception):
    """Base exception for vector store operations."""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Represents one embedding record stored in the vector database."""

    id: str
    embedding: tuple[float, ...]
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Vector record ID cannot be empty.")

        if not self.embedding:
            raise ValueError("Embedding cannot be empty.")

        if not self.document.strip():
            raise ValueError("Document text cannot be empty.")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represents one semantic search result."""

    id: str
    score: float
    document: str
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """
    Abstract interface implemented by every AutoMind vector database.
    """

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        records: Sequence[VectorRecord],
    ) -> None:
        """Insert or update vector records."""

        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        collection_name: str,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform semantic similarity search."""

        raise NotImplementedError

    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """Delete a collection."""

        raise NotImplementedError

    @abstractmethod
    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """Return True if a collection exists."""

        raise NotImplementedError