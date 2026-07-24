from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.rag.vectorstore.base import (
    BaseVectorStore,
    SearchResult,
    VectorRecord,
    VectorStoreError,
)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of the AutoMind vector store.

    A persistent client is used so vectors survive FastAPI restarts.
    """

    DEFAULT_DATABASE_PATH = "storage/chromadb"

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self._database_path = Path(database_path)

        self._database_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = chromadb.PersistentClient(
            path=str(self._database_path)
        )

    def upsert(
        self,
        collection_name: str,
        records: Sequence[VectorRecord],
    ) -> None:

        if not records:
            return

        collection = self._get_or_create_collection(collection_name)

        try:
            collection.upsert(
                ids=[record.id for record in records],
                embeddings=[
                    list(record.embedding)
                    for record in records
                ],
                documents=[
                    record.document
                    for record in records
                ],
                metadatas=[
                    record.metadata
                    for record in records
                ],
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert vectors into '{collection_name}'."
            ) from exc

    def search(
        self,
        collection_name: str,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:

        collection = self._get_collection(collection_name)

        try:
            results = collection.query(
                query_embeddings=[list(embedding)],
                n_results=top_k,
                where=where,
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Semantic search failed for '{collection_name}'."
            ) from exc

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: list[SearchResult] = []

        for vector_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            output.append(
                SearchResult(
                    id=vector_id,
                    score=float(distance),
                    document=document,
                    metadata=metadata or {},
                )
            )

        return output

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:

        try:
            self._client.delete_collection(
                collection_name
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete collection '{collection_name}'."
            ) from exc

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:

        try:
            collections = self._client.list_collections()

            return any(
                collection.name == collection_name
                for collection in collections
            )

        except Exception as exc:
            raise VectorStoreError(
                "Unable to list ChromaDB collections."
            ) from exc

    def _get_collection(
        self,
        collection_name: str,
    ) -> Collection:

        if not self.collection_exists(collection_name):
            raise VectorStoreError(
                f"Collection '{collection_name}' does not exist."
            )

        return self._client.get_collection(
            collection_name
        )

    def _get_or_create_collection(
        self,
        collection_name: str,
    ) -> Collection:

        return self._client.get_or_create_collection(
            name=collection_name
        )


chroma_vector_store = ChromaVectorStore()