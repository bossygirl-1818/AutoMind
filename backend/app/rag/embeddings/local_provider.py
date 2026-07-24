from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from sentence_transformers import SentenceTransformer

from app.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingGenerationError,
    EmbeddingResult,
)


class LocalSentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    Generate private local embeddings using Sentence Transformers.

    The model is loaded lazily on first use so that starting the FastAPI
    application does not immediately consume model-loading time or memory.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        normalized_model_name = model_name.strip()

        if not normalized_model_name:
            raise ValueError("Embedding model name cannot be empty.")

        if batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than zero.")

        self._model_name = normalized_model_name
        self._batch_size = batch_size
        self._device = device
        self._normalize_embeddings = normalize_embeddings

        self._model: SentenceTransformer | None = None
        self._dimensions: int | None = None
        self._model_lock = Lock()

    @property
    def provider_name(self) -> str:
        return "sentence-transformers-local"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load_model()

        if self._dimensions is None:
            raise EmbeddingGenerationError(
                f"Could not determine embedding dimensions for "
                f"'{self.model_name}'."
            )

        return self._dimensions

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> EmbeddingResult:
        normalized_texts = self.validate_texts(texts)
        model = self._load_model()

        try:
            encoded = model.encode(
                list(normalized_texts),
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate document embeddings using "
                f"'{self.model_name}'."
            ) from exc

        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in encoded
        )

        return EmbeddingResult(
            vectors=vectors,
            model_name=self.model_name,
            dimensions=self.dimensions,
            provider_name=self.provider_name,
        )

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        normalized_query = self.validate_query(text)
        model = self._load_model()

        try:
            encoded = model.encode(
                normalized_query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate query embedding using "
                f"'{self.model_name}'."
            ) from exc

        vector = tuple(float(value) for value in encoded)

        if len(vector) != self.dimensions:
            raise EmbeddingGenerationError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.dimensions}, received {len(vector)}."
            )

        return vector

    def _load_model(self) -> SentenceTransformer:
        """
        Load the embedding model exactly once.

        The lock protects against duplicate model loading when multiple
        requests reach the service simultaneously.
        """

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                model = SentenceTransformer(
                    self.model_name,
                    device=self._device,
                )

                dimensions = model.get_sentence_embedding_dimension()

                if dimensions is None or dimensions <= 0:
                    raise EmbeddingGenerationError(
                        f"Embedding model '{self.model_name}' returned an "
                        "invalid embedding dimension."
                    )

            except EmbeddingGenerationError:
                raise

            except Exception as exc:
                raise EmbeddingGenerationError(
                    f"Failed to load local embedding model "
                    f"'{self.model_name}'."
                ) from exc

            self._model = model
            self._dimensions = int(dimensions)

        return self._model


local_embedding_provider = LocalSentenceTransformerEmbeddingProvider()