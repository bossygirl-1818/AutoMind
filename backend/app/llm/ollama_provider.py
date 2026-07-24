from __future__ import annotations

from time import perf_counter
from typing import Any
from urllib.parse import urljoin

import httpx

from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMGenerationError,
    LLMGenerationRequest,
    LLMGenerationResult,
    LLMMessage,
    LLMTokenUsage,
)


class OllamaProvider(BaseLLMProvider):
    """
    Local Ollama implementation of AutoMind's LLM provider contract.

    The provider communicates with Ollama through its native HTTP API:

        POST /api/chat

    Ollama remains an infrastructure dependency. Prompt builders, RAG
    services, agents, and API routes depend only on BaseLLMProvider.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL_NAME = "qwen3:4b"
    DEFAULT_TIMEOUT_SECONDS = 180.0
    DEFAULT_KEEP_ALIVE = "10m"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        keep_alive: str | int | None = DEFAULT_KEEP_ALIVE,
    ) -> None:
        normalized_base_url = str(base_url).strip().rstrip("/")
        normalized_model_name = str(model_name).strip()

        if not normalized_base_url:
            raise LLMConfigurationError(
                "Ollama base URL cannot be empty."
            )

        if not normalized_model_name:
            raise LLMConfigurationError(
                "Ollama model name cannot be empty."
            )

        if timeout_seconds <= 0:
            raise LLMConfigurationError(
                "Ollama timeout must be greater than zero."
            )

        self._base_url = normalized_base_url
        self._model_name = normalized_model_name
        self._timeout_seconds = float(timeout_seconds)
        self._keep_alive = keep_alive

    @property
    def provider_name(self) -> str:
        return "ollama-local"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate(
        self,
        request: LLMGenerationRequest,
    ) -> LLMGenerationResult:
        """
        Generate one non-streaming response using Ollama's chat endpoint.

        Local model calls are synchronous here because BaseLLMProvider
        currently defines a synchronous contract. FastAPI routes will execute
        this operation inside a worker thread.
        """

        if not isinstance(request, LLMGenerationRequest):
            raise TypeError(
                "request must be an LLMGenerationRequest instance."
            )

        payload = self._build_payload(request)
        endpoint = urljoin(
            f"{self.base_url}/",
            "api/chat",
        )

        started_at = perf_counter()

        try:
            with httpx.Client(
                timeout=httpx.Timeout(
                    self._timeout_seconds,
                )
            ) as client:
                response = client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )

                response.raise_for_status()

        except httpx.ConnectError as exc:
            raise LLMGenerationError(
                "Could not connect to the local Ollama service. "
                "Ensure Ollama is installed and running."
            ) from exc

        except httpx.TimeoutException as exc:
            raise LLMGenerationError(
                f"Ollama did not respond within "
                f"{self._timeout_seconds:g} seconds."
            ) from exc

        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(
                exc.response
            )

            raise LLMGenerationError(
                f"Ollama returned HTTP "
                f"{exc.response.status_code}: {detail}"
            ) from exc

        except httpx.HTTPError as exc:
            raise LLMGenerationError(
                "The Ollama HTTP request failed."
            ) from exc

        latency_ms = round(
            (perf_counter() - started_at) * 1000,
            3,
        )

        try:
            response_data = response.json()

        except ValueError as exc:
            raise LLMGenerationError(
                "Ollama returned an invalid JSON response."
            ) from exc

        return self._build_result(
            response_data=response_data,
            latency_ms=latency_ms,
        )

    def is_available(self) -> bool:
        """
        Return whether the local Ollama API is reachable.

        This is an operational health check only. It does not prove that the
        configured model has already been downloaded.
        """

        endpoint = urljoin(
            f"{self.base_url}/",
            "api/tags",
        )

        try:
            with httpx.Client(
                timeout=httpx.Timeout(5.0)
            ) as client:
                response = client.get(endpoint)
                response.raise_for_status()

        except httpx.HTTPError:
            return False

        return True

    def is_model_available(self) -> bool:
        """
        Return whether the configured model appears in Ollama's local model
        registry.
        """

        endpoint = urljoin(
            f"{self.base_url}/",
            "api/tags",
        )

        try:
            with httpx.Client(
                timeout=httpx.Timeout(10.0)
            ) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                response_data = response.json()

        except (httpx.HTTPError, ValueError):
            return False

        models = response_data.get("models", [])

        if not isinstance(models, list):
            return False

        configured_model = self.model_name.casefold()

        for model in models:
            if not isinstance(model, dict):
                continue

            model_name = str(
                model.get("name")
                or model.get("model")
                or ""
            ).strip().casefold()

            if model_name == configured_model:
                return True

        return False

    def _build_payload(
        self,
        request: LLMGenerationRequest,
    ) -> dict[str, Any]:
        messages = [
            self._build_message_payload(message)
            for message in request.messages
        ]

        options: dict[str, Any] = {
    "temperature": request.temperature,
    "num_predict": request.max_output_tokens,
    "num_ctx": 4096,
}

        payload: dict[str, Any] = {
    "model": self.model_name,
    "messages": messages,
    "stream": False,
    "think": False,
    "options": options,
}

        if request.stop_sequences:
            options["stop"] = list(
                request.stop_sequences
            )

        if request.response_format == "json":
            payload["format"] = "json"

        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive

        return payload

    @staticmethod
    def _build_message_payload(
        message: LLMMessage,
    ) -> dict[str, str]:
        payload = {
            "role": message.role,
            "content": message.content,
        }

        return payload

    def _build_result(
        self,
        *,
        response_data: dict[str, Any],
        latency_ms: float,
    ) -> LLMGenerationResult:
        message = response_data.get("message")

        if not isinstance(message, dict):
            raise LLMGenerationError(
                "Ollama response is missing the generated message."
            )

        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            raise LLMGenerationError(
                "Ollama returned an empty response."
            )

        prompt_tokens = self._optional_non_negative_int(
            response_data.get("prompt_eval_count")
        )

        output_tokens = self._optional_non_negative_int(
            response_data.get("eval_count")
        )

        total_tokens = (
            prompt_tokens + output_tokens
            if (
                prompt_tokens is not None
                and output_tokens is not None
            )
            else None
        )

        model_name = str(
            response_data.get("model")
            or self.model_name
        ).strip()

        finish_reason = self._optional_string(
            response_data.get("done_reason")
        )

        return LLMGenerationResult(
            content=content,
            provider_name=self.provider_name,
            model_name=model_name,
            finish_reason=finish_reason,
            usage=LLMTokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            provider_request_id=None,
            response_metadata={
                "created_at": response_data.get(
                    "created_at"
                ),
                "done": bool(
                    response_data.get("done", False)
                ),
                "total_duration_ns": (
                    self._optional_non_negative_int(
                        response_data.get(
                            "total_duration"
                        )
                    )
                ),
                "load_duration_ns": (
                    self._optional_non_negative_int(
                        response_data.get(
                            "load_duration"
                        )
                    )
                ),
                "prompt_eval_duration_ns": (
                    self._optional_non_negative_int(
                        response_data.get(
                            "prompt_eval_duration"
                        )
                    )
                ),
                "eval_duration_ns": (
                    self._optional_non_negative_int(
                        response_data.get(
                            "eval_duration"
                        )
                    )
                ),
            },
        )

    @staticmethod
    def _extract_error_detail(
        response: httpx.Response,
    ) -> str:
        try:
            response_data = response.json()

        except ValueError:
            response_text = response.text.strip()

            return (
                response_text[:500]
                if response_text
                else "Unknown Ollama error."
            )

        error_detail = str(
            response_data.get("error", "")
        ).strip()

        return error_detail or "Unknown Ollama error."

    @staticmethod
    def _optional_non_negative_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            parsed_value = int(value)

        except (TypeError, ValueError):
            return None

        return parsed_value if parsed_value >= 0 else None

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()

        return normalized or None