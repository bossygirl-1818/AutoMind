from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class LLMProviderError(Exception):
    """
    Base exception raised when an LLM provider cannot complete a request.

    Provider implementations should wrap SDK-specific exceptions with this
    error so the rest of AutoMind remains independent of vendor libraries.
    """


class LLMConfigurationError(LLMProviderError):
    """
    Raised when an LLM provider is missing required configuration.
    """


class LLMGenerationError(LLMProviderError):
    """
    Raised when an LLM provider fails while generating a response.
    """


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """
    One normalized conversational message supplied to an LLM.

    AutoMind uses its own message contract instead of exposing provider SDK
    message objects to prompt builders, agents, or API services.
    """

    role: str

    content: str

    name: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
    )

    ALLOWED_ROLES = frozenset(
        {
            "system",
            "user",
            "assistant",
        }
    )

    def __post_init__(self) -> None:
        normalized_role = str(self.role).strip().lower()
        normalized_content = str(self.content).strip()

        if normalized_role not in self.ALLOWED_ROLES:
            raise ValueError(
                f"Unsupported LLM message role '{self.role}'."
            )

        if not normalized_content:
            raise ValueError(
                "LLM message content cannot be empty."
            )

        normalized_name = (
            str(self.name).strip()
            if self.name is not None
            else None
        )

        if normalized_name == "":
            normalized_name = None

        object.__setattr__(
            self,
            "role",
            normalized_role,
        )
        object.__setattr__(
            self,
            "content",
            normalized_content,
        )
        object.__setattr__(
            self,
            "name",
            normalized_name,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class LLMGenerationRequest:
    """
    Provider-neutral request for one LLM generation operation.

    The request contains only trusted generation controls. Raw provider
    parameters should not be accepted directly from API clients.
    """

    messages: tuple[LLMMessage, ...]

    temperature: float = 0.1

    max_output_tokens: int = 1200

    stop_sequences: tuple[str, ...] = ()

    response_format: str = "text"

    request_metadata: Mapping[str, Any] = field(
        default_factory=dict,
    )

    ALLOWED_RESPONSE_FORMATS = frozenset(
        {
            "text",
            "json",
        }
    )

    def __post_init__(self) -> None:
        normalized_messages = tuple(self.messages)

        if not normalized_messages:
            raise ValueError(
                "At least one LLM message is required."
            )

        if not all(
            isinstance(message, LLMMessage)
            for message in normalized_messages
        ):
            raise TypeError(
                "All messages must be LLMMessage instances."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "LLM temperature must be between 0 and 2."
            )

        if self.max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )

        if self.max_output_tokens > 32_768:
            raise ValueError(
                "max_output_tokens cannot exceed 32768."
            )

        normalized_stop_sequences = tuple(
            sequence.strip()
            for sequence in self.stop_sequences
            if sequence.strip()
        )

        if len(normalized_stop_sequences) > 8:
            raise ValueError(
                "A maximum of eight stop sequences is supported."
            )

        normalized_response_format = (
            str(self.response_format)
            .strip()
            .lower()
        )

        if (
            normalized_response_format
            not in self.ALLOWED_RESPONSE_FORMATS
        ):
            raise ValueError(
                "response_format must be either 'text' or 'json'."
            )

        object.__setattr__(
            self,
            "messages",
            normalized_messages,
        )
        object.__setattr__(
            self,
            "stop_sequences",
            normalized_stop_sequences,
        )
        object.__setattr__(
            self,
            "response_format",
            normalized_response_format,
        )
        object.__setattr__(
            self,
            "request_metadata",
            dict(self.request_metadata),
        )


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """
    Normalized token accounting returned by an LLM provider.

    Providers that do not report token usage may leave individual fields as
    None. Estimated values must not be presented as provider-reported values.
    """

    input_tokens: int | None = None

    output_tokens: int | None = None

    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens
            != self.input_tokens + self.output_tokens
        ):
            raise ValueError(
                "total_tokens must equal input_tokens plus "
                "output_tokens when all values are provided."
            )


@dataclass(frozen=True, slots=True)
class LLMGenerationResult:
    """
    Provider-neutral result returned after successful text generation.
    """

    content: str

    provider_name: str

    model_name: str

    finish_reason: str | None

    usage: LLMTokenUsage

    latency_ms: float

    provider_request_id: str | None = None

    response_metadata: Mapping[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        normalized_content = str(self.content).strip()
        normalized_provider_name = str(
            self.provider_name
        ).strip()
        normalized_model_name = str(
            self.model_name
        ).strip()

        if not normalized_content:
            raise ValueError(
                "LLM generation result content cannot be empty."
            )

        if not normalized_provider_name:
            raise ValueError(
                "LLM provider name cannot be empty."
            )

        if not normalized_model_name:
            raise ValueError(
                "LLM model name cannot be empty."
            )

        if self.latency_ms < 0:
            raise ValueError(
                "LLM latency cannot be negative."
            )

        normalized_finish_reason = (
            str(self.finish_reason).strip()
            if self.finish_reason is not None
            else None
        )

        normalized_provider_request_id = (
            str(self.provider_request_id).strip()
            if self.provider_request_id is not None
            else None
        )

        object.__setattr__(
            self,
            "content",
            normalized_content,
        )
        object.__setattr__(
            self,
            "provider_name",
            normalized_provider_name,
        )
        object.__setattr__(
            self,
            "model_name",
            normalized_model_name,
        )
        object.__setattr__(
            self,
            "finish_reason",
            normalized_finish_reason or None,
        )
        object.__setattr__(
            self,
            "provider_request_id",
            normalized_provider_request_id or None,
        )
        object.__setattr__(
            self,
            "response_metadata",
            dict(self.response_metadata),
        )


class BaseLLMProvider(ABC):
    """
    Abstract contract implemented by every AutoMind LLM provider.

    Future implementations may use:

        - a local model,
        - OpenAI,
        - Azure OpenAI,
        - an enterprise-hosted inference endpoint,
        - another approved provider.

    Prompt builders and engineering agents depend only on this abstraction.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider's stable internal identifier.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the configured model identifier.
        """

        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        request: LLMGenerationRequest,
    ) -> LLMGenerationResult:
        """
        Generate one model response.
        """

        raise NotImplementedError

    @staticmethod
    def normalize_messages(
        messages: Sequence[LLMMessage],
    ) -> tuple[LLMMessage, ...]:
        """
        Validate and normalize a message sequence.
        """

        normalized_messages = tuple(messages)

        if not normalized_messages:
            raise ValueError(
                "At least one LLM message is required."
            )

        if not all(
            isinstance(message, LLMMessage)
            for message in normalized_messages
        ):
            raise TypeError(
                "All messages must be LLMMessage instances."
            )

        return normalized_messages