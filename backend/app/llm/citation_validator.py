from __future__ import annotations

import re
from dataclasses import dataclass


class CitationValidationError(Exception):
    """
    Raised when an LLM answer contains invalid citation references.
    """


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """
    Summary of citation validation for one generated answer.
    """

    answer: str

    available_citation_ids: tuple[str, ...]

    used_citation_ids: tuple[str, ...]

    invalid_citation_ids: tuple[str, ...]

    missing_citations: bool

    is_valid: bool

    def __post_init__(self) -> None:
        normalized_answer = str(self.answer).strip()

        if not normalized_answer:
            raise ValueError(
                "Citation validation answer cannot be empty."
            )

        normalized_available_ids = self._normalize_ids(
            self.available_citation_ids
        )
        normalized_used_ids = self._normalize_ids(
            self.used_citation_ids
        )
        normalized_invalid_ids = self._normalize_ids(
            self.invalid_citation_ids
        )

        if not set(normalized_used_ids).issubset(
            set(normalized_available_ids)
        ):
            raise ValueError(
                "Used citation identifiers must be available citations."
            )

        if set(normalized_invalid_ids).intersection(
            normalized_available_ids
        ):
            raise ValueError(
                "Invalid citations cannot also be available citations."
            )

        expected_validity = (
            not normalized_invalid_ids
            and not self.missing_citations
        )

        if self.is_valid != expected_validity:
            raise ValueError(
                "Citation validation status is inconsistent."
            )

        object.__setattr__(
            self,
            "answer",
            normalized_answer,
        )
        object.__setattr__(
            self,
            "available_citation_ids",
            normalized_available_ids,
        )
        object.__setattr__(
            self,
            "used_citation_ids",
            normalized_used_ids,
        )
        object.__setattr__(
            self,
            "invalid_citation_ids",
            normalized_invalid_ids,
        )

    @staticmethod
    def _normalize_ids(
        citation_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(citation_id).strip().upper()
                for citation_id in citation_ids
                if str(citation_id).strip()
            )
        )


class CitationValidator:
    """
    Validates citation labels used in grounded LLM answers.

    Supported citation format:

        [SOURCE 1]
        [SOURCE 2]
        [SOURCE 1, SOURCE 3]

    The validator rejects fabricated or out-of-range source labels and can
    require at least one valid citation whenever project evidence was supplied.
    """

    CITATION_BLOCK_PATTERN = re.compile(
        r"\[(?P<content>[^\[\]]*SOURCE\s+\d+[^\[\]]*)\]",
        flags=re.IGNORECASE,
    )

    SOURCE_ID_PATTERN = re.compile(
        r"\bSOURCE\s+(?P<index>\d+)\b",
        flags=re.IGNORECASE,
    )

    def validate(
        self,
        *,
        answer: str,
        available_citation_ids: tuple[str, ...],
        require_citations: bool,
    ) -> CitationValidationResult:
        """
        Validate all source labels referenced by a generated answer.

        Args:
            answer:
                LLM-generated engineering response.

            available_citation_ids:
                Authoritative source identifiers provided in the prompt.

            require_citations:
                Whether at least one valid citation must appear. This should
                normally be True when retrieved project evidence was supplied.
        """

        normalized_answer = str(answer).strip()

        if not normalized_answer:
            raise CitationValidationError(
                "Generated answer cannot be empty."
            )

        normalized_available_ids = self._normalize_available_ids(
            available_citation_ids
        )

        extracted_ids = self._extract_citation_ids(
            normalized_answer
        )

        available_set = set(normalized_available_ids)

        used_ids = tuple(
            citation_id
            for citation_id in extracted_ids
            if citation_id in available_set
        )

        invalid_ids = tuple(
            citation_id
            for citation_id in extracted_ids
            if citation_id not in available_set
        )

        missing_citations = (
            require_citations
            and not used_ids
        )

        return CitationValidationResult(
            answer=normalized_answer,
            available_citation_ids=normalized_available_ids,
            used_citation_ids=tuple(
                dict.fromkeys(used_ids)
            ),
            invalid_citation_ids=tuple(
                dict.fromkeys(invalid_ids)
            ),
            missing_citations=missing_citations,
            is_valid=(
                not invalid_ids
                and not missing_citations
            ),
        )

    def ensure_valid(
        self,
        *,
        answer: str,
        available_citation_ids: tuple[str, ...],
        require_citations: bool,
    ) -> CitationValidationResult:
        """
        Validate an answer and raise when citation requirements are violated.
        """

        result = self.validate(
            answer=answer,
            available_citation_ids=available_citation_ids,
            require_citations=require_citations,
        )

        if result.invalid_citation_ids:
            invalid_labels = ", ".join(
                f"[{citation_id}]"
                for citation_id in result.invalid_citation_ids
            )

            raise CitationValidationError(
                "The generated answer referenced unavailable citations: "
                f"{invalid_labels}."
            )

        if result.missing_citations:
            raise CitationValidationError(
                "The generated answer did not include any valid citations "
                "despite receiving retrieved project evidence."
            )

        return result

    def _extract_citation_ids(
        self,
        answer: str,
    ) -> tuple[str, ...]:
        """
        Extract normalized SOURCE identifiers from bracketed citation blocks.

        Plain text such as 'SOURCE 2' outside square brackets is ignored.
        This prevents accidental prose references from being treated as
        authoritative citations.
        """

        extracted_ids: list[str] = []

        for citation_block in self.CITATION_BLOCK_PATTERN.finditer(
            answer
        ):
            block_content = citation_block.group("content")

            for source_match in self.SOURCE_ID_PATTERN.finditer(
                block_content
            ):
                source_index = int(
                    source_match.group("index")
                )

                if source_index < 1:
                    extracted_ids.append(
                        f"SOURCE {source_index}"
                    )
                    continue

                extracted_ids.append(
                    f"SOURCE {source_index}"
                )

        return tuple(
            dict.fromkeys(extracted_ids)
        )

    @staticmethod
    def _normalize_available_ids(
        citation_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized_ids = tuple(
            dict.fromkeys(
                str(citation_id).strip().upper()
                for citation_id in citation_ids
                if str(citation_id).strip()
            )
        )

        for citation_id in normalized_ids:
            if not re.fullmatch(
                r"SOURCE\s+[1-9]\d*",
                citation_id,
            ):
                raise CitationValidationError(
                    f"Invalid available citation identifier "
                    f"'{citation_id}'."
                )

        return normalized_ids