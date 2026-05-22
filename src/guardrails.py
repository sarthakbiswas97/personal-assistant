"""Input and output safety guardrails for the assistant."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FilterResult(Enum):
    """Outcome of a guardrail check."""

    PASS = "pass"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class GuardrailResponse:
    """Immutable result of a guardrail evaluation."""

    result: FilterResult
    reason: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.result == FilterResult.BLOCKED


# -- Pattern definitions --

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|your)(\s+\w+)*\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a|an)\s+", re.IGNORECASE),
    re.compile(r"(system\s*prompt|system\s*message)\s*:", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
)

_HARMFUL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)", re.IGNORECASE),
    re.compile(r"how\s+to\s+(hack|break\s+into)\s+", re.IGNORECASE),
    re.compile(r"how\s+to\s+(harm|hurt|kill|injure)\s+(someone|a\s+person|myself|yourself)", re.IGNORECASE),
    re.compile(r"(synthesize|manufacture)\s+(drugs|narcotics|methamphetamine|fentanyl)", re.IGNORECASE),
    re.compile(r"how\s+to\s+steal\s+(identity|credit\s+card|personal\s+data)", re.IGNORECASE),
)

_UNSAFE_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(here\s+is|here\s+are)\s+(how\s+to|instructions\s+for)\s+(hack|make\s+a\s+bomb|harm)", re.IGNORECASE),
    re.compile(r"step\s*\d+\s*:\s*(obtain|acquire|steal)\s+(weapons|explosives|drugs)", re.IGNORECASE),
)


def check_input(text: str) -> GuardrailResponse:
    """Check user input for prompt injection and harmful content.

    Args:
        text: The raw user message.

    Returns:
        GuardrailResponse indicating pass or blocked with reason.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Input blocked: prompt injection detected")
            return GuardrailResponse(
                result=FilterResult.BLOCKED,
                reason="Your message was flagged as a potential prompt injection attempt.",
            )

    for pattern in _HARMFUL_PATTERNS:
        if pattern.search(text):
            logger.warning("Input blocked: harmful content detected")
            return GuardrailResponse(
                result=FilterResult.BLOCKED,
                reason="Your message was flagged as containing a potentially harmful request.",
            )

    return GuardrailResponse(result=FilterResult.PASS)


def check_output(text: str) -> GuardrailResponse:
    """Check model output for unsafe content.

    Args:
        text: The model's generated response.

    Returns:
        GuardrailResponse indicating pass or blocked with reason.
    """
    for pattern in _UNSAFE_OUTPUT_PATTERNS:
        if pattern.search(text):
            logger.warning("Output blocked: unsafe content detected")
            return GuardrailResponse(
                result=FilterResult.BLOCKED,
                reason="The response was filtered due to safety concerns.",
            )

    return GuardrailResponse(result=FilterResult.PASS)
