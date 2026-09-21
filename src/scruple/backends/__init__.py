"""Probability providers (plan §9).

Swappable by design: the engine is a pure function of (corpus, codebook,
backend) -> probabilities, and every decision is made downstream in `stats/`.
"""

from .base import (
    Backend,
    BaseBackend,
    TokenBudget,
    Usage,
    question_tokens,
    split_codes,
    validate_probability,
)
from .jev import JevBackend, render_question
from .llm import AnthropicBackend, LocalBackend, OpenAIBackend, OpenAICompatibleBackend
from .recorded import RecordedBackend
from .registry import PRIVACY, REGISTRY, available, build_backend

__all__ = [
    "PRIVACY",
    "REGISTRY",
    "AnthropicBackend",
    "Backend",
    "BaseBackend",
    "JevBackend",
    "LocalBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "RecordedBackend",
    "TokenBudget",
    "Usage",
    "available",
    "build_backend",
    "question_tokens",
    "render_question",
    "split_codes",
    "validate_probability",
]
