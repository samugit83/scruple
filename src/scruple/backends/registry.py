"""Backend lookup by name (§9)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import BackendConfig
from ..errors import BackendError
from .base import Backend
from .jev import JevBackend
from .llm import AnthropicBackend, LocalBackend, OpenAIBackend
from .recorded import RecordedBackend

BackendFactory = Callable[[BackendConfig], Backend]


def _jev(config: BackendConfig) -> Backend:
    # "jev-latest" is the placeholder `scruple init` writes; it is not a model
    # the API knows, so fall back to the configured default rather than sending
    # it and failing with a confusing 400.
    model = None if config.model in ("jev-latest", "", None) else config.model
    return JevBackend(model=model, endpoint=config.endpoint)


def _openai(config: BackendConfig) -> Backend:
    model = config.model if config.model != "jev-latest" else "gpt-4o-mini"
    return OpenAIBackend(model=model, endpoint=config.endpoint)


def _anthropic(config: BackendConfig) -> Backend:
    model = config.model if config.model != "jev-latest" else "claude-sonnet-5"
    return AnthropicBackend(model=model, endpoint=config.endpoint)


def _local(config: BackendConfig) -> Backend:
    model = config.model if config.model != "jev-latest" else "local-model"
    return LocalBackend(model=model, endpoint=config.endpoint)


def _recorded(config: BackendConfig) -> Backend:
    if not config.endpoint:
        raise BackendError(
            "the recorded backend needs `backend.endpoint` set to a probabilities file",
            hint="examples/vaccine_survey ships one, so the pipeline runs with no API key.",
        )
    return RecordedBackend.from_path(Path(config.endpoint))


REGISTRY: dict[str, BackendFactory] = {
    "jev": _jev,
    "openai": _openai,
    "anthropic": _anthropic,
    "local": _local,
    "recorded": _recorded,
}

PRIVACY: dict[str, str] = {
    "jev": "Item text is sent to the configured System One endpoint (TypeSafe, or a gateway).",
    "openai": "Item text is sent to api.openai.com.",
    "anthropic": "Item text is sent to api.anthropic.com.",
    "local": "Item text is sent only to the endpoint you configure.",
    "recorded": "Nothing leaves this machine; probabilities are replayed from a file.",
}


def available() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def build_backend(config: BackendConfig) -> Backend:
    """Construct the configured backend."""
    try:
        factory = REGISTRY[config.name]
    except KeyError:
        raise BackendError(
            f"unknown backend {config.name!r}",
            hint=f"Available: {', '.join(available())}.",
        ) from None
    return factory(config)
