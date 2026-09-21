"""Contract-test fixtures: recorded responses, never a live call (plan §14.7).

§14.2 forbids network access in every tier, and CI runs with networking
disabled. These fixtures serve recorded bodies through an httpx transport, so a
backend exercises its real request-building and response-parsing code without a
socket ever opening.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import httpx
import pytest

from scruple.codebook import Code


@dataclass
class RecordedCall:
    """One request the transport saw, for asserting on what was sent."""

    url: str
    headers: dict[str, str]
    body: dict[str, object]


@dataclass
class FakeTransport:
    """Serves queued responses and records the requests that arrived."""

    responses: list[httpx.Response]
    calls: list[RecordedCall] = field(default_factory=list)
    repeat_last: bool = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            RecordedCall(
                url=str(request.url),
                headers={k.lower(): v for k, v in request.headers.items()},
                body=json.loads(request.content or b"{}"),
            )
        )
        if len(self.responses) > 1 or not self.repeat_last:
            return self.responses.pop(0)
        return self.responses[0]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def transport() -> Callable[..., FakeTransport]:
    def build(*responses: httpx.Response, repeat_last: bool = True) -> FakeTransport:
        return FakeTransport(responses=list(responses), repeat_last=repeat_last)

    return build


def json_response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def error_response(status: int, *, retry_after: str | None = None) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after else {}
    return httpx.Response(status, json={"error": "nope"}, headers=headers)


@pytest.fixture
def codes() -> Sequence[Code]:
    return (
        Code(id="access_barrier", definition="A practical obstacle to attending."),
        Code(id="distrust_pharma", definition="Distrust of pharmaceutical companies."),
    )


@pytest.fixture
def no_sleep() -> list[float]:
    """Collects backoff durations instead of waiting them out."""
    return []
