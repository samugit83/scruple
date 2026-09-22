"""Shared HTTP behaviour for network backends (§10.2).

Rate limits and transient server errors are retried with exponential backoff and
jitter; `Retry-After` is honoured when the server sends it. Nothing is ever
silently dropped -- once retries are exhausted the error propagates, and the
engine records that item/code as ``None`` rather than as a negative judgement.

The sleep function is injectable so tests exercise the backoff path without
actually waiting (§14.2 forbids slow, network-shaped tests).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from ..errors import BackendError

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
DEFAULT_BACKOFF = 0.5
MAX_BACKOFF = 30.0

SleepFn = Callable[[float], None]


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # HTTP allows an absolute date here; backoff is a fine fallback.
        return None


def post_json(
    client: httpx.Client,
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    max_retries: int = 5,
    sleep: SleepFn | None = None,
    rng: random.Random | None = None,
    on_retry: Callable[[], None] | None = None,
    backend: str = "backend",
) -> dict[str, Any]:
    """POST JSON, retrying rate limits and transient failures."""
    import time

    sleeper: SleepFn = sleep if sleep is not None else time.sleep
    jitter = rng or random.Random(0)
    last: str = "no attempt was made"

    for attempt in range(max_retries + 1):
        retry_wait: float | None = None
        try:
            response = client.post(url, json=dict(payload), headers=dict(headers or {}))
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code < 400:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise BackendError(f"{backend} returned a non-JSON response: {exc}") from exc
                if not isinstance(body, dict):
                    raise BackendError(
                        f"{backend} returned {type(body).__name__}, expected an object"
                    )
                return body
            if response.status_code not in RETRY_STATUS:
                # A 400 or 401 will not fix itself; retrying wastes the user's
                # money and hides the real problem.
                raise BackendError(
                    f"{backend} returned HTTP {response.status_code}: {response.text[:300]}"
                )
            last = f"HTTP {response.status_code}"
            retry_wait = _retry_after(response)

        if attempt == max_retries:
            break
        if on_retry is not None:
            on_retry()
        backoff = (
            retry_wait
            if retry_wait is not None
            else min(MAX_BACKOFF, DEFAULT_BACKOFF * (2**attempt))
        )
        sleeper(backoff * (0.5 + jitter.random()))

    raise BackendError(
        f"{backend} failed after {max_retries + 1} attempts ({last})",
        hint="Check the endpoint and your network, then re-run with `--retry-failed`.",
    )
