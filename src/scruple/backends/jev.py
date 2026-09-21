"""The Jev backend (plan §9.1) -- the default, and the model the thesis rests on.

Jev is trained with RLCD to produce calibrated decisions, which is the property
the whole project depends on: without probabilities that mean what they say,
abstention carries no information and §8.6 has nothing to work with.

All codes go in a single call. Questions are evaluated in parallel and in
isolation, so fanning out over codes is nearly free -- but the request still has
to fit the two ceilings in `TokenBudget`, and `BaseBackend.score` splits it when
it does not.

WIRE FORMAT IS PROVISIONAL. This adapter was written without access to a live
endpoint, so the response shape below is an assumption, not a confirmed contract.
It accepts a few plausible spellings of the same fields and refuses anything else
rather than coercing it. Before first release, verify against the real API and
narrow this parser -- `tests/contract/` holds the recorded fixtures to update.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from ..codebook import Code
from ..errors import BackendError
from .base import BaseBackend, TokenBudget
from .http import SleepFn, post_json

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
API_KEY_ENV = "TYPESAFE_API_KEY"

# Accepted spellings for the same field, because the live schema is unconfirmed.
_ANSWER_KEYS = ("answers", "results", "questions")
_PROBABILITY_KEYS = ("probability", "p", "confidence", "value")
_ID_KEYS = ("id", "question_id", "name")
_VERSION_KEYS = ("model", "model_version", "version")


def render_question(code: Code) -> str:
    """Turn a code definition into one Noul question.

    The definition is the researcher's, verbatim. scruple adds the framing and
    nothing else -- it never paraphrases a definition, because the definition is
    the intellectual contribution of the study (§4).
    """
    lines = [f"Does this apply to the text: {code.normalised_definition}"]
    if code.examples_yes:
        lines.append("Examples where it applies: " + " | ".join(code.examples_yes))
    if code.examples_no:
        lines.append("Examples where it does not apply: " + " | ".join(code.examples_no))
    return "\n".join(lines)


class JevBackend(BaseBackend):
    """Calibrated probabilities from TypeSafe's Jev."""

    name = "jev"

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        endpoint: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        budget: TokenBudget | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
        sleep: SleepFn | None = None,
    ) -> None:
        super().__init__(budget=budget, sleep=sleep)
        self.model = model
        # §9.1: an endpoint override lets a gateway provider stand in, so users
        # without waitlist access are not blocked.
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        self.max_retries = max_retries
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._observed_version: str | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def model_version(self) -> str:
        """The version string the API actually returned, recorded verbatim (§7.3).

        Before the first call the configured name is all we have; afterwards the
        observed value wins, because the manifest must record what ran and not
        what was asked for.
        """
        return self._observed_version or self.model

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise BackendError(
                "no API key for the jev backend",
                hint=(
                    f"Set {API_KEY_ENV}, or use `backend.name: local` to keep data on this machine."
                ),
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _first(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    def _score_batch(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        payload = {
            "model": self.model,
            "state": state,
            "questions": [
                {"id": code.id, "type": "Noul", "question": render_question(code)} for code in codes
            ],
        }
        body = post_json(
            self.client,
            self.endpoint,
            payload,
            headers=self._headers(),
            max_retries=self.max_retries,
            sleep=self.sleep,
            on_retry=lambda: setattr(self.usage, "retries", self.usage.retries + 1),
            backend=self.name,
        )
        self.usage.calls += 1

        version = self._first(body, _VERSION_KEYS)
        if isinstance(version, str) and version:
            self._observed_version = version

        usage = body.get("usage")
        if isinstance(usage, dict):
            self.usage.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.usage.output_tokens += int(usage.get("output_tokens", 0) or 0)

        answers = self._first(body, _ANSWER_KEYS)
        if not isinstance(answers, list):
            raise BackendError(
                f"{self.name} response has no answer list (looked for {', '.join(_ANSWER_KEYS)})",
                hint="The wire format may have changed; see the note in backends/jev.py.",
            )

        out: dict[str, float | None] = {}
        for answer in answers:
            if not isinstance(answer, dict):
                raise BackendError(f"{self.name} returned a non-object answer: {answer!r}")
            code_id = self._first(answer, _ID_KEYS)
            if not isinstance(code_id, str):
                raise BackendError(f"{self.name} returned an answer with no id: {answer!r}")
            # Validation of the value itself happens in BaseBackend.score, which
            # refuses out-of-range numbers rather than clamping them.
            out[code_id] = self._first(answer, _PROBABILITY_KEYS)
        return out
