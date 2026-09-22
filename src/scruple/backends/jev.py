"""The Jev backend (§9.1) -- the default, and the model the thesis rests on.

Jev is trained with RLCD to produce calibrated decisions, which is the property
the whole project depends on: without probabilities that mean what they say,
abstention carries no information and §8.6 has nothing to work with.

All codes go in a single call. Questions are evaluated in parallel and in
isolation, so fanning out over codes is nearly free -- but the request still has
to fit the two ceilings in `TokenBudget`, and `BaseBackend.score` splits it when
it does not.

The wire format below was established against the live endpoint, not inferred:

    POST {endpoint}
    {
      "model": "typesafe-ai/jev",
      "state": "<the item text>",
      "questions": {
        "<code_id>": {"type": "noul", "instructions": "<the code definition>"}
      }
    }

    200 ->
    {
      "model": "typesafe-ai/jev",
      "answers": {"<code_id>": {"type": "noul", "noul": 0.98}},
      "usage": {"input_tokens": 323, "output_tokens": 43},
      "provider_metadata": {"gateway": {"marketCost": "0.00001281", ...}}
    }

Two details worth knowing. `questions` is a record keyed by question id, not a
list, so code ids come back attached to their answers rather than by position.
And a Noul question needs `instructions` (or `criteria`): a bare `question`
string is rejected by the model. The researcher's definition goes into
`instructions` verbatim -- scruple never paraphrases it, because the definition
is the intellectual contribution of the study (§4).

Parsing is strict. An unexpected shape raises rather than being coerced into a
plausible number, because every guarantee downstream assumes the probability
means something (§14.7).
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

DEFAULT_ENDPOINT = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
DEFAULT_MODEL = "typesafe-ai/jev"

API_KEY_ENV = "JEV_API_KEY"
ENDPOINT_ENV = "JEV_BASE_URL"
MODEL_ENV = "JEV_MODEL"
LEGACY_API_KEY_ENV = "TYPESAFE_API_KEY"

NOUL = "noul"


def render_instructions(code: Code) -> str:
    """The `instructions` payload for one code.

    The definition is the researcher's, verbatim. Examples are appended because
    the API ignores fields it does not know, so an `examples` key would silently
    throw them away -- and a researcher who wrote examples expects them to count.
    """
    lines = [code.normalised_definition]
    if code.examples_yes:
        lines.append("Applies, for example: " + " | ".join(code.examples_yes))
    if code.examples_no:
        lines.append("Does not apply, for example: " + " | ".join(code.examples_no))
    return "\n".join(lines)


class JevBackend(BaseBackend):
    """Calibrated probabilities from TypeSafe's Jev."""

    name = "jev"

    def __init__(
        self,
        *,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        budget: TokenBudget | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
        sleep: SleepFn | None = None,
    ) -> None:
        super().__init__(budget=budget, sleep=sleep)
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        # §9.1: an endpoint override lets a gateway stand in, so users without
        # direct waitlist access are not blocked. The default is such a gateway.
        self.endpoint = endpoint or os.environ.get(ENDPOINT_ENV) or DEFAULT_ENDPOINT
        self.api_key = api_key or os.environ.get(API_KEY_ENV) or os.environ.get(LEGACY_API_KEY_ENV)
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
        observed value wins, because the manifest must record what ran rather
        than what was asked for.
        """
        return self._observed_version or self.model

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise BackendError(
                "no API key for the jev backend",
                hint=(
                    f"Set {API_KEY_ENV} (a .env file in the project directory is read "
                    "automatically), or use `backend.name: local` to keep data on this machine."
                ),
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _record_usage(self, body: dict[str, Any]) -> None:
        usage = body.get("usage")
        if isinstance(usage, dict):
            self.usage.input_tokens += int(usage.get("input_tokens") or 0)
            self.usage.output_tokens += int(usage.get("output_tokens") or 0)

        # The gateway reports what the call actually cost. Real numbers beat the
        # estimate in §10.1, and the manifest carries them into the report.
        metadata = body.get("provider_metadata")
        gateway = metadata.get("gateway") if isinstance(metadata, dict) else None
        if isinstance(gateway, dict):
            for field in ("cost", "marketCost"):
                raw = gateway.get(field)
                try:
                    value = float(raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if value > 0.0:
                    self.usage.cost_usd += value
                    break

    def _score_batch(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                code.id: {"type": NOUL, "instructions": render_instructions(code)} for code in codes
            },
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

        returned_model = body.get("model")
        if isinstance(returned_model, str) and returned_model:
            self._observed_version = returned_model

        self._record_usage(body)

        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise BackendError(
                f"{self.name} response has no `answers` object",
                hint="The wire format may have changed; re-check against the live API.",
            )

        out: dict[str, float | None] = {}
        for code in codes:
            answer = answers.get(code.id)
            if answer is None:
                # Leave it absent: BaseBackend.score reports the omission by id,
                # which is more useful than a None that looks like a refusal.
                continue
            if not isinstance(answer, dict):
                raise BackendError(
                    f"{self.name} returned a non-object answer for {code.id!r}: {answer!r}"
                )
            if NOUL not in answer:
                raise BackendError(
                    f"{self.name} returned an answer for {code.id!r} with no {NOUL!r} field; "
                    f"got keys {sorted(answer)}"
                )
            # Range and type validation happen in BaseBackend.score, which
            # refuses an out-of-range number rather than clamping it.
            out[code.id] = answer[NOUL]
        return out
