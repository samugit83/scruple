"""Contract tests for the Jev backend (plan §9.1, §14.7).

The request and response shapes asserted here were captured from the live
System One endpoint, not inferred -- see `tests/contract/fixtures/jev/` and the
`bench/probe_jev.py` script that records them. No test opens a socket: §14.2
forbids network access in every tier, and CI runs with networking disabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scruple.backends import TokenBudget
from scruple.backends.jev import (
    API_KEY_ENV,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    JevBackend,
    render_instructions,
)
from scruple.codebook import Code
from scruple.errors import BackendError

from .conftest import error_response, json_response

FIXTURES = Path(__file__).parent / "fixtures" / "jev"


def recorded(name: str) -> dict[str, object]:
    """A response body recorded from the live API."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def answer(**pairs: float) -> dict[str, object]:
    """Build a response in the shape the live API returns."""
    return {
        "model": DEFAULT_MODEL,
        "answers": {code: {"type": "noul", "noul": value} for code, value in pairs.items()},
        "usage": {"input_tokens": 323, "output_tokens": 43},
        "provider_metadata": {"gateway": {"marketCost": "0.00001281"}},
    }


class TestRequestShape:
    def test_questions_are_a_record_keyed_by_code_id(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        """The live API rejects a list: "questions: expected record, received array".

        Keying by code id is also why answers come back attached to their code
        rather than matched by position.
        """
        t = transport(json_response(answer(access_barrier=0.98, distrust_pharma=0.02)))
        JevBackend(api_key="k", client=t.client()).score("an answer", codes)

        sent = t.calls[0].body
        assert isinstance(sent["questions"], dict)
        assert set(sent["questions"]) == {"access_barrier", "distrust_pharma"}  # type: ignore[arg-type]

    def test_each_question_is_a_noul_with_instructions(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # A bare `question` string is refused by the model: "Noul question must
        # have criteria or instructions".
        t = transport(json_response(answer(access_barrier=0.9, distrust_pharma=0.1)))
        JevBackend(api_key="k", client=t.client()).score("an answer", codes)

        question = t.calls[0].body["questions"]["access_barrier"]  # type: ignore[index]
        assert question["type"] == "noul"
        assert "A practical obstacle to attending." in question["instructions"]

    def test_all_codes_go_in_a_single_call(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # §9.1: questions are evaluated in parallel and in isolation, so fanning
        # out over codes is nearly free and must not become many calls.
        t = transport(json_response(answer(access_barrier=0.9, distrust_pharma=0.1)))
        JevBackend(api_key="k", client=t.client()).score("an answer", codes)
        assert len(t.calls) == 1

    def test_the_state_is_the_item_text_verbatim(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.9, distrust_pharma=0.1)))
        JevBackend(api_key="k", client=t.client()).score("the exact answer text", codes)
        assert t.calls[0].body["state"] == "the exact answer text"

    def test_authorises_with_a_bearer_token(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.5, distrust_pharma=0.5)))
        JevBackend(api_key="secret", client=t.client()).score("x", codes)
        assert t.calls[0].headers["authorization"] == "Bearer secret"

    def test_posts_to_the_gateway_by_default(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.5, distrust_pharma=0.5)))
        JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert t.calls[0].url == DEFAULT_ENDPOINT

    def test_endpoint_override_lets_a_proxy_stand_in(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.5, distrust_pharma=0.5)))
        backend = JevBackend(api_key="k", client=t.client(), endpoint="https://proxy.example/v1/x")
        backend.score("x", codes)
        assert t.calls[0].url == "https://proxy.example/v1/x"

    def test_reads_the_endpoint_and_model_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JEV_BASE_URL", "https://from-env.example/v1")
        monkeypatch.setenv("JEV_MODEL", "typesafe-ai/jev-1.2.3")
        backend = JevBackend(api_key="k")
        assert backend.endpoint == "https://from-env.example/v1"
        assert backend.model == "typesafe-ai/jev-1.2.3"

    def test_a_missing_api_key_names_the_variable_and_the_private_alternative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        backend = JevBackend()
        with pytest.raises(BackendError, match="no API key") as caught:
            backend.score("x", [Code(id="abc_code", definition="d")])
        assert API_KEY_ENV in (caught.value.hint or "")
        assert "local" in (caught.value.hint or "")


class TestInstructions:
    def test_the_definition_is_used_verbatim(self) -> None:
        # scruple never paraphrases a definition: it is the study's contribution.
        code = Code(id="abc_code", definition="A very specific phrasing that matters.")
        assert render_instructions(code).startswith("A very specific phrasing that matters.")

    def test_examples_are_included_and_labelled_by_polarity(self) -> None:
        """The API silently ignores fields it does not know, so an `examples`
        key would throw them away -- and a researcher who wrote examples expects
        them to count."""
        code = Code(
            id="abc_code", definition="d", examples_yes=("yes case",), examples_no=("no case",)
        )
        rendered = render_instructions(code)
        assert rendered.index("yes case") < rendered.index("no case")
        assert "Does not apply" in rendered

    def test_a_code_without_examples_sends_only_the_definition(self) -> None:
        assert render_instructions(Code(id="abc_code", definition="Just this.")) == "Just this."


class TestResponseParsing:
    def test_reads_the_noul_probability_per_code(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.98, distrust_pharma=0.02)))
        result = JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert result == {"access_barrier": 0.98, "distrust_pharma": 0.02}

    def test_parses_a_recorded_live_response(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        """The real thing, byte for byte, as captured from the endpoint."""
        t = transport(json_response(recorded("two_codes")))
        result = JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert result["access_barrier"] == pytest.approx(0.98)
        assert result["distrust_pharma"] == pytest.approx(0.02)

    def test_records_the_exact_returned_model_version(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # §7.3: the manifest quotes what ran, not what was asked for.
        body = answer(access_barrier=0.5, distrust_pharma=0.5)
        body["model"] = "typesafe-ai/jev-2026-09-01"
        t = transport(json_response(body))
        backend = JevBackend(model="typesafe-ai/jev", api_key="k", client=t.client())
        assert backend.model_version() == "typesafe-ai/jev"
        backend.score("x", codes)
        assert backend.model_version() == "typesafe-ai/jev-2026-09-01"

    def test_accumulates_token_usage(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.5, distrust_pharma=0.5)))
        backend = JevBackend(api_key="k", client=t.client())
        backend.score("x", codes)
        assert backend.usage.calls == 1
        assert backend.usage.input_tokens == 323
        assert backend.usage.output_tokens == 43

    def test_records_the_cost_the_gateway_reports(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # Real numbers beat the §10.1 estimate, and the manifest carries them
        # through to the report.
        t = transport(json_response(answer(access_barrier=0.5, distrust_pharma=0.5)))
        backend = JevBackend(api_key="k", client=t.client())
        backend.score("x", codes)
        assert backend.usage.cost_usd == pytest.approx(0.00001281)

    def test_a_response_without_cost_metadata_is_fine(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        body = answer(access_barrier=0.5, distrust_pharma=0.5)
        del body["provider_metadata"]
        t = transport(json_response(body))
        backend = JevBackend(api_key="k", client=t.client())
        backend.score("x", codes)
        assert backend.usage.cost_usd == 0.0


class TestMalformedResponses:
    """§14.7: rejected with a clear error rather than coerced.

    Clamping or defaulting a bad value would hide a real backend bug behind a
    plausible number, and every guarantee downstream assumes the probability
    means something.
    """

    def test_a_probability_above_one_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=1.5, distrust_pharma=0.1)))
        with pytest.raises(BackendError, match=r"outside \[0, 1\]"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_negative_probability_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=-0.2, distrust_pharma=0.1)))
        with pytest.raises(BackendError, match="outside"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_non_numeric_probability_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        body = answer(access_barrier=0.5, distrust_pharma=0.5)
        body["answers"]["access_barrier"]["noul"] = "high"  # type: ignore[index]
        t = transport(json_response(body))
        with pytest.raises(BackendError, match="not a number"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_boolean_probability_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # True would silently become 1.0, which is a confident decision.
        body = answer(access_barrier=0.5, distrust_pharma=0.5)
        body["answers"]["access_barrier"]["noul"] = True  # type: ignore[index]
        t = transport(json_response(body))
        with pytest.raises(BackendError, match="not a number"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_an_answer_without_a_noul_field_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        body = answer(access_barrier=0.5, distrust_pharma=0.5)
        body["answers"]["access_barrier"] = {"type": "noul", "score": 3}  # type: ignore[index]
        t = transport(json_response(body))
        with pytest.raises(BackendError, match="no 'noul' field"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_non_object_answer_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        body = answer(access_barrier=0.5, distrust_pharma=0.5)
        body["answers"]["access_barrier"] = 0.5  # type: ignore[index]
        t = transport(json_response(body))
        with pytest.raises(BackendError, match="non-object answer"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_an_omitted_code_is_refused_by_id(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(answer(access_barrier=0.5)))
        with pytest.raises(BackendError, match="omitted code 'distrust_pharma'"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_missing_answers_object_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response({"model": DEFAULT_MODEL, "usage": {}}))
        with pytest.raises(BackendError, match="no `answers` object"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_recorded_validation_error_is_reported_not_retried(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        """A 400 from the live API, recorded. It will not fix itself, so
        retrying only spends the user's money and hides the real problem."""
        t = transport(json_response(recorded("error_bad_question_type"), status=400))
        with pytest.raises(BackendError, match="HTTP 400"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert len(t.calls) == 1

    def test_a_non_json_body_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        import httpx

        t = transport(httpx.Response(200, text="<html>gateway timeout</html>"))
        with pytest.raises(BackendError, match="non-JSON"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)


class TestBudgetSplitting:
    def test_a_codebook_over_budget_is_split_covering_every_code_exactly_once(
        self,
    ) -> None:
        """§14.7: multiple calls, no duplicates, no omissions."""
        import httpx

        many = [Code(id=f"code_{i:02d}", definition="word " * 200) for i in range(20)]
        seen: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent = json.loads(request.content)
            ids = list(sent["questions"])
            seen.append(ids)
            return json_response(answer(**dict.fromkeys(ids, 0.5)))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        backend = JevBackend(
            api_key="k",
            client=client,
            budget=TokenBudget(max_state_and_question=800, max_state_and_all=1600),
        )
        result = backend.score("a short state", many)

        assert len(seen) > 1, "expected the budget to force a split"
        flat = [code_id for call in seen for code_id in call]
        assert sorted(flat) == sorted(c.id for c in many)
        assert len(flat) == len(set(flat)), "a code was asked about twice"
        assert set(result) == {c.id for c in many}


class TestRetries:
    def test_rate_limits_are_retried_and_no_code_is_dropped(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        waits: list[float] = []
        t = transport(
            error_response(429),
            error_response(503),
            json_response(answer(access_barrier=0.8, distrust_pharma=0.2)),
            repeat_last=False,
        )
        backend = JevBackend(api_key="k", client=t.client(), sleep=waits.append)
        result = backend.score("x", codes)

        assert result == {"access_barrier": 0.8, "distrust_pharma": 0.2}
        assert len(t.calls) == 3
        assert len(waits) == 2, "each failure should have been followed by a backoff"
        assert backend.usage.retries == 2

    def test_a_client_error_is_not_retried(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(error_response(401))
        with pytest.raises(BackendError, match="HTTP 401"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert len(t.calls) == 1
