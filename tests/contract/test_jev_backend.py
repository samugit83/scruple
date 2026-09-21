"""Contract tests for the Jev backend (plan §9.1, §14.7)."""

from __future__ import annotations

import pytest

from scruple.backends import TokenBudget
from scruple.backends.jev import API_KEY_ENV, DEFAULT_ENDPOINT, JevBackend, render_question
from scruple.codebook import Code
from scruple.errors import BackendError

from .conftest import error_response, json_response


def answer(code_id: str, probability: float | None) -> dict[str, object]:
    return {"id": code_id, "probability": probability}


def body(*answers: dict[str, object], model: str = "jev-1.4.2") -> dict[str, object]:
    return {
        "model": model,
        "answers": list(answers),
        "usage": {"input_tokens": 120, "output_tokens": 8},
    }


class TestRequestShape:
    def test_sends_one_noul_question_per_code_in_a_single_call(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # §9.1: questions are evaluated in parallel and in isolation, so fanning
        # out over codes is nearly free and should not become many calls.
        t = transport(
            json_response(body(answer("access_barrier", 0.9), answer("distrust_pharma", 0.1)))
        )
        backend = JevBackend(api_key="k", client=t.client())
        backend.score("an answer", codes)

        assert len(t.calls) == 1
        sent = t.calls[0].body
        assert sent["state"] == "an answer"
        assert [q["id"] for q in sent["questions"]] == ["access_barrier", "distrust_pharma"]  # type: ignore[index]
        assert {q["type"] for q in sent["questions"]} == {"Noul"}  # type: ignore[index]

    def test_authorises_with_a_bearer_token(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", 0.5), answer("distrust_pharma", 0.5)))
        )
        JevBackend(api_key="secret", client=t.client()).score("x", codes)
        assert t.calls[0].headers["authorization"] == "Bearer secret"

    def test_posts_to_the_documented_endpoint_by_default(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", 0.5), answer("distrust_pharma", 0.5)))
        )
        JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert t.calls[0].url == DEFAULT_ENDPOINT

    def test_endpoint_override_lets_a_gateway_stand_in(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # §9.1: users without waitlist access must not be blocked.
        t = transport(
            json_response(body(answer("access_barrier", 0.5), answer("distrust_pharma", 0.5)))
        )
        backend = JevBackend(
            api_key="k", client=t.client(), endpoint="https://gateway.example/v1/x"
        )
        backend.score("x", codes)
        assert t.calls[0].url == "https://gateway.example/v1/x"

    def test_a_missing_api_key_names_the_variable_and_the_private_alternative(self) -> None:
        backend = JevBackend(api_key=None)
        backend.api_key = None
        with pytest.raises(BackendError, match="no API key") as caught:
            backend.score("x", [Code(id="abc_code", definition="d")])
        assert API_KEY_ENV in (caught.value.hint or "")
        assert "local" in (caught.value.hint or "")

    def test_the_question_uses_the_definition_verbatim(self) -> None:
        # scruple never paraphrases a definition: it is the study's contribution.
        code = Code(id="abc_code", definition="A very specific phrasing that matters.")
        assert "A very specific phrasing that matters." in render_question(code)

    def test_examples_are_labelled_by_polarity(self) -> None:
        code = Code(
            id="abc_code", definition="d", examples_yes=("yes case",), examples_no=("no case",)
        )
        rendered = render_question(code)
        assert rendered.index("yes case") < rendered.index("no case")
        assert "does not apply" in rendered


class TestResponseParsing:
    def test_returns_a_probability_per_code(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", 0.91), answer("distrust_pharma", 0.03)))
        )
        result = JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert result == {"access_barrier": 0.91, "distrust_pharma": 0.03}

    def test_records_the_exact_returned_model_version(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # §7.3: the manifest quotes what ran, not what was asked for.
        t = transport(
            json_response(
                body(
                    answer("access_barrier", 0.5), answer("distrust_pharma", 0.5), model="jev-2.0.1"
                )
            )
        )
        backend = JevBackend(model="jev-latest", api_key="k", client=t.client())
        assert backend.model_version() == "jev-latest"
        backend.score("x", codes)
        assert backend.model_version() == "jev-2.0.1"

    def test_accumulates_token_usage(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", 0.5), answer("distrust_pharma", 0.5)))
        )
        backend = JevBackend(api_key="k", client=t.client())
        backend.score("x", codes)
        assert backend.usage.calls == 1
        assert backend.usage.input_tokens == 120
        assert backend.usage.output_tokens == 8

    def test_a_null_probability_for_one_code_is_none_not_zero(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # §10.2: a missing probability is not a negative judgement.
        t = transport(
            json_response(body(answer("access_barrier", None), answer("distrust_pharma", 0.4)))
        )
        result = JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert result["access_barrier"] is None
        assert result["distrust_pharma"] == 0.4

    @pytest.mark.parametrize("alias", ["results", "questions"])
    def test_tolerates_documented_alternative_answer_keys(
        self, transport, codes, alias: str
    ) -> None:  # type: ignore[no-untyped-def]
        # The live schema is unconfirmed (see the note in backends/jev.py).
        payload = {
            "model": "jev-1",
            alias: [answer("access_barrier", 0.7), answer("distrust_pharma", 0.2)],
        }
        t = transport(json_response(payload))
        result = JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert result["access_barrier"] == 0.7

    @pytest.mark.parametrize("alias", ["p", "confidence", "value"])
    def test_tolerates_documented_alternative_probability_keys(
        self, transport, codes, alias: str
    ) -> None:  # type: ignore[no-untyped-def]
        payload = {
            "model": "jev-1",
            "answers": [
                {"id": "access_barrier", alias: 0.6},
                {"id": "distrust_pharma", alias: 0.1},
            ],
        }
        t = transport(json_response(payload))
        assert JevBackend(api_key="k", client=t.client()).score("x", codes)["access_barrier"] == 0.6


class TestMalformedResponses:
    """§14.7: rejected with a clear error rather than coerced.

    Clamping or defaulting a bad value would hide a real backend bug behind a
    plausible number, and every guarantee downstream assumes the probability
    means something.
    """

    def test_a_probability_above_one_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", 1.5), answer("distrust_pharma", 0.1)))
        )
        with pytest.raises(BackendError, match="outside \\[0, 1\\]"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_negative_probability_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", -0.2), answer("distrust_pharma", 0.1)))
        )
        with pytest.raises(BackendError, match="outside"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_non_numeric_probability_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(
            json_response(body(answer("access_barrier", "high"), answer("distrust_pharma", 0.1)))
        )  # type: ignore[arg-type]
        with pytest.raises(BackendError, match="not a number"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_boolean_probability_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # True would silently become 1.0, which is a confident decision.
        t = transport(
            json_response(body(answer("access_barrier", True), answer("distrust_pharma", 0.1)))
        )  # type: ignore[arg-type]
        with pytest.raises(BackendError, match="not a number"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_an_omitted_code_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response(body(answer("access_barrier", 0.5))))
        with pytest.raises(BackendError, match="omitted code 'distrust_pharma'"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_missing_answer_list_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response({"model": "jev-1"}))
        with pytest.raises(BackendError, match="no answer list"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_non_object_answer_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response({"model": "jev-1", "answers": ["nope"]}))
        with pytest.raises(BackendError, match="non-object answer"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_an_answer_without_an_id_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response({"model": "jev-1", "answers": [{"probability": 0.5}]}))
        with pytest.raises(BackendError, match="no id"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_non_json_body_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        import httpx

        t = transport(httpx.Response(200, text="<html>gateway</html>"))
        with pytest.raises(BackendError, match="non-JSON"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)

    def test_a_json_array_body_is_refused(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        t = transport(json_response([1, 2, 3]))
        with pytest.raises(BackendError, match="expected an object"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)


class TestBudgetSplitting:
    def test_a_codebook_over_budget_is_split_covering_every_code_exactly_once(
        self, transport
    ) -> None:  # type: ignore[no-untyped-def]
        """§14.7: multiple calls, no duplicates, no omissions."""
        many = [Code(id=f"code_{i:02d}", definition="word " * 200) for i in range(20)]

        def responder(request):  # type: ignore[no-untyped-def]
            import json as _json

            sent = _json.loads(request.content)
            ids = [q["id"] for q in sent["questions"]]
            return json_response({"model": "jev-1", "answers": [answer(i, 0.5) for i in ids]})

        import httpx

        calls: list[list[str]] = []

        def recording(request: httpx.Request) -> httpx.Response:
            import json as _json

            calls.append([q["id"] for q in _json.loads(request.content)["questions"]])
            return responder(request)

        client = httpx.Client(transport=httpx.MockTransport(recording))
        backend = JevBackend(
            api_key="k",
            client=client,
            budget=TokenBudget(max_state_and_question=800, max_state_and_all=1600),
        )
        result = backend.score("a short state", many)

        assert len(calls) > 1, "expected the budget to force a split"
        flat = [code_id for call in calls for code_id in call]
        assert sorted(flat) == sorted(c.id for c in many)
        assert len(flat) == len(set(flat)), "a code was asked about twice"
        assert set(result) == {c.id for c in many}


class TestRetries:
    def test_rate_limits_are_retried_and_no_code_is_dropped(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        waits: list[float] = []
        t = transport(
            error_response(429),
            error_response(503),
            json_response(body(answer("access_barrier", 0.8), answer("distrust_pharma", 0.2))),
            repeat_last=False,
        )
        backend = JevBackend(api_key="k", client=t.client(), sleep=waits.append)
        result = backend.score("x", codes)

        assert result == {"access_barrier": 0.8, "distrust_pharma": 0.2}
        assert len(t.calls) == 3
        assert len(waits) == 2, "each failure should have been followed by a backoff"
        assert backend.usage.retries == 2

    def test_backoff_grows_between_attempts(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        import httpx

        from scruple.backends.http import post_json

        waits: list[float] = []
        t = transport(error_response(429))
        with pytest.raises(BackendError, match="failed after 4 attempts"):
            post_json(
                t.client(),
                "https://example/x",
                {},
                max_retries=3,
                sleep=waits.append,
                backend="jev",
            )
        assert len(waits) == 3
        assert waits[-1] > waits[0]
        del httpx

    def test_retry_after_is_honoured(self, transport) -> None:  # type: ignore[no-untyped-def]
        from scruple.backends.http import post_json

        waits: list[float] = []
        t = transport(error_response(429, retry_after="7"))
        with pytest.raises(BackendError):
            post_json(
                t.client(),
                "https://example/x",
                {},
                max_retries=1,
                sleep=waits.append,
                backend="jev",
            )
        # Jittered within [0.5, 1.5] of the requested delay, never ignored.
        assert 3.4 <= waits[0] <= 10.6

    def test_a_client_error_is_not_retried(self, transport, codes) -> None:  # type: ignore[no-untyped-def]
        # A 401 will not fix itself; retrying wastes the user's money.
        t = transport(error_response(401))
        with pytest.raises(BackendError, match="HTTP 401"):
            JevBackend(api_key="k", client=t.client()).score("x", codes)
        assert len(t.calls) == 1

    def test_a_transport_error_is_retried_then_reported(self, transport) -> None:  # type: ignore[no-untyped-def]
        import httpx

        from scruple.backends.http import post_json

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        client = httpx.Client(transport=httpx.MockTransport(boom))
        waits: list[float] = []
        with pytest.raises(BackendError, match="ConnectError"):
            post_json(client, "https://example/x", {}, max_retries=2, sleep=waits.append)
        assert len(waits) == 2
