"""Backend registry and the offline recorded backend (plan §9, §13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scruple.backends import (
    PRIVACY,
    Backend,
    RecordedBackend,
    available,
    build_backend,
)
from scruple.codebook import Code
from scruple.config import BackendConfig
from scruple.errors import BackendError
from scruple.hashing import item_hash

CODE = Code(id="abc_code", definition="A definition.")


class TestRegistry:
    def test_every_registered_backend_satisfies_the_protocol(self) -> None:
        """§14.7: every backend satisfies the Protocol of §9."""
        for name in available():
            config = BackendConfig(name=name, endpoint=None)
            if name == "recorded":
                continue  # needs a probabilities file; covered below
            backend = build_backend(config)
            assert isinstance(backend, Backend)
            assert isinstance(backend.model_version(), str)

    def test_the_documented_backends_are_present(self) -> None:
        assert set(available()) == {"jev", "openai", "anthropic", "local", "recorded"}

    def test_unknown_backend_lists_the_alternatives(self) -> None:
        with pytest.raises(BackendError, match="unknown backend") as caught:
            build_backend(BackendConfig(name="gpt5000"))
        assert "jev" in (caught.value.hint or "")

    def test_every_backend_declares_what_leaves_the_machine(self) -> None:
        # §9.3 and the README's privacy table: a researcher must be able to see,
        # per backend, exactly what is transmitted.
        assert set(PRIVACY) == set(available())
        assert "nothing leaves" in PRIVACY["recorded"].lower()
        assert "only to the endpoint you configure" in PRIVACY["local"]

    def test_a_jev_model_name_is_not_carried_into_an_llm_backend(self) -> None:
        # Switching `backend.name` without editing `backend.model` must not send
        # "jev-latest" to OpenAI and fail confusingly.
        backend = build_backend(BackendConfig(name="openai", model="jev-latest"))
        assert backend.model_version() != "jev-latest"

    def test_an_explicit_model_is_respected(self) -> None:
        backend = build_backend(BackendConfig(name="openai", model="gpt-4o"))
        assert backend.model_version() == "gpt-4o"

    def test_recorded_backend_needs_a_probabilities_file(self) -> None:
        with pytest.raises(BackendError, match=r"needs `backend\.endpoint`"):
            build_backend(BackendConfig(name="recorded", endpoint=None))


class TestRecordedBackend:
    def table(self) -> dict[str, dict[str, float | None]]:
        return {item_hash("an answer"): {"abc_code": 0.91}}

    def test_replays_a_recorded_probability(self) -> None:
        backend = RecordedBackend(self.table())
        assert backend.score("an answer", [CODE]) == {"abc_code": 0.91}

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(RecordedBackend(self.table()), Backend)

    def test_keys_on_content_so_identical_texts_share_an_entry(self) -> None:
        backend = RecordedBackend(self.table())
        assert backend.score("an answer", [CODE]) == backend.score("an answer", [CODE])

    def test_an_uncovered_item_is_refused_in_strict_mode(self) -> None:
        # Silently returning None would push every item into the abstention band
        # and look like a working run that needed a lot of human review.
        with pytest.raises(BackendError, match="no recorded probability"):
            RecordedBackend(self.table()).score("a different answer", [CODE])

    def test_an_uncovered_code_is_refused_in_strict_mode(self) -> None:
        with pytest.raises(BackendError, match="does not cover code"):
            RecordedBackend(self.table()).score("an answer", [Code(id="new_code", definition="d")])

    def test_lenient_mode_reports_failures_as_none(self) -> None:
        backend = RecordedBackend(self.table(), strict=False)
        assert backend.score("uncovered", [CODE]) == {"abc_code": None}

    def test_loads_from_a_file_with_a_model_version(self, tmp_path: Path) -> None:
        path = tmp_path / "probabilities.json"
        path.write_text(
            json.dumps({"model_version": "jev-1.4.2-recorded", "probabilities": self.table()}),
            encoding="utf-8",
        )
        backend = RecordedBackend.from_path(path)
        assert backend.model_version() == "jev-1.4.2-recorded"
        assert backend.score("an answer", [CODE])["abc_code"] == 0.91

    def test_loads_a_bare_table(self, tmp_path: Path) -> None:
        path = tmp_path / "probabilities.json"
        path.write_text(json.dumps(self.table()), encoding="utf-8")
        assert RecordedBackend.from_path(path).score("an answer", [CODE])["abc_code"] == 0.91

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="no recorded probabilities"):
            RecordedBackend.from_path(tmp_path / "absent.json")

    def test_malformed_file(self, tmp_path: Path) -> None:
        path = tmp_path / "probabilities.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BackendError, match="not valid JSON"):
            RecordedBackend.from_path(path)

    def test_non_object_file(self, tmp_path: Path) -> None:
        path = tmp_path / "probabilities.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(BackendError, match="must map item hashes"):
            RecordedBackend.from_path(path)

    def test_a_recorded_none_stays_none(self) -> None:
        backend = RecordedBackend({item_hash("x"): {"abc_code": None}})
        assert backend.score("x", [CODE]) == {"abc_code": None}

    def test_an_out_of_range_recorded_value_is_still_refused(self) -> None:
        # Validation lives in BaseBackend, so a bad recording cannot slip past.
        backend = RecordedBackend({item_hash("x"): {"abc_code": 1.4}})
        with pytest.raises(BackendError, match="outside"):
            backend.score("x", [CODE])
