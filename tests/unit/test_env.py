"""Unit tests for reading `.env` (§9.1).

A key that reaches a log, a manifest or a report is a leaked key, so these
check the loader keeps values out of its own return value as well as parsing
them correctly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scruple.env import ENV_FILENAME, is_secret, load_env, parse_env


class TestParseEnv:
    def test_parses_simple_assignments(self) -> None:
        assert parse_env("A=1\nB=two\n") == {"A": "1", "B": "two"}

    def test_ignores_comments_and_blank_lines(self) -> None:
        assert parse_env("# a comment\n\nA=1\n   \n# another\n") == {"A": "1"}

    def test_strips_surrounding_quotes(self) -> None:
        assert parse_env("A=\"quoted\"\nB='single'\n") == {"A": "quoted", "B": "single"}

    def test_keeps_internal_equals_signs(self) -> None:
        # Base64 keys and URLs with query strings both contain '='.
        assert parse_env("KEY=abc=def=\n") == {"KEY": "abc=def="}

    def test_strips_whitespace_around_the_name_and_value(self) -> None:
        assert parse_env("  A  =  1  \n") == {"A": "1"}

    def test_ignores_lines_without_an_equals_sign(self) -> None:
        assert parse_env("nonsense\nA=1\n") == {"A": "1"}

    def test_ignores_an_empty_name(self) -> None:
        assert parse_env("=orphan\nA=1\n") == {"A": "1"}

    def test_an_unmatched_quote_is_left_alone(self) -> None:
        assert parse_env('A="unclosed\n') == {"A": '"unclosed'}

    def test_empty_input(self) -> None:
        assert parse_env("") == {}


class TestLoadEnv:
    def test_loads_a_file_and_reports_only_the_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The return value must never carry a value: callers print it."""
        monkeypatch.delenv("SCRUPLE_TEST_KEY", raising=False)
        (tmp_path / ENV_FILENAME).write_text("SCRUPLE_TEST_KEY=s3cret\n", encoding="utf-8")

        loaded = load_env(tmp_path)
        assert loaded == ["SCRUPLE_TEST_KEY"]
        assert "s3cret" not in str(loaded)
        assert os.environ["SCRUPLE_TEST_KEY"] == "s3cret"

    def test_an_existing_variable_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An explicit export is a deliberate act and must not be overwritten by
        # a file the user may have forgotten about.
        monkeypatch.setenv("SCRUPLE_TEST_KEY", "from-shell")
        (tmp_path / ENV_FILENAME).write_text("SCRUPLE_TEST_KEY=from-file\n", encoding="utf-8")

        assert load_env(tmp_path) == []
        assert os.environ["SCRUPLE_TEST_KEY"] == "from-shell"

    def test_override_forces_the_file_to_win(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SCRUPLE_TEST_KEY", "from-shell")
        (tmp_path / ENV_FILENAME).write_text("SCRUPLE_TEST_KEY=from-file\n", encoding="utf-8")

        assert load_env(tmp_path, override=True) == ["SCRUPLE_TEST_KEY"]
        assert os.environ["SCRUPLE_TEST_KEY"] == "from-file"

    def test_an_empty_value_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ".env.example" ships with blank values; loading them would mask a real
        # variable with an empty string.
        monkeypatch.delenv("SCRUPLE_TEST_KEY", raising=False)
        (tmp_path / ENV_FILENAME).write_text("SCRUPLE_TEST_KEY=\n", encoding="utf-8")

        assert load_env(tmp_path) == []
        assert "SCRUPLE_TEST_KEY" not in os.environ

    def test_no_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_env(tmp_path) == []

    def test_defaults_to_the_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SCRUPLE_TEST_KEY", raising=False)
        (tmp_path / ENV_FILENAME).write_text("SCRUPLE_TEST_KEY=here\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert load_env() == ["SCRUPLE_TEST_KEY"]

    def test_a_directory_named_env_is_not_read(self, tmp_path: Path) -> None:
        (tmp_path / ENV_FILENAME).mkdir()
        assert load_env(tmp_path) == []


class TestSecretDetection:
    @pytest.mark.parametrize(
        "name", ["JEV_API_KEY", "openai_api_key", "SOME_TOKEN", "db_password", "MY_SECRET"]
    )
    def test_credential_names_are_recognised(self, name: str) -> None:
        assert is_secret(name) is True

    @pytest.mark.parametrize("name", ["JEV_BASE_URL", "JEV_MODEL", "PATH"])
    def test_ordinary_names_are_not(self, name: str) -> None:
        assert is_secret(name) is False
