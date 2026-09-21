"""Unit tests for canonical normalisation and hashing (plan §7.1, §9.4)."""

from __future__ import annotations

import pytest

from scruple.hashing import (
    cache_key,
    code_hash,
    corpus_hash,
    file_hash,
    item_hash,
    normalise_definition,
    normalise_item_text,
    sha256_hex,
)


class TestNormaliseDefinition:
    def test_collapses_yaml_folding(self) -> None:
        # A definition re-wrapped by an editor must keep the same hash, or every
        # cached probability is thrown away for nothing.
        folded = "The respondent cites a practical\n  obstacle to attending:\n  time or cost."
        flat = "The respondent cites a practical obstacle to attending: time or cost."
        assert normalise_definition(folded) == flat

    def test_strips_and_deduplicates_whitespace(self) -> None:
        assert normalise_definition("  a   b\t\tc \n") == "a b c"

    def test_applies_unicode_nfc(self) -> None:
        # "é" as e + combining acute must hash as the precomposed form.
        assert normalise_definition("café") == normalise_definition("café")


class TestNormaliseItemText:
    def test_preserves_paragraph_breaks(self) -> None:
        # §10.3 chunks at paragraph boundaries, so this structure must survive.
        assert normalise_item_text("para one\n\npara two") == "para one\n\npara two"

    def test_collapses_runs_of_blank_lines(self) -> None:
        assert normalise_item_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_normalises_line_endings(self) -> None:
        assert normalise_item_text("a\r\nb\rc") == "a\nb\nc"

    def test_strips_trailing_whitespace_per_line(self) -> None:
        assert normalise_item_text("a   \nb\t\n") == "a\nb"

    def test_collapses_inline_runs_but_not_newlines(self) -> None:
        assert normalise_item_text("a    b\nc") == "a b\nc"

    def test_empty_input(self) -> None:
        assert normalise_item_text("   \n\n  ") == ""


class TestSha256Hex:
    def test_is_stable(self) -> None:
        assert sha256_hex("a") == sha256_hex("a")

    def test_separator_prevents_boundary_collisions(self) -> None:
        # Without a length prefix, ("ab","c") and ("a","bc") would concatenate
        # to the same bytes and silently share a cache entry.
        assert sha256_hex("ab", "c") != sha256_hex("a", "bc")

    def test_no_parts(self) -> None:
        assert len(sha256_hex()) == 64


class TestCodeHash:
    def test_depends_on_the_definition(self) -> None:
        assert code_hash("a", [], []) != code_hash("b", [], [])

    def test_ignores_definition_rewrapping(self) -> None:
        assert code_hash("a  b", [], []) == code_hash("a b", [], [])

    def test_depends_on_examples(self) -> None:
        assert code_hash("a", ["x"], []) != code_hash("a", [], [])

    def test_distinguishes_positive_from_negative_examples(self) -> None:
        # The same string as a yes-example and as a no-example mean opposite
        # things and must not share a cache entry.
        assert code_hash("a", ["x"], []) != code_hash("a", [], ["x"])

    def test_depends_on_example_order(self) -> None:
        assert code_hash("a", ["x", "y"], []) != code_hash("a", ["y", "x"], [])


class TestCorpusHash:
    def test_is_order_independent(self) -> None:
        # Re-sorting a CSV is not a different corpus.
        a, b, c = item_hash("one"), item_hash("two"), item_hash("three")
        assert corpus_hash([a, b, c]) == corpus_hash([c, a, b])

    def test_changes_when_an_item_changes(self) -> None:
        assert corpus_hash([item_hash("one")]) != corpus_hash([item_hash("two")])


class TestCacheKey:
    @pytest.mark.parametrize(
        "changed",
        ["backend", "model_version", "item", "code_id", "code"],
    )
    def test_key_changes_when_any_component_changes(self, changed: str) -> None:
        """§14.5: the cache key changes if and only if one of its inputs changes."""
        base = {
            "backend": "jev",
            "model_version": "v1",
            "item": "ih",
            "code_id": "access",
            "code": "ch",
        }
        altered = {**base, changed: "different"}
        assert cache_key(**base) != cache_key(**altered)  # type: ignore[arg-type]

    def test_key_is_stable_for_identical_inputs(self) -> None:
        base = {
            "backend": "jev",
            "model_version": "v1",
            "item": "ih",
            "code_id": "access",
            "code": "ch",
        }
        assert cache_key(**base) == cache_key(**base)  # type: ignore[arg-type]


def test_file_hash(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "corpus.csv"
    path.write_text("text\nhello\n", encoding="utf-8")
    first = file_hash(path)
    assert len(first) == 64
    path.write_text("text\ngoodbye\n", encoding="utf-8")
    assert file_hash(path) != first
