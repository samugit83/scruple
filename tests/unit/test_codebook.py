"""Unit tests for codebook parsing and validation (§7.1)."""

from __future__ import annotations

import pytest

from scruple.codebook import (
    MAX_EXAMPLES,
    TEMPLATE,
    Codebook,
    CodeType,
    load_codebook,
    parse_codebook,
    single_sentence_warnings,
)
from scruple.errors import ValidationError


def book(body: str) -> Codebook:
    return parse_codebook(f"version: 1\ncodes:\n{body}")


MINIMAL = "  access_barrier:\n    definition: A practical obstacle.\n"


class TestValidCodebooks:
    def test_the_template_parses(self) -> None:
        cb = parse_codebook(TEMPLATE)
        assert cb.ids == ("access_barrier",)
        assert cb["access_barrier"].type is CodeType.NOUL
        assert cb["access_barrier"].examples_yes
        assert cb["access_barrier"].examples_no

    def test_type_defaults_to_noul(self) -> None:
        assert book(MINIMAL)["access_barrier"].type is CodeType.NOUL

    def test_declaration_order_is_preserved(self) -> None:
        # The `check` table reads in the order the researcher wrote.
        cb = book("  zebra_code:\n    definition: One.\n  alpha_code:\n    definition: Two.\n")
        assert cb.ids == ("zebra_code", "alpha_code")

    def test_examples_are_stripped(self) -> None:
        cb = book(MINIMAL + '    examples_yes:\n      - "  padded  "\n')
        assert cb["access_barrier"].examples_yes == ("padded",)

    def test_max_examples_is_allowed(self) -> None:
        examples = "".join(f'      - "e{i}"\n' for i in range(MAX_EXAMPLES))
        cb = book(MINIMAL + f"    examples_yes:\n{examples}")
        assert len(cb["access_barrier"].examples_yes) == MAX_EXAMPLES


class TestCodebookRejections:
    @pytest.mark.parametrize(
        "code_id",
        ["A", "ab", "1abc", "has-dash", "has space", "has_UPPER", "_leading", "x" * 42],
    )
    def test_invalid_ids(self, code_id: str) -> None:
        with pytest.raises(ValidationError, match="invalid code id"):
            book(f"  {code_id}:\n    definition: x is y.\n")

    def test_duplicate_ids_are_an_error_not_a_silent_overwrite(self) -> None:
        """PyYAML keeps the last of two identical keys; §7.1 requires an error.

        Silently dropping one of two definitions of the same code is the kind of
        quiet data loss that invalidates a study without anyone noticing.
        """
        with pytest.raises(ValidationError, match="duplicate key 'dup_code'"):
            book("  dup_code:\n    definition: One.\n  dup_code:\n    definition: Two.\n")

    def test_missing_definition(self) -> None:
        with pytest.raises(ValidationError, match="non-empty definition"):
            book("  access_barrier:\n    type: noul\n")

    def test_blank_definition(self) -> None:
        with pytest.raises(ValidationError, match="non-empty definition"):
            book('  access_barrier:\n    definition: "   "\n')

    def test_unknown_key(self) -> None:
        with pytest.raises(ValidationError, match="unknown keys: weight"):
            book(MINIMAL + "    weight: 3\n")

    def test_unknown_type(self) -> None:
        with pytest.raises(ValidationError, match="unknown type"):
            book("  access_barrier:\n    type: freeform\n    definition: x.\n")

    @pytest.mark.parametrize("unimplemented", ["score", "choice"])
    def test_v1_5_types_are_refused_with_a_pointer_to_the_scope(self, unimplemented: str) -> None:
        with pytest.raises(ValidationError, match="not implemented in v1"):
            book(f"  access_barrier:\n    type: {unimplemented}\n    definition: x.\n")

    def test_too_many_examples(self) -> None:
        examples = "".join(f'      - "e{i}"\n' for i in range(MAX_EXAMPLES + 1))
        with pytest.raises(ValidationError, match="maximum is 5"):
            book(MINIMAL + f"    examples_yes:\n{examples}")

    def test_examples_must_be_a_list(self) -> None:
        with pytest.raises(ValidationError, match="must be a list"):
            book(MINIMAL + "    examples_yes: nope\n")

    def test_empty_example(self) -> None:
        with pytest.raises(ValidationError, match="empty or non-string"):
            book(MINIMAL + '    examples_no:\n      - ""\n')

    def test_no_codes_section(self) -> None:
        with pytest.raises(ValidationError, match="no `codes:` section"):
            parse_codebook("version: 1\n")

    def test_empty_codes_section(self) -> None:
        with pytest.raises(ValidationError, match="defines no codes"):
            parse_codebook("version: 1\ncodes: {}\n")

    def test_empty_file(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            parse_codebook("")

    def test_not_a_mapping(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            parse_codebook("- a\n- b\n")

    def test_code_body_not_a_mapping(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            book("  access_barrier: just a string\n")

    def test_codes_section_not_a_mapping(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            parse_codebook("version: 1\ncodes:\n  - a\n")

    def test_unsupported_version(self) -> None:
        with pytest.raises(ValidationError, match="unsupported codebook version"):
            parse_codebook("version: 99\ncodes: {}\n")

    def test_invalid_yaml(self) -> None:
        with pytest.raises(ValidationError, match="not valid YAML"):
            parse_codebook("codes:\n  - [unclosed\n")


class TestHashing:
    def test_editing_one_definition_changes_only_that_code_hash(self) -> None:
        """This is what makes the `try` loop of §12 feel instant."""
        before = book("  code_one:\n    definition: One.\n  code_two:\n    definition: Two.\n")
        after = book(
            "  code_one:\n    definition: One, revised.\n  code_two:\n    definition: Two.\n"
        )
        assert before["code_one"].code_hash != after["code_one"].code_hash
        assert before["code_two"].code_hash == after["code_two"].code_hash
        assert before.hash != after.hash

    def test_rewrapping_a_definition_does_not_change_its_hash(self) -> None:
        wrapped = parse_codebook(
            "version: 1\ncodes:\n  code_one:\n    definition: >\n      A practical\n      obstacle.\n"
        )
        flat = book("  code_one:\n    definition: A practical obstacle.\n")
        assert wrapped["code_one"].code_hash == flat["code_one"].code_hash

    def test_code_hashes_map_covers_every_code(self) -> None:
        cb = book("  code_one:\n    definition: One.\n  code_two:\n    definition: Two.\n")
        assert set(cb.code_hashes) == {"code_one", "code_two"}


class TestCodebookAccess:
    def test_membership_and_length(self) -> None:
        cb = book(MINIMAL)
        assert "access_barrier" in cb
        assert "nope" not in cb
        assert len(cb) == 1
        assert [c.id for c in cb] == ["access_barrier"]

    def test_unknown_code_names_the_alternatives(self) -> None:
        with pytest.raises(KeyError, match="access_barrier"):
            book(MINIMAL)["nope"]

    def test_subset_preserves_order_and_drops_the_rest(self) -> None:
        cb = book(
            "  code_one:\n    definition: One.\n"
            "  code_two:\n    definition: Two.\n"
            "  code_three:\n    definition: Three.\n"
        )
        assert cb.subset(["code_three", "code_one"]).ids == ("code_one", "code_three")

    def test_subset_rejects_unknown_codes(self) -> None:
        with pytest.raises(KeyError, match="unknown codes: ghost"):
            book(MINIMAL).subset(["ghost"])


class TestAdvisoryWarnings:
    def test_a_multi_sentence_definition_is_flagged_not_refused(self) -> None:
        cb = book("  code_one:\n    definition: One thing. Also another. And a third.\n")
        warnings = single_sentence_warnings(cb)
        assert any("two codes" in w for w in warnings)

    def test_a_very_long_definition_is_flagged(self) -> None:
        cb = book(f"  code_one:\n    definition: {'word ' * 100}.\n")
        assert any("consider trimming" in w for w in single_sentence_warnings(cb))

    def test_a_clean_definition_produces_no_warnings(self) -> None:
        assert single_sentence_warnings(book(MINIMAL)) == []


class TestLoadFromDisk:
    def test_loads_a_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "codebook.yml"
        path.write_text(TEMPLATE, encoding="utf-8")
        cb = load_codebook(path)
        assert cb.path == str(path)

    def test_missing_file_points_at_init(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValidationError, match="no codebook at") as caught:
            load_codebook(tmp_path / "absent.yml")
        assert caught.value.hint is not None
        assert "scruple init" in caught.value.hint
