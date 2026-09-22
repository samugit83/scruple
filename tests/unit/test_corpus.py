"""Unit tests for corpus loading, normalisation and chunking (§7.2, §10.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scruple.corpus import Corpus, aggregate, chunk_text, estimate_tokens, load_corpus
from scruple.corpus.loader import load_csv, load_jsonl, load_text_folder
from scruple.errors import ValidationError


def write_csv(tmp_path: Path, rows: str, name: str = "corpus.csv") -> Path:
    path = tmp_path / name
    path.write_text(rows, encoding="utf-8")
    return path


class TestLoadCsv:
    def test_loads_and_generates_stable_ids(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "text\nfirst answer\nsecond answer\n")
        corpus = load_csv(path, text_column="text")
        assert len(corpus) == 2
        assert corpus.ids == ("row_1", "row_2")

    def test_generated_ids_are_zero_padded_to_a_constant_width(self, tmp_path: Path) -> None:
        # So that sorting ids lexically matches sorting them numerically.
        rows = "text\n" + "".join(f"answer {i}\n" for i in range(12))
        corpus = load_csv(write_csv(tmp_path, rows), text_column="text")
        assert corpus.ids[0] == "row_01"
        assert corpus.ids[-1] == "row_12"

    def test_uses_an_id_column_when_given(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "rid,text\nr9,first\nr4,second\n")
        corpus = load_csv(path, text_column="text", id_column="rid")
        assert corpus.ids == ("r9", "r4")

    def test_keeps_original_columns_for_export(self, tmp_path: Path) -> None:
        # §7.4: coded.csv carries the original columns through.
        path = write_csv(tmp_path, "rid,age,text\nr1,42,first\n")
        corpus = load_csv(path, text_column="text", id_column="rid")
        assert corpus.columns == ("rid", "age", "text")
        assert corpus.items[0].row["age"] == "42"

    def test_normalises_text_at_load(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, 'text\n"  spaced    out  "\n')
        assert load_csv(path, text_column="text").items[0].text == "spaced out"

    def test_missing_text_column_lists_what_is_available(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "response\nfirst\n")
        with pytest.raises(ValidationError, match="no column 'text'") as caught:
            load_csv(path, text_column="text")
        assert "response" in (caught.value.hint or "")

    def test_missing_id_column(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "text\nfirst\n")
        with pytest.raises(ValidationError, match="no id column"):
            load_csv(path, text_column="text", id_column="rid")

    def test_blank_rows_are_refused_not_dropped(self, tmp_path: Path) -> None:
        """Dropping rows silently changes the denominator of every statistic."""
        path = write_csv(tmp_path, 'text\nfirst\n""\nthird\n')
        with pytest.raises(ValidationError, match="1 row\\(s\\) have empty") as caught:
            load_csv(path, text_column="text")
        assert "denominator" in (caught.value.hint or "")

    def test_many_blank_rows_are_summarised_not_all_listed(self, tmp_path: Path) -> None:
        rows = "text\n" + '""\n' * 9
        with pytest.raises(ValidationError, match="and 4 more"):
            load_csv(write_csv(tmp_path, rows), text_column="text")

    def test_duplicate_ids_are_refused(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "rid,text\nr1,first\nr1,second\n")
        with pytest.raises(ValidationError, match="duplicate item id 'r1' at rows 1 and 2"):
            load_csv(path, text_column="text", id_column="rid")

    def test_empty_id_value_is_refused(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, 'rid,text\n"",first\n')
        with pytest.raises(ValidationError, match="empty 'rid'"):
            load_csv(path, text_column="text", id_column="rid")

    def test_no_rows(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="no rows"):
            load_csv(write_csv(tmp_path, "text\n"), text_column="text")

    def test_unreadable_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.csv"
        path.write_bytes(b"\xff\xfe\x00invalid")
        with pytest.raises(ValidationError, match="could not read"):
            load_csv(path, text_column="text")

    def test_tsv_is_detected_by_suffix(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "rid\ttext\nr1\tfirst answer\n", name="corpus.tsv")
        corpus = load_csv(path, text_column="text", id_column="rid")
        assert corpus.items[0].text == "first answer"


class TestLoadJsonl:
    def test_loads_objects(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            json.dumps({"rid": "a", "text": "first"})
            + "\n"
            + json.dumps({"rid": "b", "text": "second"})
            + "\n",
            encoding="utf-8",
        )
        corpus = load_jsonl(path, text_column="text", id_column="rid")
        assert corpus.ids == ("a", "b")

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"text": "first"}\n\n{"text": "second"}\n', encoding="utf-8")
        assert len(load_jsonl(path, text_column="text")) == 2

    def test_malformed_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"text": "ok"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(ValidationError, match=":2 is not valid JSON"):
            load_jsonl(path, text_column="text")

    def test_non_object_line_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"text": "ok"}\n[1, 2]\n', encoding="utf-8")
        with pytest.raises(ValidationError, match=":2 is not a JSON object"):
            load_jsonl(path, text_column="text")


class TestLoadTextFolder:
    def test_filename_stems_become_ids(self, tmp_path: Path) -> None:
        # Interview transcripts keep the names the researcher already uses.
        (tmp_path / "P01.txt").write_text("first interview", encoding="utf-8")
        (tmp_path / "P02.md").write_text("second interview", encoding="utf-8")
        corpus = load_text_folder(tmp_path)
        assert corpus.ids == ("P01", "P02")
        assert corpus.items[0].row["source_file"] == "P01.txt"

    def test_other_suffixes_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "keep.txt").write_text("text", encoding="utf-8")
        (tmp_path / "skip.pdf").write_bytes(b"%PDF")
        assert load_text_folder(tmp_path).ids == ("keep",)

    def test_a_folder_with_no_text_files(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match=r"no \.txt or \.md files"):
            load_text_folder(tmp_path)


class TestLoadCorpusDispatch:
    def test_dispatches_on_suffix(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path, "text\nfirst\n")
        assert len(load_corpus(path, text_column="text")) == 1

    def test_dispatches_to_a_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "texts"
        folder.mkdir()
        (folder / "a.txt").write_text("content", encoding="utf-8")
        assert load_corpus(folder).ids == ("a",)

    def test_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="no corpus at"):
            load_corpus(tmp_path / "absent.csv")

    def test_unsupported_format_lists_the_supported_ones(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.xlsx"
        path.write_bytes(b"x")
        with pytest.raises(ValidationError, match="unsupported corpus format") as caught:
            load_corpus(path)
        assert ".jsonl" in (caught.value.hint or "")


class TestCorpusObject:
    @pytest.fixture
    def corpus(self, tmp_path: Path) -> Corpus:
        rows = "rid,text\n" + "".join(f"r{i},answer number {i}\n" for i in range(10))
        return load_csv(write_csv(tmp_path, rows), text_column="text", id_column="rid")

    def test_identical_text_shares_an_item_hash(self, tmp_path: Path) -> None:
        # Repeated answers like "n/a" must cost the backend nothing twice (§9.4).
        path = write_csv(tmp_path, "rid,text\na,n/a\nb,n/a\n")
        corpus = load_csv(path, text_column="text", id_column="rid")
        assert corpus.items[0].item_hash == corpus.items[1].item_hash

    def test_corpus_hash_is_order_independent(self, tmp_path: Path) -> None:
        forward = load_csv(
            write_csv(tmp_path, "rid,text\na,one\nb,two\n", "f.csv"),
            text_column="text",
            id_column="rid",
        )
        reversed_ = load_csv(
            write_csv(tmp_path, "rid,text\nb,two\na,one\n", "r.csv"),
            text_column="text",
            id_column="rid",
        )
        assert forward.hash == reversed_.hash

    def test_subset_preserves_corpus_order(self, corpus: Corpus) -> None:
        assert corpus.subset(["r5", "r1"]).ids == ("r1", "r5")

    def test_by_id(self, corpus: Corpus) -> None:
        assert corpus.by_id()["r3"].text == "answer number 3"

    def test_length_stats(self, corpus: Corpus) -> None:
        stats = corpus.length_stats()
        assert stats["n"] == 10
        assert stats["min"] <= stats["median"] <= stats["max"]

    def test_length_stats_of_an_empty_corpus(self) -> None:
        empty = Corpus(items=(), source="x", text_column="text")
        assert empty.length_stats() == {}


class TestChunking:
    def test_short_text_is_one_chunk(self) -> None:
        assert chunk_text("a short answer", max_tokens=1000) == ["a short answer"]

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("   ", max_tokens=100) == []

    def test_splits_at_paragraph_boundaries(self) -> None:
        text = "\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(6))
        chunks = chunk_text(text, max_tokens=80)
        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)

    def test_no_text_is_lost(self) -> None:
        words = [f"w{i}" for i in range(400)]
        text = "\n\n".join(" ".join(words[i : i + 20]) for i in range(0, 400, 20))
        chunks = chunk_text(text, max_tokens=60)
        joined = " ".join(chunks)
        assert all(word in joined for word in words)

    def test_overlap_carries_context_forward(self) -> None:
        text = "\n\n".join(f"para{i} " + "word " * 30 for i in range(5))
        with_overlap = chunk_text(text, max_tokens=60, overlap_tokens=10)
        without = chunk_text(text, max_tokens=60, overlap_tokens=0)
        assert sum(len(c) for c in with_overlap) > sum(len(c) for c in without)

    def test_an_oversized_single_paragraph_is_split_at_sentences(self) -> None:
        text = ". ".join(f"sentence number {i} with some padding words" for i in range(40)) + "."
        chunks = chunk_text(text, max_tokens=50)
        assert len(chunks) > 1

    def test_an_oversized_single_sentence_is_hard_wrapped(self) -> None:
        text = "word " * 500
        chunks = chunk_text(text, max_tokens=40)
        assert len(chunks) > 1
        assert all(len(c) <= 40 * 4 + 10 for c in chunks)

    def test_a_single_unbroken_token_is_still_split(self) -> None:
        chunks = chunk_text("x" * 1000, max_tokens=20)
        assert len(chunks) > 1

    @pytest.mark.parametrize(
        ("max_tokens", "overlap", "problem"),
        [(0, 0, "max_tokens"), (100, -1, "overlap_tokens"), (100, 100, "smaller")],
    )
    def test_invalid_budgets(self, max_tokens: int, overlap: int, problem: str) -> None:
        with pytest.raises(ValueError, match=problem):
            chunk_text("text", max_tokens=max_tokens, overlap_tokens=overlap)

    def test_estimate_tokens_grows_with_length(self) -> None:
        assert estimate_tokens("word " * 100) > estimate_tokens("word")


class TestAggregation:
    def test_max(self) -> None:
        assert aggregate([0.1, 0.9, 0.3], rule="max") == pytest.approx(0.9)

    def test_mean(self) -> None:
        assert aggregate([0.0, 1.0], rule="mean") == pytest.approx(0.5)

    def test_failed_passages_are_skipped_not_counted_as_zero(self) -> None:
        # §10.2: a missing probability is not a negative judgement.
        assert aggregate([None, 0.8], rule="mean") == pytest.approx(0.8)

    def test_all_passages_failed_gives_none(self) -> None:
        assert aggregate([None, None], rule="max") is None

    def test_no_passages_gives_none(self) -> None:
        assert aggregate([], rule="max") is None

    def test_unknown_rule_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown aggregation rule"):
            aggregate([0.5], rule="median")
