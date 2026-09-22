"""Engine integration tests (§14.8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scruple.codebook import parse_codebook
from scruple.config import parse_config
from scruple.corpus import load_csv
from scruple.engine import Cache, Engine, build_units, plan_units, save_run
from scruple.errors import BudgetRefused

from .conftest import StubBackend


def engine(backend, config, cache, **kwargs):  # type: ignore[no-untyped-def]
    return Engine(backend=backend, config=config, cache=cache, **kwargs)


class TestCaching:
    def test_a_warm_cache_makes_zero_backend_calls(self, corpus, codebook, config, cache) -> None:  # type: ignore[no-untyped-def]
        """§14.8: cold cache then warm -- assert zero backend calls on the second run."""
        backend = StubBackend()
        first = engine(backend, config, cache).run(corpus, codebook)
        assert backend.scored_units == len(corpus)
        calls_after_cold = backend.usage.calls

        second = engine(backend, config, cache).run(corpus, codebook)
        assert backend.usage.calls == calls_after_cold, "the warm run hit the backend"
        assert second.scores == first.scores

    def test_editing_one_code_definition_reruns_only_that_code(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        """§14.8. This is what makes cheap codebook revision possible (§10.4)."""
        backend = StubBackend()
        engine(backend, config, cache).run(corpus, codebook)
        backend.calls.clear()

        edited = parse_codebook(
            "version: 1\n"
            "codes:\n"
            "  access_barrier:\n    definition: A practical obstacle, revised.\n"
            "  distrust_pharma:\n    definition: Distrust of manufacturers.\n"
        )
        engine(backend, config, cache).run(corpus, edited)

        assert set(backend.scored_code_ids) == {"access_barrier"}
        assert len(backend.scored_code_ids) == len(corpus)

    def test_rewrapping_a_definition_does_not_invalidate_the_cache(
        self, corpus, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        backend = StubBackend()
        flat = parse_codebook(
            "version: 1\ncodes:\n  access_barrier:\n    definition: A practical obstacle.\n"
        )
        engine(backend, config, cache).run(corpus, flat)
        backend.calls.clear()

        folded = parse_codebook(
            "version: 1\ncodes:\n  access_barrier:\n    definition: >\n      A practical\n      obstacle.\n"
        )
        engine(backend, config, cache).run(corpus, folded)
        assert backend.scored_units == 0

    def test_a_new_model_version_invalidates_the_cache(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        # A different model is a different coder; reusing its numbers would make
        # the manifest's model_version a lie.
        engine(StubBackend(version="stub-1"), config, cache).run(corpus, codebook)
        upgraded = StubBackend(version="stub-2")
        engine(upgraded, config, cache).run(corpus, codebook)
        assert upgraded.scored_units == len(corpus)

    def test_identical_texts_share_one_backend_call(
        self, tmp_path: Path, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "corpus.csv"
        path.write_text("rid,text\na,n/a\nb,n/a\nc,n/a\n", encoding="utf-8")
        duplicated = load_csv(path, text_column="text", id_column="rid")
        backend = StubBackend()
        result = engine(backend, config, cache).run(duplicated, codebook)
        assert backend.scored_units == 1
        assert result.scores["a"] == result.scores["c"]

    def test_cache_hit_counts_are_reported(self, corpus, codebook, config, cache) -> None:  # type: ignore[no-untyped-def]
        # §9.4: report cache hits so users understand why iteration is cheap.
        engine(StubBackend(), config, cache).run(corpus, codebook)
        result = engine(StubBackend(), config, cache).run(corpus, codebook)
        assert result.manifest.cache_hits > 0

    def test_failures_are_not_cached_so_retry_can_work(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        failing = StubBackend(fail_codes=frozenset({"access_barrier"}))
        engine(failing, config, cache).run(corpus, codebook)

        recovered = StubBackend()
        result = engine(recovered, config, cache).run(corpus, codebook)
        assert recovered.scored_units == len(corpus)
        assert all(scores["access_barrier"] is not None for scores in result.scores.values())


class TestFailureHandling:
    def test_a_run_completes_with_partial_results_and_reports_the_count(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        """§10.2: runs complete with partial results and report the failure count."""
        failing_texts = frozenset(item.text for item in list(corpus)[:5])
        backend = StubBackend(fail_texts=failing_texts)
        result = engine(backend, config, cache).run(corpus, codebook)

        assert len(result.scores) == len(corpus)
        assert result.manifest.failures == 5 * len(codebook)
        assert len(result.failed_item_ids) == 5

    def test_a_failure_is_none_and_never_zero(self, corpus, codebook, config, cache) -> None:  # type: ignore[no-untyped-def]
        # §10.2: a missing probability is not a negative judgement.
        backend = StubBackend(fail_texts=frozenset({next(iter(corpus)).text}))
        result = engine(backend, config, cache).run(corpus, codebook)
        first = next(iter(corpus)).id
        assert result.scores[first]["access_barrier"] is None

    def test_a_raising_backend_does_not_abort_the_run(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        backend = StubBackend(raise_on_texts=frozenset({list(corpus)[3].text}))
        result = engine(backend, config, cache).run(corpus, codebook)
        assert len(result.scores) == len(corpus)
        assert result.manifest.failures == len(codebook)

    def test_retry_failed_reruns_exactly_the_failed_items(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        """§14.8: `--retry-failed` re-runs exactly those items."""
        failing_texts = frozenset(item.text for item in list(corpus)[:5])
        first = engine(StubBackend(fail_texts=failing_texts), config, cache).run(corpus, codebook)
        assert len(first.failed_item_ids) == 5

        retry_backend = StubBackend()
        retry_corpus = corpus.subset(first.failed_item_ids)
        engine(retry_backend, config, cache).run(retry_corpus, codebook)
        assert retry_backend.scored_units == 5


class TestChunkingAndAggregation:
    def long_corpus(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        paragraphs = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(12))
        path = tmp_path / "long.csv"
        path.write_text(f'rid,text\nlong_one,"{paragraphs}"\n', encoding="utf-8")
        return load_csv(path, text_column="text", id_column="rid")

    def test_a_long_item_is_split_into_several_units(self, tmp_path: Path, codebook) -> None:  # type: ignore[no-untyped-def]
        config = parse_config("chunking:\n  max_tokens: 300\n  overlap_tokens: 20\n")
        corpus = self.long_corpus(tmp_path)
        units = plan_units(corpus, config)
        assert len(units) > 1
        assert {u.item_id for u in units} == {"long_one"}

    def test_a_short_item_is_a_single_unit(self, corpus, config) -> None:  # type: ignore[no-untyped-def]
        assert len(build_units(next(iter(corpus)), config)) == 1

    def test_calibration_sees_the_aggregate_not_the_passages(
        self, tmp_path: Path, codebook, cache
    ) -> None:  # type: ignore[no-untyped-def]
        """§10.3: calibration MUST be fitted on the aggregated score.

        The engine returns exactly one probability per (item, code) whatever the
        passage count, so there is no passage-level score for a caller to fit on
        even by accident.
        """
        config = parse_config("chunking:\n  max_tokens: 300\n  overlap_tokens: 20\n")
        corpus = self.long_corpus(tmp_path)
        backend = StubBackend()
        result = Engine(backend=backend, config=config, cache=cache).run(corpus, codebook)

        assert backend.scored_units > 1, "expected the item to be chunked"
        assert set(result.scores) == {"long_one"}
        assert set(result.scores["long_one"]) == set(codebook.ids)
        assert result.chunked_items == 1

    def test_max_aggregation_takes_the_highest_passage(
        self, tmp_path: Path, codebook, cache
    ) -> None:  # type: ignore[no-untyped-def]
        config = parse_config("chunking:\n  max_tokens: 300\n  aggregation: max\n")
        corpus = self.long_corpus(tmp_path)
        backend = StubBackend(probability=0.4)
        result = Engine(backend=backend, config=config, cache=cache).run(corpus, codebook)
        value = result.scores["long_one"]["access_barrier"]
        assert value is not None
        assert value >= 0.4

    def test_the_aggregation_rule_is_recorded_in_the_manifest(
        self, tmp_path: Path, codebook, cache
    ) -> None:  # type: ignore[no-untyped-def]
        # §10.3: the rule must be recorded, because changing it invalidates
        # calibration exactly as editing a definition does.
        config = parse_config("chunking:\n  max_tokens: 300\n  aggregation: mean\n")
        result = Engine(backend=StubBackend(), config=config, cache=cache).run(
            self.long_corpus(tmp_path), codebook
        )
        assert result.manifest.chunking["aggregation"] == "mean"
        assert result.manifest.chunking["max_tokens"] == 300

    def test_a_chunked_item_with_all_passages_failing_is_none(
        self, tmp_path: Path, codebook, cache
    ) -> None:  # type: ignore[no-untyped-def]
        config = parse_config("chunking:\n  max_tokens: 300\n")
        corpus = self.long_corpus(tmp_path)
        units = plan_units(corpus, config)
        backend = StubBackend(fail_texts=frozenset(u.text for u in units))
        result = Engine(backend=backend, config=config, cache=cache).run(corpus, codebook)
        assert result.scores["long_one"]["access_barrier"] is None


class TestCostGate:
    def test_a_run_over_budget_halts_and_requires_confirmation(
        self, corpus, codebook, cache
    ) -> None:  # type: ignore[no-untyped-def]
        """§14.8: a run projected above budget halts and requires confirmation."""
        config = parse_config(
            "engine:\n"
            "  price_per_million_input: 1000.0\n"
            "  price_per_million_output: 1000.0\n"
            "  budget_usd: 0.001\n"
        )
        backend = StubBackend()
        with pytest.raises(BudgetRefused, match="projected to cost"):
            Engine(backend=backend, config=config, cache=cache).run(
                corpus, codebook, approved=False
            )
        assert backend.scored_units == 0, "money must not be spent before the gate"

    def test_an_approved_run_proceeds(self, corpus, codebook, cache) -> None:  # type: ignore[no-untyped-def]
        config = parse_config("engine:\n  price_per_million_input: 1000.0\n  budget_usd: 0.001\n")
        result = Engine(backend=StubBackend(), config=config, cache=cache).run(
            corpus, codebook, approved=True
        )
        assert len(result.scores) == len(corpus)

    def test_an_estimate_is_available_without_spending(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        backend = StubBackend()
        estimate = Engine(backend=backend, config=config, cache=cache).estimate(corpus, codebook)
        assert estimate.items == len(corpus)
        assert estimate.codes == len(codebook)
        assert estimate.input_tokens > 0
        assert backend.scored_units == 0


class TestManifestAndPersistence:
    def test_the_manifest_records_the_documented_provenance(
        self, corpus, codebook, config, cache, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """§7.3: the report quotes these verbatim."""
        backend = StubBackend(version="stub-7.7.7")
        result = Engine(backend=backend, config=config, cache=cache).run(
            corpus, codebook, corpus_path=Path(corpus.source), splits_seed=42
        )
        manifest = result.manifest

        assert manifest.backend == "stub"
        assert manifest.model_version == "stub-7.7.7"
        assert manifest.codebook_hash == codebook.hash
        assert manifest.code_hashes == codebook.code_hashes
        assert manifest.corpus_hash == corpus.hash
        assert manifest.corpus_file_hash
        assert manifest.splits_seed == 42
        assert manifest.item_count == len(corpus)
        assert manifest.code_count == len(codebook)
        assert manifest.calls > 0
        assert manifest.wall_clock_seconds >= 0
        assert manifest.scruple_version
        assert manifest.python_version

    def test_raw_probabilities_are_stored_never_just_decisions(
        self, corpus, codebook, config, cache, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """§6, design rule 4: every number must be reproducible from files on disk,
        and re-thresholding must never require re-running the model."""
        import json

        result = Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)
        directory = save_run(result, tmp_path / "runs")

        stored = json.loads((directory / "probabilities.json").read_text(encoding="utf-8"))
        assert set(stored) == set(corpus.ids)
        first = stored[next(iter(corpus.ids))]
        assert isinstance(first["access_barrier"], float)
        assert (directory / "manifest.json").exists()

    def test_failures_are_persisted_for_retry(
        self, corpus, codebook, config, cache, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        backend = StubBackend(fail_texts=frozenset({next(iter(corpus)).text}))
        result = Engine(backend=backend, config=config, cache=cache).run(corpus, codebook)
        directory = save_run(result, tmp_path / "runs")
        assert (directory / "failures.json").exists()

    def test_a_clean_run_writes_no_failures_file(
        self, corpus, codebook, config, cache, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        result = Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)
        assert not (save_run(result, tmp_path / "runs") / "failures.json").exists()

    def test_the_manifest_round_trips(
        self, corpus, codebook, config, cache, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        from scruple.engine import Manifest

        result = Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)
        path = tmp_path / "manifest.json"
        result.manifest.save(path)
        assert Manifest.load(path).codebook_hash == codebook.hash


class TestConcurrency:
    def test_results_are_independent_of_worker_count(
        self, corpus, codebook, tmp_path: Path
    ) -> None:  # type: ignore[no-untyped-def]
        outcomes = []
        for workers in (1, 8):
            config = parse_config(f"engine:\n  concurrency: {workers}\n")
            with Cache(tmp_path / f"cache_{workers}.sqlite") as cache:
                outcomes.append(
                    Engine(backend=StubBackend(), config=config, cache=cache)
                    .run(corpus, codebook)
                    .scores
                )
        assert outcomes[0] == outcomes[1]

    def test_every_item_is_scored_under_concurrency(self, corpus, codebook, cache) -> None:  # type: ignore[no-untyped-def]
        # §10.1: never silently drop an item.
        config = parse_config("engine:\n  concurrency: 16\n")
        result = Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)
        assert set(result.scores) == set(corpus.ids)
        assert all(set(v) == set(codebook.ids) for v in result.scores.values())

    def test_probabilities_for_one_code_are_extractable(
        self, corpus, codebook, config, cache
    ) -> None:  # type: ignore[no-untyped-def]
        result = Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)
        per_code = result.probabilities_for("access_barrier")
        assert set(per_code) == set(corpus.ids)


class TestCacheStore:
    def test_the_cache_stores_no_text(self, corpus, codebook, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """§6 and §9.4: hashes only, never text.

        `gold.jsonl` and `review.jsonl` are personal data at rest; the cache
        deliberately is not, so `scruple purge --gold` can remove the personal
        data while leaving the expensive work intact.
        """
        path = tmp_path / "cache.sqlite"
        with Cache(path) as cache:
            Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)

        raw = path.read_bytes()
        for item in corpus:
            assert item.text.encode("utf-8") not in raw
        for code in codebook:
            assert code.definition.encode("utf-8") not in raw

    def test_purge_empties_the_cache(self, corpus, codebook, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        with Cache(tmp_path / "cache.sqlite") as cache:
            Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, codebook)
            assert len(cache) > 0
            removed = cache.purge()
            assert removed > 0
            assert len(cache) == 0

    def test_a_closed_cache_refuses_use(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path / "cache.sqlite")
        cache.close()
        with pytest.raises(RuntimeError, match="closed"):
            cache.get("k")

    def test_closing_twice_is_harmless(self, tmp_path: Path) -> None:
        cache = Cache(tmp_path / "cache.sqlite")
        cache.close()
        cache.close()

    def test_survives_a_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.sqlite"
        with Cache(path) as cache:
            cache.put("k", 0.5)
            cache.commit()
        with Cache(path) as reopened:
            assert reopened.get("k") == 0.5

    def test_get_many_handles_more_keys_than_sqlite_allows_variables(self, tmp_path: Path) -> None:
        # SQLite caps bound variables per statement; the lookup must window.
        with Cache(tmp_path / "cache.sqlite") as cache:
            for i in range(1200):
                cache.put(f"k{i}", 0.5)
            cache.commit()
            found = cache.get_many([f"k{i}" for i in range(1200)])
            assert len(found) == 1200

    def test_deduplication_is_visible_in_the_hit_counters(
        self, tmp_path: Path, codebook, config
    ) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "corpus.csv"
        path.write_text(
            "rid,text\n" + "".join(f"r{i},repeated answer\n" for i in range(20)),
            encoding="utf-8",
        )
        duplicated = load_csv(path, text_column="text", id_column="rid")
        backend = StubBackend()
        with Cache(tmp_path / "cache.sqlite") as cache:
            Engine(backend=backend, config=config, cache=cache).run(duplicated, codebook)
        assert backend.scored_units == 1
