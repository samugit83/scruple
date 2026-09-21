"""Performance guards (plan §14.12).

Not micro-benchmarks. These catch the shapes of mistake that make a tool
unusable on a real corpus: accidental quadratic behaviour, and a warm cache
that is no faster than a cold one.

Ceilings are deliberately loose -- several times what the operation takes on a
laptop -- so they fail on a regression of kind rather than on a slow CI runner.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from scruple.codebook import parse_codebook
from scruple.config import parse_config
from scruple.corpus import load_csv
from scruple.engine import Cache, Engine

from .conftest import StubBackend

pytestmark = pytest.mark.slow

CODEBOOK = parse_codebook(
    "version: 1\n"
    "codes:\n"
    "  access_barrier:\n    definition: A practical obstacle.\n"
    "  distrust_pharma:\n    definition: Distrust of manufacturers.\n"
    "  no_recommendation:\n    definition: Nobody recommended it.\n"
)


def corpus_of(tmp_path: Path, n: int):  # type: ignore[no-untyped-def]
    path = tmp_path / f"corpus_{n}.csv"
    path.write_text(
        "rid,text\n" + "".join(f"r{i:06d},answer number {i} with some padding\n" for i in range(n)),
        encoding="utf-8",
    )
    return load_csv(path, text_column="text", id_column="rid")


class TestRunThroughput:
    def test_ten_thousand_items_complete_within_the_ceiling(self, tmp_path: Path) -> None:
        """§14.12: a 10,000-item run against a stub backend, under a wall-clock
        ceiling. The backend is instant, so this measures the engine."""
        corpus = corpus_of(tmp_path, 10_000)
        config = parse_config("engine:\n  concurrency: 8\n")

        started = time.monotonic()
        with Cache(tmp_path / "cache.sqlite") as cache:
            result = Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, CODEBOOK)
        elapsed = time.monotonic() - started

        assert len(result.scores) == 10_000
        assert elapsed < 60.0, f"10k items took {elapsed:.1f}s"

    def test_a_warm_run_is_much_faster_than_a_cold_one(self, tmp_path: Path) -> None:
        """A cache that is not faster is a cache that is not working, and the
        §12 iteration loop depends on it feeling instant."""
        corpus = corpus_of(tmp_path, 4_000)
        config = parse_config("engine:\n  concurrency: 8\n")
        backend = StubBackend(delay=0.0002)

        with Cache(tmp_path / "cache.sqlite") as cache:
            engine = Engine(backend=backend, config=config, cache=cache)
            started = time.monotonic()
            engine.run(corpus, CODEBOOK)
            cold = time.monotonic() - started

            started = time.monotonic()
            engine.run(corpus, CODEBOOK)
            warm = time.monotonic() - started

        assert warm < cold, f"warm run ({warm:.2f}s) was not faster than cold ({cold:.2f}s)"

    def test_run_time_grows_about_linearly_with_corpus_size(self, tmp_path: Path) -> None:
        """The guard against accidental quadratic behaviour.

        Ten times the items should cost roughly ten times the work, not a
        hundred. The allowance is generous because small runs are dominated by
        fixed costs, which flatters the smaller measurement.
        """
        config = parse_config("engine:\n  concurrency: 8\n")

        def timed(n: int) -> float:
            corpus = corpus_of(tmp_path, n)
            with Cache(tmp_path / f"c{n}.sqlite") as cache:
                started = time.monotonic()
                Engine(backend=StubBackend(), config=config, cache=cache).run(corpus, CODEBOOK)
                return time.monotonic() - started

        small = max(timed(1_000), 1e-3)
        large = timed(10_000)
        assert large / small < 40, f"10x the items cost {large / small:.1f}x the time"


class TestCacheLookup:
    def test_lookup_is_constant_time_in_the_cache_size(self, tmp_path: Path) -> None:
        """§14.12: cache lookup is O(1). It is a primary-key hit; if that ever
        becomes a scan, a large project degrades without any other symptom."""
        with Cache(tmp_path / "cache.sqlite") as cache:
            for i in range(50_000):
                cache.put(f"key_{i:06d}", 0.5)
            cache.commit()

            def probe(keys: list[str]) -> float:
                started = time.monotonic()
                cache.get_many(keys)
                return time.monotonic() - started

            early = probe([f"key_{i:06d}" for i in range(400)])
            late = probe([f"key_{i:06d}" for i in range(49_600, 50_000)])

        assert max(early, late) < 1.0
        # Position in the table must not matter.
        assert late < max(early * 12, 0.25)

    def test_report_generation_is_not_quadratic(self, tmp_path: Path) -> None:
        """Guard against accidental O(n^2) in report and export generation."""
        from scruple.engine.apply import apply_calibration
        from scruple.engine.calibration import CodeCalibration, new_record
        from scruple.export import coded_rows

        record = new_record(
            backend="stub",
            model_version="stub-1",
            codebook_hash="h",
            code_hashes=CODEBOOK.code_hashes,
            corpus_hash="h",
            splits_seed=42,
            chunking={},
            alpha=0.1,
            delta=0.05,
            delta_per_candidate=0.025,
            gold_n=0,
            gold_n_double_coded=0,
            exact_guarantee=True,
        )
        record.codes = {
            code.id: CodeCalibration(
                code_id=code.id,
                verdict="ok",
                reason=None,
                t_lo=0.1,
                t_hi=0.9,
                coverage=0.9,
                held_out_kappa=0.8,
                kappa_ci=None,
                n_gold=100,
                n_gold_positive=40,
            )
            for code in CODEBOOK
        }

        def timed(n: int) -> float:
            corpus = corpus_of(tmp_path, n)
            scores = {item.id: dict.fromkeys(CODEBOOK.ids, 0.95) for item in corpus}
            started = time.monotonic()
            applied = apply_calibration(
                corpus=corpus, codebook=CODEBOOK, record=record, scores=scores
            )
            coded_rows(corpus, applied)
            return time.monotonic() - started

        small = max(timed(1_000), 1e-3)
        large = timed(10_000)
        assert large / small < 40, f"10x the items cost {large / small:.1f}x the time"
