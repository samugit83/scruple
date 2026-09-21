"""Unit tests for scruple.yml parsing (plan §7.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scruple.config import (
    DEFAULT_BACKEND,
    ProjectConfig,
    config_template,
    dump_config,
    load_config,
    parse_config,
)
from scruple.errors import ValidationError


class TestDefaults:
    def test_the_template_parses(self) -> None:
        config = parse_config(config_template())
        assert config.version == 1
        assert config.backend.name == DEFAULT_BACKEND

    def test_an_empty_file_gives_the_documented_defaults(self) -> None:
        config = parse_config("")
        assert config.splits.seed == 42
        assert (config.splits.dev, config.splits.calibration, config.splits.test) == (
            0.10,
            0.45,
            0.45,
        )
        # §7.2 proposed gold.n = 300 and alpha = 0.05. Measurement showed those
        # to be mutually incompatible: certifying a backend with 2% per-class
        # error at alpha = 0.05 needs roughly 1,400 gold items, where alpha =
        # 0.10 needs about 360. The defaults are the pair that actually works
        # together; see ThresholdsConfig and GoldConfig for the arithmetic.
        assert config.gold.n == 600
        assert config.gold.double_coded_overlap == 150
        assert config.thresholds.alpha == 0.10
        assert config.thresholds.delta == 0.05
        assert config.chunking.max_tokens == 28_000
        assert config.chunking.aggregation == "max"
        assert config.engine.concurrency == 16
        assert config.engine.budget_usd == 5.0

    def test_the_default_backend_is_the_one_chosen_in_17_4(self) -> None:
        # A values decision: hosted and calibrated by default, with the privacy
        # trade-off documented and a local backend available (§9.3).
        assert DEFAULT_BACKEND == "jev"
        assert config_template(backend="local").count("name: local") == 1

    def test_round_trips_through_yaml(self) -> None:
        original = parse_config(config_template())
        assert parse_config(dump_config(original)) == original


class TestRejections:
    def test_unsupported_version(self) -> None:
        with pytest.raises(ValidationError, match="unsupported config version"):
            parse_config("version: 2")

    def test_unknown_top_level_key(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            parse_config("nonsense: 1")

    def test_unknown_nested_key_is_named(self) -> None:
        # A typo must not be silently ignored: it would look like it took effect.
        with pytest.raises(ValidationError, match=r"chunking\.agregation"):
            parse_config("chunking:\n  agregation: max\n")

    def test_split_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"must be 1\.0"):
            parse_config("splits:\n  dev: 0.5\n  calibration: 0.4\n  test: 0.4\n")

    def test_alpha_must_be_a_probability(self) -> None:
        with pytest.raises(ValidationError, match=r"thresholds\.alpha"):
            parse_config("thresholds:\n  alpha: 0\n")

    def test_delta_must_be_a_probability(self) -> None:
        with pytest.raises(ValidationError, match=r"thresholds\.delta"):
            parse_config("thresholds:\n  delta: 1.0\n")

    def test_overlap_cannot_exceed_the_gold_sample(self) -> None:
        with pytest.raises(ValidationError, match=r"cannot exceed gold\.n"):
            parse_config("gold:\n  n: 50\n  double_coded_overlap: 80\n")

    def test_chunk_overlap_must_be_smaller_than_the_budget(self) -> None:
        with pytest.raises(ValidationError, match="smaller than max_tokens"):
            parse_config("chunking:\n  max_tokens: 1000\n  overlap_tokens: 1000\n")

    def test_unknown_aggregation_rule(self) -> None:
        # The rule is frozen before calibration (§10.3), so it must be a known one.
        with pytest.raises(ValidationError, match=r"chunking\.aggregation"):
            parse_config("chunking:\n  aggregation: median\n")

    def test_concurrency_bounds(self) -> None:
        with pytest.raises(ValidationError, match=r"engine\.concurrency"):
            parse_config("engine:\n  concurrency: 0\n")

    def test_invalid_yaml(self) -> None:
        with pytest.raises(ValidationError, match="not valid YAML"):
            parse_config("corpus:\n  - [unclosed\n")

    def test_not_a_mapping(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            parse_config("- a\n")


class TestLoadFromDisk:
    def test_loads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scruple.yml"
        path.write_text(config_template(), encoding="utf-8")
        assert isinstance(load_config(path), ProjectConfig)

    def test_missing_file_points_at_init(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match=r"no scruple\.yml") as caught:
            load_config(tmp_path / "scruple.yml")
        assert "scruple init" in (caught.value.hint or "")
