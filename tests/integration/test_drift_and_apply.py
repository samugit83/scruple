"""Drift detection and applying calibration (§10.4, §7.4, §8.9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scruple.codebook import parse_codebook
from scruple.config import parse_config
from scruple.corpus import load_csv
from scruple.engine import (
    CalibrationRecord,
    chunking_fingerprint,
    detect_drift,
    new_record,
    require_no_drift,
)
from scruple.engine.apply import Source, apply_calibration
from scruple.engine.calibration import CodeCalibration
from scruple.errors import DriftError, ProjectError

CODEBOOK = parse_codebook(
    "version: 1\ncodes:\n  code_one:\n    definition: One.\n  code_two:\n    definition: Two.\n"
)


def record_for(codebook=CODEBOOK, config=None):  # type: ignore[no-untyped-def]
    config = config or parse_config("")
    record = new_record(
        backend=config.backend.name,
        model_version="stub-1",
        codebook_hash=codebook.hash,
        code_hashes=codebook.code_hashes,
        corpus_hash="corpus-hash",
        splits_seed=config.splits.seed,
        chunking=chunking_fingerprint(config),
        alpha=config.thresholds.alpha,
        delta=config.thresholds.delta,
        delta_per_candidate=0.025,
        gold_n=600,
        gold_n_double_coded=150,
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
            held_out_kappa=0.85,
            kappa_ci=[0.8, 0.9],
            n_gold=300,
            n_gold_positive=90,
        )
        for code in codebook
    }
    return record


class TestDrift:
    def test_an_unchanged_project_has_no_drift(self) -> None:
        assert detect_drift(record_for(), CODEBOOK, parse_config("")).any is False

    def test_an_edited_definition_blocks_the_run(self) -> None:
        edited = parse_codebook(
            "version: 1\ncodes:\n  code_one:\n    definition: One, revised.\n"
            "  code_two:\n    definition: Two.\n"
        )
        with pytest.raises(DriftError, match="code_one") as caught:
            require_no_drift(record_for(), edited, parse_config(""))
        assert "no longer true" in (caught.value.hint or "")

    def test_a_removed_code_is_noticed_but_does_not_block(self) -> None:
        # Nothing false can be claimed about a code that is gone.
        smaller = parse_codebook("version: 1\ncodes:\n  code_one:\n    definition: One.\n")
        drift = detect_drift(record_for(), smaller, parse_config(""))
        assert drift.removed_codes == ["code_two"]
        assert drift.blocks_run is False
        assert "removed" in drift.describe()

    def test_a_changed_backend_blocks_the_run(self) -> None:
        # A different model is a different coder; the thresholds were not
        # fitted to it.
        with pytest.raises(DriftError, match=r"backend\.name"):
            require_no_drift(record_for(), CODEBOOK, parse_config("backend:\n  name: local\n"))

    def test_a_changed_split_seed_blocks_the_run(self) -> None:
        with pytest.raises(DriftError, match=r"splits\.seed"):
            require_no_drift(record_for(), CODEBOOK, parse_config("splits:\n  seed: 7\n"))

    @pytest.mark.parametrize(
        "setting", ["max_tokens: 4096", "overlap_tokens: 50", "aggregation: mean"]
    )
    def test_any_chunking_change_blocks_the_run(self, setting: str) -> None:
        # §10.3: calibration is fitted on the aggregated score, so any of these
        # changes what was calibrated.
        with pytest.raises(DriftError, match="chunking"):
            require_no_drift(record_for(), CODEBOOK, parse_config(f"chunking:\n  {setting}\n"))

    def test_the_description_names_everything_that_changed(self) -> None:
        edited = parse_codebook(
            "version: 1\ncodes:\n  code_one:\n    definition: Revised.\n"
            "  code_three:\n    definition: New.\n"
        )
        drift = detect_drift(record_for(), edited, parse_config("chunking:\n  aggregation: mean\n"))
        described = drift.describe()
        assert "code_one" in described
        assert "code_three" in described
        assert "code_two" in described
        assert "aggregation" in described


class TestCalibrationRecord:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        record = record_for()
        record.save(tmp_path / "calibration.json")
        reloaded = CalibrationRecord.load(tmp_path / "calibration.json")
        assert reloaded.code_hashes == record.code_hashes
        assert reloaded.usable_codes == ("code_one", "code_two")
        assert reloaded.codes["code_one"].pair is not None

    def test_a_missing_file_points_at_the_next_step(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectError, match="not been calibrated") as caught:
            CalibrationRecord.load(tmp_path / "absent.json")
        assert "scruple gold" in (caught.value.hint or "")

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "calibration.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProjectError, match="not valid JSON"):
            CalibrationRecord.load(path)

    def test_an_unsupported_version(self, tmp_path: Path) -> None:
        path = tmp_path / "calibration.json"
        path.write_text('{"version": 99}', encoding="utf-8")
        with pytest.raises(ProjectError, match="unsupported calibration version"):
            CalibrationRecord.load(path)


class TestApplyCalibration:
    def corpus(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        path = tmp_path / "corpus.csv"
        path.write_text("rid,text\na,first\nb,second\nc,third\n", encoding="utf-8")
        return load_csv(path, text_column="text", id_column="rid")

    def test_confident_items_are_auto_coded(self, tmp_path: Path) -> None:
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"a": {"code_one": 0.95}, "b": {"code_one": 0.02}, "c": {"code_one": 0.5}},
        )
        assert applied.decisions["a"]["code_one"].label == 1
        assert applied.decisions["a"]["code_one"].source is Source.AUTO
        assert applied.decisions["b"]["code_one"].label == 0

    def test_items_in_the_band_abstain(self, tmp_path: Path) -> None:
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"c": {"code_one": 0.5}},
        )
        decision = applied.decisions["c"]["code_one"]
        assert decision.label is None
        assert decision.source is Source.ABSTAIN_UNREVIEWED

    def test_a_human_label_always_wins(self, tmp_path: Path) -> None:
        """A person's judgement is the answer; the probability is recorded
        beside it rather than over it."""
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"a": {"code_one": 0.99}},
            human_labels={("a", "code_one"): 0},
        )
        decision = applied.decisions["a"]["code_one"]
        assert decision.label == 0
        assert decision.source is Source.HUMAN
        assert decision.probability == 0.99

    def test_an_uncertified_code_gets_no_number_at_all(self, tmp_path: Path) -> None:
        # A guess on an uncertified code is exactly the output this project
        # exists to avoid producing.
        record = record_for()
        record.codes["code_one"] = CodeCalibration(
            code_id="code_one",
            verdict="NOT_AUTOMATABLE",
            reason="NO_VALID_THRESHOLD",
            t_lo=None,
            t_hi=None,
            coverage=None,
            held_out_kappa=None,
            kappa_ci=None,
            n_gold=300,
            n_gold_positive=90,
        )
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record,
            scores={"a": {"code_one": 0.99}},
        )
        decision = applied.decisions["a"]["code_one"]
        assert decision.label is None
        assert decision.source is Source.NOT_AUTOMATABLE

    def test_a_failed_probability_abstains_rather_than_deciding(self, tmp_path: Path) -> None:
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"a": {"code_one": None}},
        )
        assert applied.decisions["a"]["code_one"].source is Source.ABSTAIN_UNREVIEWED

    def test_the_provenance_breakdown_covers_every_cell(self, tmp_path: Path) -> None:
        # §8.9: the per-source breakdown must accompany the headline kappa.
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"a": {"code_one": 0.99, "code_two": 0.5}},
            human_labels={("b", "code_one"): 1},
        )
        counts = applied.counts_by_source()
        assert sum(counts.values()) == 3 * len(CODEBOOK)
        assert counts["auto"] >= 1
        assert counts["human"] >= 1

    def test_coverage_for_one_code(self, tmp_path: Path) -> None:
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"a": {"code_one": 0.99}, "b": {"code_one": 0.5}, "c": {"code_one": 0.01}},
        )
        assert applied.coverage_for("code_one") == pytest.approx(2 / 3)

    def test_abstentions_are_listed(self, tmp_path: Path) -> None:
        applied = apply_calibration(
            corpus=self.corpus(tmp_path),
            codebook=CODEBOOK,
            record=record_for(),
            scores={"a": {"code_one": 0.5}},
        )
        assert any(d.item_id == "a" for d in applied.abstentions())
