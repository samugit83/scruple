"""CLI surface tests (plan §12, §13.1).

`--json` is a public versioned interface: the Phase 2 R package wraps this CLI
as a subprocess contract rather than reimplementing the statistics, so the
envelope shape is pinned here as well as the behaviour.

All offline: the example uses the `recorded` backend.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scruple.cli.main import app
from scruple.errors import ExitCode

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "vaccine_survey"
runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "vaccine_survey"
    shutil.copytree(EXAMPLE, root)
    shutil.rmtree(root / "out", ignore_errors=True)
    for path in (root / ".scruple").iterdir():
        if path.name in {"splits.json", "gold.jsonl"}:
            continue
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
    monkeypatch.chdir(root)
    return root


def invoke(*args: str, stdin: str | None = None):  # type: ignore[no-untyped-def]
    return runner.invoke(app, list(args), input=stdin, catch_exceptions=False)


def data(result) -> dict:  # type: ignore[no-untyped-def]
    payload = json.loads(result.stdout)
    assert payload["ok"] is True, payload
    return payload["data"]


class TestJsonEnvelope:
    def test_every_success_carries_the_same_envelope(self, project: Path) -> None:
        result = invoke("codebook", "check", "--json")
        payload = json.loads(result.stdout)
        assert set(payload) == {"scruple", "command", "ok", "data"}
        assert payload["command"] == "codebook check"
        assert payload["ok"] is True

    def test_every_failure_carries_the_same_envelope(self, project: Path) -> None:
        (project / "codebook.yml").write_text("version: 1\ncodes:\n  BAD:\n    definition: x\n")
        payload = json.loads(invoke("codebook", "check", "--json").stdout)
        assert set(payload) == {"scruple", "command", "ok", "error"}
        assert set(payload["error"]) == {"code", "exit_code", "message", "hint"}
        assert payload["error"]["code"] == "VALIDATION"

    def test_the_version_is_reported(self, project: Path) -> None:
        result = invoke("--version")
        assert result.exit_code == ExitCode.OK
        assert "scruple" in result.stdout


class TestInitAndLoad:
    def test_init_refuses_to_overwrite_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert invoke("init").exit_code == ExitCode.OK
        assert invoke("init").exit_code == ExitCode.VALIDATION
        assert invoke("init", "--force").exit_code == ExitCode.OK

    def test_init_rejects_an_unknown_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = invoke("init", "--backend", "gpt5000", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "jev" in json.loads(result.stdout)["error"]["hint"]

    def test_init_reports_what_the_backend_transmits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A researcher choosing a backend must be told what leaves the machine.
        monkeypatch.chdir(tmp_path)
        payload = data(invoke("init", "--backend", "local", "--json"))
        assert "only to the endpoint you configure" in payload["privacy"]

    def test_load_refuses_a_second_time(self, project: Path) -> None:
        result = invoke("load", "data/corpus.csv", "--text-col", "response", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "splits reset" in json.loads(result.stdout)["error"]["hint"]

    def test_load_reports_the_length_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # §12: a p99 of 40,000 characters tells you immediately that you are in
        # the chunking regime, before spending anything.
        monkeypatch.chdir(tmp_path)
        invoke("init")
        (tmp_path / "corpus.csv").write_text(
            "text\n" + "".join(f"answer number {i}\n" for i in range(20)), encoding="utf-8"
        )
        payload = data(invoke("load", "corpus.csv", "--text-col", "text", "--json"))
        assert payload["items"] == 20
        assert set(payload["splits"]) == {"dev", "calibration", "test"}
        assert payload["length"]["median"] > 0


class TestSplits:
    def test_show(self, project: Path) -> None:
        payload = data(invoke("splits", "show", "--json"))
        assert payload["seed"] == 42
        assert sum(payload["counts"].values()) == 1200

    def test_reset_clears_the_split_and_calibration(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        assert (project / ".scruple" / "calibration.json").exists()

        assert invoke("splits", "reset", "--yes", "--json").exit_code == ExitCode.OK
        assert not (project / ".scruple" / "splits.json").exists()
        assert not (project / ".scruple" / "calibration.json").exists()

    def test_reset_without_a_split_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        invoke("init")
        assert invoke("splits", "reset", "--yes", "--json").exit_code == ExitCode.VALIDATION

    def test_reset_asks_before_destroying_calibration(self, project: Path) -> None:
        # §8.4: changing the partition invalidates everything, loudly.
        result = invoke("splits", "reset", stdin="n\n")
        assert result.exit_code != ExitCode.OK
        assert (project / ".scruple" / "splits.json").exists()


class TestTry:
    def test_can_be_restricted_to_named_codes(self, project: Path) -> None:
        payload = data(invoke("try", "--n", "3", "--codes", "access_barrier", "--json"))
        for per_code in payload["scores"].values():
            assert set(per_code) == {"access_barrier"}

    def test_an_unknown_code_is_reported(self, project: Path) -> None:
        result = invoke("try", "--codes", "no_such_code", "--json")
        assert result.exit_code != ExitCode.OK

    def test_human_readable_output_shows_the_text_and_probability(self, project: Path) -> None:
        result = invoke("try", "--n", "2", "--codes", "access_barrier")
        assert result.exit_code == ExitCode.OK
        assert "access_barrier" in result.stdout
        assert "cache hits" in result.stdout


class TestGold:
    def test_reports_the_work_to_be_done(self, project: Path) -> None:
        # The shipped sample is already coded, so a fresh coder has work.
        payload = data(invoke("gold", "--n", "4", "--coder", "coder_3", "--json"))
        assert payload["tasks"] > 0
        assert payload["coder"] == "coder_3"

    def test_a_coder_with_nothing_left_is_told_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        invoke("init", "--backend", "local")
        (tmp_path / "corpus.csv").write_text(
            "text\n" + "".join(f"answer number {i}\n" for i in range(40)), encoding="utf-8"
        )
        invoke("load", "corpus.csv", "--text-col", "text")
        # One item, one code, coded once -- then there is nothing left for them.
        assert invoke("gold", "--n", "1", "--coder", "solo", stdin="y\ny\n").exit_code == 0

        result = invoke("gold", "--n", "1", "--coder", "solo", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "already been judged" in json.loads(result.stdout)["error"]["hint"]

    def test_coding_a_sample_appends_blind_records(self, project: Path) -> None:
        """§6: gold records that model output was hidden -- always."""
        from scruple.gold import GoldStore

        before = len(GoldStore(project / ".scruple" / "gold.jsonl").all_records())
        # Two answers then quit: `q` stops early and keeps what was coded.
        result = invoke("gold", "--n", "1", "--coder", "coder_9", stdin="y\nn\nq\n")
        assert result.exit_code == ExitCode.OK

        store = GoldStore(project / ".scruple" / "gold.jsonl")
        added = [r for r in store.all_records() if r.coder == "coder_9"]
        assert len(store.all_records()) > before
        assert added
        assert all(r.model_visible is False for r in added)

    def test_overlap_needs_a_first_coder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        invoke("init")
        (tmp_path / "corpus.csv").write_text(
            "text\n" + "".join(f"answer {i}\n" for i in range(30)), encoding="utf-8"
        )
        invoke("load", "corpus.csv", "--text-col", "text")
        result = invoke("gold", "--overlap", "5", "--coder", "coder_2", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "scruple gold" in json.loads(result.stdout)["error"]["hint"]

    def test_overlap_draws_from_the_existing_sample(self, project: Path) -> None:
        payload = data(invoke("gold", "--overlap", "5", "--coder", "coder_4", "--json"))
        assert payload["tasks"] > 0

    def test_enrich_requires_a_known_code(self, project: Path) -> None:
        result = invoke("gold", "--enrich", "no_such_code", "--json")
        assert result.exit_code == ExitCode.VALIDATION

    def test_enrich_samples_likely_positives(self, project: Path) -> None:
        # §8.5: the remedy `check` recommends for a rare code must work.
        payload = data(
            invoke(
                "gold", "--enrich", "religious_objection", "--n", "20", "--coder", "c5", "--json"
            )
        )
        assert payload["tasks"] > 0


class TestReviewAndCompare:
    def test_review_lists_the_abstentions(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        invoke("run", "--yes")
        payload = data(invoke("review", "--json"))
        assert payload["pending"] > 0

    def test_review_records_decisions_with_the_probability_visible(self, project: Path) -> None:
        # §12: showing the probability is fine here -- the researcher is
        # adjudicating a case the machine declined, not rating blind.
        from scruple.gold import ReviewStore

        invoke("check", "--resamples", "50")
        invoke("run", "--yes")
        result = invoke("review", "--limit", "2", stdin="y\nn\n")
        assert result.exit_code == ExitCode.OK

        decisions = ReviewStore(project / ".scruple" / "review.jsonl").latest()
        assert decisions
        assert all(d.model_visible is True for d in decisions.values())

    def test_review_after_check_but_before_run_is_refused(self, project: Path) -> None:
        """A `check` writes probabilities too, but only for the gold sample.

        Applying those would produce a coded.csv in which most of the corpus is
        unreviewed -- an export that looks like it worked and did not.
        """
        invoke("check", "--resamples", "50")
        result = invoke("review", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "scruple run" in json.loads(result.stdout)["error"]["hint"]

    def test_export_after_check_but_before_run_is_refused(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        assert invoke("export", "--json").exit_code == ExitCode.VALIDATION

    def test_compare_needs_two_backends(self, project: Path) -> None:
        result = invoke("compare", "--backends", "recorded", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "at least two" in json.loads(result.stdout)["error"]["message"]

    def test_compare_reports_each_backend(self, project: Path) -> None:
        payload = data(
            invoke(
                "compare", "--backends", "recorded,recorded", "--resamples", "50", "--yes", "--json"
            )
        )
        assert len(payload["backends"]) == 2
        assert payload["backends"][0]["usable"]


class TestRunOptions:
    def test_can_be_restricted_to_named_codes(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        payload = data(invoke("run", "--codes", "access_barrier", "--yes", "--json"))
        assert payload["items"] == 1200

    def test_retry_failed_needs_recorded_failures(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        invoke("run", "--yes")
        result = invoke("run", "--retry-failed", "--yes", "--json")
        assert result.exit_code == ExitCode.VALIDATION
        assert "no recorded failures" in json.loads(result.stdout)["error"]["message"]

    def test_export_can_skip_the_report(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        invoke("run", "--yes")
        payload = data(invoke("export", "--no-report", "--json"))
        assert len(payload["files"]) == 2
        assert not (project / "out" / "validation_report.md").exists()

    @pytest.mark.parametrize("fmt", ["parquet", "stata"])
    def test_other_export_formats(self, project: Path, fmt: str) -> None:
        invoke("check", "--resamples", "50")
        invoke("run", "--yes")
        result = invoke("export", "--format", fmt, "--no-report", "--json")
        assert result.exit_code == ExitCode.OK

    def test_an_unknown_export_format_is_refused(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        invoke("run", "--yes")
        result = invoke("export", "--format", "sqlite", "--json")
        assert result.exit_code == ExitCode.VALIDATION


class TestPurge:
    def test_all_removes_every_derived_artefact(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        payload = data(invoke("purge", "--all", "--yes", "--json"))
        assert "gold.jsonl" in payload["removed"]
        assert "cache.sqlite" in payload["removed"]
        assert not (project / ".scruple" / "runs").exists()

    def test_cache_only_leaves_the_gold_sample(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        payload = data(invoke("purge", "--cache", "--yes", "--json"))
        assert payload["removed"] == ["cache.sqlite"]
        assert (project / ".scruple" / "gold.jsonl").exists()

    def test_purging_nothing_present_is_not_an_error(self, project: Path) -> None:
        invoke("purge", "--cache", "--yes")
        payload = data(invoke("purge", "--cache", "--yes", "--json"))
        assert payload["removed"] == []

    def test_purge_asks_before_destroying_personal_data(self, project: Path) -> None:
        result = invoke("purge", "--gold", stdin="n\n")
        assert result.exit_code != ExitCode.OK
        assert (project / ".scruple" / "gold.jsonl").exists()
