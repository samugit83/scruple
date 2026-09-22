"""End-to-end tests over the worked example (§14.9).

The example must run the entire pipeline **offline with no API key**, because
first-run experience decides adoption and "get an API key first" loses most of
it (§13). Nothing here touches the network: the `recorded` backend replays
probabilities from a file that ships with the example.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scruple.cli.main import app
from scruple.errors import ExitCode
from scruple.report import normalise_for_comparison

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "vaccine_survey"
runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the example, so tests never mutate the shipped one."""
    root = shipped_copy(tmp_path / "vaccine_survey")
    monkeypatch.chdir(root)
    # Prove no key is needed: remove every one a backend might reach for.
    for variable in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    return root


def invoke(*args: str):  # type: ignore[no-untyped-def]
    return runner.invoke(app, list(args), catch_exceptions=False)


def shipped_copy(root: Path) -> Path:
    """A copy of the example in exactly the state it ships in.

    Only splits.json and gold.jsonl are committed; everything else under
    .scruple/ is derived, and a stale calibration.json left by a local run
    would quietly make "refuses without calibration" pass for the wrong reason.
    """
    shutil.copytree(EXAMPLE, root)
    shutil.rmtree(root / "out", ignore_errors=True)
    for path in (root / ".scruple").iterdir():
        if path.name in {"splits.json", "gold.jsonl"}:
            continue
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
    return root


@pytest.fixture(scope="module")
def exported_once(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The full pipeline, run once for every test that only reads its outputs.

    Each check/run/export cycle over the 1,200-item example takes a couple of
    seconds; running it per test made the suite slow enough to be skipped,
    which is the state in which tests stop catching anything.
    """
    root = shipped_copy(tmp_path_factory.mktemp("exported") / "vaccine_survey")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        for variable in ("TYPESAFE_API_KEY", "JEV_API_KEY", "OPENAI_API_KEY"):
            patch.delenv(variable, raising=False)
        assert invoke("check", "--resamples", "200").exit_code == ExitCode.OK
        assert invoke("run", "--yes").exit_code == ExitCode.OK
        assert invoke("export").exit_code == ExitCode.OK
    return root


class TestTheExampleRunsOffline:
    def test_codebook_validates(self, project: Path) -> None:
        result = invoke("codebook", "check")
        assert result.exit_code == ExitCode.OK
        assert "6 codes" in result.stdout

    def test_splits_are_frozen_and_shipped(self, project: Path) -> None:
        result = invoke("splits", "show", "--json")
        data = json.loads(result.stdout)["data"]
        assert data["seed"] == 42
        assert sum(data["counts"].values()) == 1200

    def test_try_runs_on_dev_only(self, project: Path) -> None:
        result = invoke("try", "--n", "5", "--json")
        assert result.exit_code == ExitCode.OK
        scores = json.loads(result.stdout)["data"]["scores"]
        dev = set(json.loads((project / ".scruple" / "splits.json").read_text())["items"]["dev"])
        assert set(scores) <= dev

    def test_the_whole_pipeline_from_check_to_export(self, project: Path) -> None:
        """§14.9: init to export, offline, no API key."""
        check = invoke("check", "--resamples", "200", "--json")
        payload = json.loads(check.stdout)["data"]
        assert payload["usable"], "the example should certify at least one code"

        run = invoke("run", "--yes", "--json")
        assert run.exit_code == ExitCode.OK
        assert json.loads(run.stdout)["data"]["items"] == 1200
        assert json.loads(run.stdout)["data"]["failures"] == 0

        export = invoke("export", "--json")
        assert export.exit_code == ExitCode.OK
        for name in ("coded.csv", "abstentions.csv", "validation_report.md"):
            assert (project / "out" / name).exists(), name

    def test_init_creates_a_usable_project_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        monkeypatch.chdir(fresh)
        assert invoke("init", "--backend", "local").exit_code == ExitCode.OK
        assert (fresh / "scruple.yml").exists()
        assert (fresh / "codebook.yml").exists()
        assert invoke("codebook", "check").exit_code == ExitCode.OK


class TestOutputSchema:
    """§14.9: snapshot the schema; changes must be deliberate and appear in the diff."""

    @pytest.fixture
    def exported(self, exported_once: Path) -> Path:
        return exported_once / "out"

    def test_coded_csv_carries_original_columns_plus_three_per_code(self, exported: Path) -> None:
        with (exported / "coded.csv").open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header[:3] == ["respondent_id", "age_band", "response"]
        for code in ("access_barrier", "dignity_violation"):
            assert code in header
            assert f"{code}_p" in header
            assert f"{code}_src" in header

    def test_code_columns_are_zero_one_or_blank_never_floats(self, exported: Path) -> None:
        # §7.4 says 0/1/NA. "1.0" does not read back as a binary variable.
        with (exported / "coded.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1200
        assert {row["access_barrier"] for row in rows} <= {"0", "1", ""}

    def test_every_source_value_is_one_of_the_documented_four(self, exported: Path) -> None:
        with (exported / "coded.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["access_barrier_src"] for row in rows} <= {
            "auto",
            "human",
            "abstain_unreviewed",
            "not_automatable",
        }

    def test_abstentions_carry_the_text_and_probability(self, exported: Path) -> None:
        with (exported / "abstentions.csv").open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header == ["item_id", "code_id", "probability", "text"]


class TestReportStability:
    """§14.9: byte-stable modulo timestamps, paths and durations."""

    def build(self, project: Path) -> str:
        invoke("check", "--resamples", "100")
        invoke("run", "--yes")
        invoke("export", "--no-chart")
        return (project / "out" / "validation_report.md").read_text(encoding="utf-8")

    @pytest.fixture
    def report(self, exported_once: Path) -> str:
        return (exported_once / "out" / "validation_report.md").read_text(encoding="utf-8")

    def test_two_runs_produce_the_same_report(self, project: Path) -> None:
        first = normalise_for_comparison(self.build(project))
        second = normalise_for_comparison(self.build(project))
        assert first == second

    def test_the_volatile_lines_are_the_only_difference(self, project: Path) -> None:
        raw_first = self.build(project)
        raw_second = self.build(project)
        # Unnormalised they may differ (timestamps, run ids, wall clock)...
        differing = [
            (a, b)
            for a, b in zip(raw_first.splitlines(), raw_second.splitlines(), strict=True)
            if a != b
        ]
        # ...and every difference must be on a line marked volatile.
        assert all("<!-- volatile -->" in a for a, _ in differing)

    def test_the_report_states_what_the_guarantee_does_not_cover(self, report: str) -> None:
        # §8.9: the report MUST say the guarantee covers only the model-decided
        # subset, rather than letting a reader assume it covers everything.
        assert "only the model-decided subset" in report
        assert "per code" in report
        assert "marginal over items" in report

    def test_the_report_states_the_sampling_assumption(self, report: str) -> None:
        # §8.6: the assumption must be stated in plain language.
        assert "random sample of the corpus" in report
        assert "hand-picked sample would break it" in report

    def test_the_report_names_the_aggregation_rule(self, report: str) -> None:
        # §10.3: changing it invalidates calibration, so it has to be on record.
        assert "aggregation=`max`" in report

    def test_the_report_carries_a_methods_paragraph(self, report: str) -> None:
        assert "## Methods paragraph" in report
        assert "coded deductively against an author-written codebook" in report

    def test_the_report_lists_limitations_as_a_section(self, report: str) -> None:
        # §15: honest limitations are first-class, not a footnote.
        assert "## Limitations" in report
        assert "not ground truth" in report.lower()


class TestExitCodes:
    """§14.9: 0 for success, non-zero and distinct for each refusal."""

    def test_success_is_zero(self, project: Path) -> None:
        assert invoke("codebook", "check").exit_code == ExitCode.OK

    def test_a_broken_codebook_is_a_validation_failure(self, project: Path) -> None:
        (project / "codebook.yml").write_text("version: 1\ncodes:\n  BAD:\n    definition: x\n")
        result = invoke("codebook", "check")
        assert result.exit_code == ExitCode.VALIDATION

    def test_an_over_budget_run_is_refused_with_its_own_code(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        config = (project / "scruple.yml").read_text(encoding="utf-8")
        config = config.replace(
            "engine:\n  concurrency: 8", "engine:\n  concurrency: 8\n  confirm_above_calls: 1"
        )
        (project / "scruple.yml").write_text(config, encoding="utf-8")
        result = invoke("run")
        assert result.exit_code == ExitCode.BUDGET_REFUSED

    def test_editing_a_definition_after_check_refuses_the_run(self, project: Path) -> None:
        """§10.4: cheap codebook revision is a selling point, but revision
        invalidates calibration and reusing stale thresholds would print a
        guarantee that is no longer true."""
        invoke("check", "--resamples", "50")
        codebook = (project / "codebook.yml").read_text(encoding="utf-8")
        codebook = codebook.replace(
            "The respondent cites a practical obstacle",
            "The respondent mentions any obstacle whatsoever",
        )
        (project / "codebook.yml").write_text(codebook, encoding="utf-8")

        result = invoke("run", "--yes")
        assert result.exit_code == ExitCode.DRIFT_REFUSED
        assert "access_barrier" in result.stdout

    def test_changing_the_aggregation_rule_refuses_the_run(self, project: Path) -> None:
        invoke("check", "--resamples", "50")
        config = (project / "scruple.yml").read_text(encoding="utf-8")
        (project / "scruple.yml").write_text(
            config.replace("aggregation: max", "aggregation: mean"), encoding="utf-8"
        )
        result = invoke("run", "--yes")
        assert result.exit_code == ExitCode.DRIFT_REFUSED
        assert "aggregation" in result.stdout

    def test_adding_a_code_after_check_does_not_block_the_run(self, project: Path) -> None:
        # A new code has no thresholds, so it cannot be auto-coded and cannot
        # make a false claim. Blocking would punish an edit that is safe.
        invoke("check", "--resamples", "50")
        codebook = (project / "codebook.yml").read_text(encoding="utf-8")
        codebook += (
            "  brand_new_code:\n"
            "    type: noul\n"
            "    definition: Something the researcher thought of later.\n"
        )
        (project / "codebook.yml").write_text(codebook, encoding="utf-8")
        assert invoke("run", "--yes").exit_code == ExitCode.OK

    def test_running_without_calibration_is_refused(self, project: Path) -> None:
        result = invoke("run", "--yes")
        assert result.exit_code == ExitCode.VALIDATION
        assert "scruple check" in result.stdout

    def test_json_errors_carry_the_code_and_the_hint(self, project: Path) -> None:
        result = invoke("run", "--yes", "--json")
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "VALIDATION"
        assert payload["error"]["hint"]


class TestPrivacyCommands:
    def test_purge_removes_the_gold_sample(self, project: Path) -> None:
        assert (project / ".scruple" / "gold.jsonl").exists()
        result = invoke("purge", "--gold", "--yes", "--json")
        assert result.exit_code == ExitCode.OK
        assert not (project / ".scruple" / "gold.jsonl").exists()

    def test_purge_needs_a_target(self, project: Path) -> None:
        result = invoke("purge", "--yes")
        assert result.exit_code == ExitCode.VALIDATION

    def test_the_cache_survives_a_gold_purge(self, project: Path) -> None:
        # §6: the cache holds hashes only, so the expensive work can be kept
        # while the personal data is destroyed.
        invoke("check", "--resamples", "50")
        assert (project / ".scruple" / "cache.sqlite").exists()
        invoke("purge", "--gold", "--yes")
        assert (project / ".scruple" / "cache.sqlite").exists()
