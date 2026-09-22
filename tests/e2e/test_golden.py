"""Golden snapshot tests (§14.9).

Snapshots the `scruple check` table and the coded.csv schema. A change to
either must be deliberate and must show up in a diff, rather than arriving
unnoticed in someone's output.

Run with `--snapshot-update` to re-record after an intended change:

    uv run pytest tests/e2e/test_golden.py --snapshot-update
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scruple.cli.main import app

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "vaccine_survey"
SNAPSHOTS = Path(__file__).parent / "snapshots"
runner = CliRunner()


SNAPSHOT_WIDTH = 100
"""rich wraps to the terminal width, so a snapshot recorded on one machine
would not match another. Pinning it makes the comparison about content."""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from scruple.cli import main as cli_main

    monkeypatch.setattr(cli_main.console, "width", SNAPSHOT_WIDTH)

    root = tmp_path / "vaccine_survey"
    shutil.copytree(EXAMPLE, root)
    shutil.rmtree(root / "out", ignore_errors=True)
    for path in (root / ".scruple").iterdir():
        if path.name in {"splits.json", "gold.jsonl"}:
            continue
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
    monkeypatch.chdir(root)
    return root


def compare(name: str, actual: str, request: pytest.FixtureRequest) -> None:
    """Compare against the recorded snapshot, or re-record when asked."""
    SNAPSHOTS.mkdir(exist_ok=True)
    path = SNAPSHOTS / name
    if request.config.getoption("--snapshot-update", default=False) or not path.exists():
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"recorded snapshot {name}")
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{name} changed. If that was deliberate, re-record with --snapshot-update "
        "and make sure the diff is in your commit."
    )


def test_check_table_is_stable(project: Path, request: pytest.FixtureRequest) -> None:
    """The `check` table is the product (§12); its shape should not drift."""
    result = runner.invoke(app, ["check", "--resamples", "200"], catch_exceptions=False)
    # Width is pinned above; collapse runs of spaces so column padding does not
    # make the snapshot brittle about alignment.
    collapsed = re.sub(r"[ \t]+", " ", result.stdout)
    collapsed = re.sub(r"\n+", "\n", collapsed).strip()
    compare("check_table.txt", collapsed + "\n", request)


def test_coded_csv_schema_is_stable(project: Path, request: pytest.FixtureRequest) -> None:
    runner.invoke(app, ["check", "--resamples", "50"], catch_exceptions=False)
    runner.invoke(app, ["run", "--yes"], catch_exceptions=False)
    runner.invoke(app, ["export", "--no-report"], catch_exceptions=False)

    with (project / "out" / "coded.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    compare("coded_schema.txt", "\n".join(header) + "\n", request)


def test_abstentions_schema_is_stable(project: Path, request: pytest.FixtureRequest) -> None:
    runner.invoke(app, ["check", "--resamples", "50"], catch_exceptions=False)
    runner.invoke(app, ["run", "--yes"], catch_exceptions=False)
    runner.invoke(app, ["export", "--no-report"], catch_exceptions=False)

    with (project / "out" / "abstentions.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    compare("abstentions_schema.txt", "\n".join(header) + "\n", request)
