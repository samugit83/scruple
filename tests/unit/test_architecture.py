"""Architectural rules from the design (§6).

These are not style checks. The layering is what makes the statistics testable
in isolation and the engine free of decisions, and both properties are easy to
lose one convenient import at a time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "scruple"

# Third-party modules the statistics layer is allowed to depend on. Anything
# else means the maths has acquired a dependency, which §13.2 also rules out.
STATS_ALLOWED_THIRD_PARTY = {
    "numpy",
    "scipy",
    "math",
    "abc",
    "typing",
    "dataclasses",
    "enum",
    "collections",
    "__future__",
}


def imported_modules(path: Path) -> set[str]:
    """Absolute and relative module names imported by one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("." * node.level + (node.module or ""))
            elif node.module:
                names.add(node.module)
    return names


def package_files(package: str) -> list[Path]:
    return sorted((SRC / package).glob("*.py"))


class TestStatsIsSelfContained:
    """§6, design rule 1: `stats/` MUST NOT import from `backends/`.

    The statistics are the credibility of this project, so they stay testable
    in isolation against synthetic data with known properties. In practice that
    means no imports from anywhere else in the package at all -- an import of
    `corpus` or `engine` would drag the same weight in by a different door.
    """

    @pytest.mark.parametrize("path", package_files("stats"), ids=lambda p: p.name)
    def test_no_imports_from_the_rest_of_the_package(self, path: Path) -> None:
        for module in imported_modules(path):
            if module.startswith("scruple"):
                pytest.fail(f"{path.name} imports {module}")
            # A relative import of "..something" reaches outside the package.
            if module.startswith("..") or module.startswith("scruple."):
                pytest.fail(f"{path.name} reaches outside stats/ via {module}")

    @pytest.mark.parametrize("path", package_files("stats"), ids=lambda p: p.name)
    def test_depends_only_on_numpy_and_scipy(self, path: Path) -> None:
        # §13.2: no ML framework, no agent framework. This must still install
        # cleanly in three years.
        for module in imported_modules(path):
            if module.startswith("."):
                continue
            root = module.split(".")[0]
            assert root in STATS_ALLOWED_THIRD_PARTY, f"{path.name} imports {module}"


class TestEngineMakesNoDecisions:
    """§6, design rule 2: the engine is a pure function of
    (corpus, codebook, backend) -> probabilities.

    All decisioning happens downstream in `stats/`, which is what makes
    re-thresholding free rather than a reason to re-run the model.
    """

    def test_the_runner_does_not_import_the_threshold_machinery(self) -> None:
        modules = imported_modules(SRC / "engine" / "runner.py")
        assert not any("thresholds" in m for m in modules)
        assert not any("selective" in m for m in modules)


class TestBackendsMakeNoDecisions:
    @pytest.mark.parametrize("path", package_files("backends"), ids=lambda p: p.name)
    def test_a_backend_never_imports_the_statistics(self, path: Path) -> None:
        # A backend that could see a threshold would eventually apply one.
        for module in imported_modules(path):
            assert "stats" not in module, f"{path.name} imports {module}"
