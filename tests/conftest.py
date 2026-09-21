"""Shared pytest configuration."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """`--snapshot-update` re-records the golden files of §14.9."""
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        help="Re-record golden snapshots instead of comparing against them.",
    )
