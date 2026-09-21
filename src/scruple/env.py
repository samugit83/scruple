"""Reading `.env` (plan §9.1).

A researcher should not have to export shell variables to use the tool. If a
`.env` sits in the project directory, scruple reads it -- without overwriting
anything already set in the environment, so an explicit export still wins.

Nothing here ever logs a value. A key that reaches a log file or a run manifest
is a leaked key, and §6 already treats the project directory as holding data
that must not spread.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"

SECRET_MARKERS = ("key", "token", "secret", "password")


def parse_env(text: str) -> dict[str, str]:
    """Parse `KEY=value` lines, ignoring comments and blanks."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def load_env(directory: Path | None = None, *, override: bool = False) -> list[str]:
    """Load `.env` from `directory` into the environment.

    Returns the names loaded, never the values, so a caller can report what was
    picked up without printing a secret.
    """
    root = Path.cwd() if directory is None else directory
    path = root / ENV_FILENAME
    if not path.is_file():
        return []

    loaded = []
    for name, value in parse_env(path.read_text(encoding="utf-8")).items():
        if not value:
            continue
        if override or name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def is_secret(name: str) -> bool:
    """Whether a variable name looks like something that must not be printed."""
    lowered = name.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)
