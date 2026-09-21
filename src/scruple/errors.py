"""Exceptions and process exit codes.

§14.9 requires distinct, asserted exit codes: a script wrapping scruple has to
be able to tell "your codebook is invalid" from "I refused to spend your money"
without parsing English.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes. Part of the public CLI contract (§13.1)."""

    OK = 0
    USAGE = 1
    VALIDATION = 2
    BUDGET_REFUSED = 3
    DRIFT_REFUSED = 4
    NOT_CERTIFIED = 5
    BACKEND = 6
    SPLIT_VIOLATION = 7


class ScrupleError(Exception):
    """Base class for errors scruple reports to the user rather than tracebacks."""

    exit_code: ExitCode = ExitCode.USAGE

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ValidationError(ScrupleError):
    """A codebook, config or corpus file is not usable as given."""

    exit_code = ExitCode.VALIDATION


class ProjectError(ScrupleError):
    """The project directory is missing, incomplete or inconsistent."""

    exit_code = ExitCode.VALIDATION


class BudgetRefused(ScrupleError):
    """A run was projected to cost more than the configured budget (§10.1)."""

    exit_code = ExitCode.BUDGET_REFUSED


class DriftError(ScrupleError):
    """Codebook, chunking or aggregation settings changed since calibration (§10.4)."""

    exit_code = ExitCode.DRIFT_REFUSED


class NotCertified(ScrupleError):
    """No code passed `check`, so there is nothing to run."""

    exit_code = ExitCode.NOT_CERTIFIED


class BackendError(ScrupleError):
    """A backend could not be reached or returned an unusable response."""

    exit_code = ExitCode.BACKEND


class SplitViolation(ScrupleError):
    """Something tried to read a split it must not touch (§8.4)."""

    exit_code = ExitCode.SPLIT_VIOLATION
