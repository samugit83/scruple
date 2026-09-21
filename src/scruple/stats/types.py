"""Shared result types for the statistics layer.

Design rule (plan §6): this package MUST NOT import from ``scruple.backends``.
The statistics are the credibility of the project, so they stay testable in
isolation against synthetic data with known properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum


class ReasonCode(str, Enum):
    """Why a statistic is undefined, or why a code cannot be certified.

    These strings surface in ``--json`` output, which §13.1 makes a public
    versioned interface.
    """

    EMPTY_INPUT = "EMPTY_INPUT"
    """No items were supplied."""

    DEGENERATE_SINGLE_CLASS = "DEGENERATE_SINGLE_CLASS"
    """One class is entirely absent, so chance agreement is 1 and kappa is undefined."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    """Too few positive instances to say anything (the §8.5 hard gate)."""

    NO_VALID_THRESHOLD = "NO_VALID_THRESHOLD"
    """No candidate threshold pair controlled class-conditional risk (§8.6 step 5)."""

    ALL_ABSTAINED = "ALL_ABSTAINED"
    """The band accepted nothing, so there is no decided subset to score."""

    ZERO_WEIGHT = "ZERO_WEIGHT"
    """Every sampling weight was zero, so no weighted estimate exists."""


class Verdict(str, Enum):
    """Per-code outcome of ``scruple check`` (plan §12)."""

    OK = "ok"
    WEAK = "weak"
    NOT_AUTOMATABLE = "NOT_AUTOMATABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    @property
    def usable(self) -> bool:
        """Whether the code may be auto-coded in a run."""
        return self in (Verdict.OK, Verdict.WEAK)


@dataclass(frozen=True)
class Estimate:
    """A statistic that may legitimately be undefined.

    Invariant: ``value`` and ``reason`` are mutually exclusive and jointly
    exhaustive. A degenerate input yields ``value=None`` with a reason code, and
    never a silent ``NaN`` (§14.3).
    """

    value: float | None
    n: int
    reason: ReasonCode | None = None
    ci: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
        if self.value is None and self.reason is None:
            raise ValueError("an undefined Estimate must carry a reason code")
        if self.value is not None and self.reason is not None:
            raise ValueError("a defined Estimate must not carry a reason code")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("Estimate value must be finite, never NaN or infinite")
        if self.ci is not None and self.ci[0] > self.ci[1]:
            raise ValueError(f"confidence interval must be ordered, got {self.ci}")

    @property
    def defined(self) -> bool:
        """True when the statistic could be computed."""
        return self.value is not None

    def with_ci(self, ci: tuple[float, float] | None) -> Estimate:
        """Return a copy carrying ``ci``. The type is frozen, so this is a copy."""
        return replace(self, ci=ci)


@dataclass(frozen=True)
class Confusion:
    """A 2x2 agreement table between a gold label and a decision.

    Cells are floats rather than ints because under the enriched sampling of
    §8.5 they are sums of inverse-probability weights.

    ``a`` both positive, ``b`` gold positive / decision negative,
    ``c`` gold negative / decision positive, ``d`` both negative.
    """

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.c, self.d) < 0:
            raise ValueError("confusion cells must not be negative")

    @property
    def n(self) -> float:
        return self.a + self.b + self.c + self.d

    @property
    def gold_positive(self) -> float:
        return self.a + self.b

    @property
    def gold_negative(self) -> float:
        return self.c + self.d

    @property
    def pred_positive(self) -> float:
        return self.a + self.c

    @property
    def pred_negative(self) -> float:
        return self.b + self.d

    @property
    def errors(self) -> float:
        return self.b + self.c

    @property
    def agreements(self) -> float:
        return self.a + self.d
