"""The backend protocol and shared behaviour (plan §9).

A backend turns (text, codes) into one probability per code. Nothing above this
layer knows which model produced a number, and nothing below it makes a
decision: all thresholding happens downstream in `stats/` (§6, design rule 2),
so re-thresholding never requires re-running the model.

Three invariants every backend must honour, enforced here rather than trusted
(§14.7):

1. The returned mapping contains **every** requested code id.
2. Values are floats in [0, 1] or ``None``.
3. A failure is ``None``, never 0.0. A missing probability is not a negative
   judgement (§10.2).
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..codebook import Code
from ..corpus.chunking import estimate_tokens
from ..errors import BackendError
from .http import SleepFn

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Per-call token limits (§9.1).

    Jev evaluates questions in parallel and in isolation, so fanning out over
    codes is nearly free -- but the request still has to fit. Two separate
    ceilings apply: the state plus any single question, and the state plus all
    questions together.
    """

    max_state_and_question: int = 32_000
    max_state_and_all: int = 64_000

    def __post_init__(self) -> None:
        if self.max_state_and_question < 1 or self.max_state_and_all < 1:
            raise ValueError("token budgets must be positive")
        if self.max_state_and_all < self.max_state_and_question:
            raise ValueError("max_state_and_all cannot be below max_state_and_question")


@dataclass
class Usage:
    """Accumulated cost and traffic, reported in the run manifest (§7.3)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    failures: int = 0
    retries: int = 0

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.failures += other.failures
        self.retries += other.retries


@runtime_checkable
class Backend(Protocol):
    """The contract of §9. Deliberately tiny."""

    name: str

    def model_version(self) -> str:
        """The exact model version string, recorded verbatim in the manifest (§7.3)."""
        ...

    def score(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        """P(code applies) in [0, 1] per code.

        Must not raise on individual code failure; return ``None`` for that code
        and log it.
        """
        ...


def question_tokens(code: Code) -> int:
    """Token cost of asking about one code."""
    parts = [code.definition, *code.examples_yes, *code.examples_no]
    return estimate_tokens(" ".join(parts))


def split_codes(state: str, codes: Sequence[Code], budget: TokenBudget) -> list[list[Code]]:
    """Group codes into calls that fit the budget (§9.1).

    Every code appears in exactly one batch -- no duplicates, no omissions
    (§14.7). A code whose question alone exceeds the per-question ceiling is
    still emitted, in a batch of its own: refusing to ask would silently drop it,
    and a clear failure from the backend is better than a missing code.
    """
    if not codes:
        return []
    state_tokens = estimate_tokens(state)
    room = budget.max_state_and_all - state_tokens
    if room <= 0:
        # The state alone busts the budget; the caller should have chunked it.
        # One code per call at least keeps the failure attributable.
        return [[code] for code in codes]

    batches: list[list[Code]] = []
    current: list[Code] = []
    used = 0
    for code in codes:
        cost = question_tokens(code)
        if current and used + cost > room:
            batches.append(current)
            current = []
            used = 0
        current.append(code)
        used += cost
    if current:
        batches.append(current)
    return batches


def validate_probability(value: object, code_id: str, backend: str) -> float | None:
    """Coerce one backend value to a probability, or refuse it clearly.

    §14.7: a malformed response is rejected rather than coerced. Clamping an
    out-of-range number would hide a real backend bug behind a plausible value,
    and every guarantee downstream assumes the probability is meaningful.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendError(
            f"{backend} returned {value!r} for code {code_id!r}, which is not a number"
        )
    probability = float(value)
    if probability != probability:  # NaN
        raise BackendError(f"{backend} returned NaN for code {code_id!r}")
    if not 0.0 <= probability <= 1.0:
        raise BackendError(f"{backend} returned {probability} for code {code_id!r}, outside [0, 1]")
    return probability


class BaseBackend(abc.ABC):
    """Shared batching, validation and usage accounting."""

    name: str = "base"

    def __init__(self, *, budget: TokenBudget | None = None, sleep: SleepFn | None = None) -> None:
        self.budget = budget or TokenBudget()
        self.usage = Usage()
        # Injectable so contract tests exercise the backoff path without waiting
        # it out. §14.2 keeps every tier fast and offline.
        self.sleep = sleep

    @abc.abstractmethod
    def model_version(self) -> str: ...

    @abc.abstractmethod
    def _score_batch(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        """Score one batch that already fits the token budget."""

    def price_per_million_tokens(self) -> tuple[float, float]:
        """(input, output) USD per million tokens, for the §10.1 cost gate.

        Zero means "unknown or free", and the cost gate says so rather than
        inventing a number.
        """
        return (0.0, 0.0)

    def score(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        """Score every code, splitting into as many calls as the budget needs."""
        out: dict[str, float | None] = {}
        for batch in split_codes(state, codes, self.budget):
            try:
                scored = self._score_batch(state, batch)
            except BackendError:
                raise
            except Exception as exc:
                # A whole batch failing must not lose the other batches, and must
                # not be recorded as a negative judgement.
                logger.warning("%s: batch of %d codes failed: %s", self.name, len(batch), exc)
                self.usage.failures += len(batch)
                scored = {code.id: None for code in batch}
            for code in batch:
                if code.id not in scored:
                    raise BackendError(
                        f"{self.name} omitted code {code.id!r} from its response",
                        hint="Every requested code must appear in the result (§14.7).",
                    )
                out[code.id] = validate_probability(scored[code.id], code.id, self.name)
        return out

    def estimate_tokens_for(self, state: str, codes: Sequence[Code]) -> int:
        """Projected input tokens, used by the cost gate before spending (§10.1)."""
        batches = split_codes(state, codes, self.budget)
        state_cost = estimate_tokens(state)
        return sum(state_cost + sum(question_tokens(c) for c in batch) for batch in batches)
