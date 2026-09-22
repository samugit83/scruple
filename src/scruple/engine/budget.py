"""The cost gate (§10.1).

Researchers are not engineers and must never be surprised by a bill. So before
any run that looks expensive, scruple prints what it is about to do and asks.

Prices come from configuration rather than from a table inside the package: a
hardcoded per-model price list would be wrong within months. When prices are not
configured the estimate still reports calls and tokens, and says plainly that
the cost is unknown -- which is more useful than a confident wrong number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..codebook import Codebook
from ..config import EngineConfig
from ..corpus.chunking import estimate_tokens
from ..errors import BudgetRefused

# Output is a probability per code, so completions are tiny next to the prompt.
ESTIMATED_OUTPUT_TOKENS_PER_CALL = 32


@dataclass(frozen=True)
class CostEstimate:
    """A projection, before spending anything."""

    items: int
    codes: int
    units: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    @property
    def cost_known(self) -> bool:
        return self.cost_usd is not None

    def describe(self) -> str:
        chunk_note = f" ({self.units} passages after chunking)" if self.units != self.items else ""
        lines = [
            f"{self.items} items x {self.codes} codes{chunk_note}",
            f"{self.calls} backend call(s), about {self.input_tokens:,} input tokens",
        ]
        if self.cost_usd is not None:
            lines.append(f"estimated cost: ${self.cost_usd:,.2f}")
        else:
            lines.append(
                "estimated cost: unknown -- set engine.price_per_million_input and "
                "price_per_million_output in scruple.yml to see money here"
            )
        return "\n".join(lines)


def estimate_cost(
    *,
    texts: Sequence[str],
    codebook: Codebook,
    engine: EngineConfig,
    calls_per_unit: int = 1,
    question_tokens: int = 0,
) -> CostEstimate:
    """Project the cost of scoring `texts` against `codebook`.

    `texts` is the list of units actually sent to the backend, so a chunked
    corpus is counted at passage level rather than item level.
    """
    units = len(texts)
    calls = units * max(1, calls_per_unit)
    input_tokens = sum(estimate_tokens(text) for text in texts) * max(1, calls_per_unit)
    input_tokens += question_tokens * calls
    output_tokens = calls * ESTIMATED_OUTPUT_TOKENS_PER_CALL

    cost: float | None = None
    if engine.price_per_million_input is not None or engine.price_per_million_output is not None:
        price_in = engine.price_per_million_input or 0.0
        price_out = engine.price_per_million_output or 0.0
        cost = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

    return CostEstimate(
        items=units,
        codes=len(codebook),
        units=units,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )


def needs_confirmation(estimate: CostEstimate, engine: EngineConfig) -> bool:
    """Whether this run should stop and ask first.

    With prices configured, the test is the configured dollar budget. Without
    them, call count stands in: it is a crude proxy, but it does mean an
    unpriced backend cannot quietly run up a 50,000-call bill.
    """
    if estimate.cost_usd is not None:
        return estimate.cost_usd > engine.budget_usd
    return estimate.calls > engine.confirm_above_calls


def require_confirmation(
    estimate: CostEstimate,
    engine: EngineConfig,
    *,
    approved: bool,
) -> None:
    """Raise unless an over-budget run has been approved."""
    if not needs_confirmation(estimate, engine) or approved:
        return
    if estimate.cost_usd is not None:
        headline = (
            f"this run is projected to cost ${estimate.cost_usd:,.2f}, "
            f"above the ${engine.budget_usd:,.2f} budget"
        )
    else:
        headline = (
            f"this run needs {estimate.calls:,} backend calls, above the "
            f"{engine.confirm_above_calls:,} that run without asking"
        )
    raise BudgetRefused(
        headline,
        hint="Re-run with --yes to approve, or raise engine.budget_usd in scruple.yml.",
    )
