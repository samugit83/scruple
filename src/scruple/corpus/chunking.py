"""Chunking long documents and aggregating passage scores (§10.3).

Interview transcripts are the headline use case and they routinely exceed a
backend's state budget, so they are split into passages, scored separately and
aggregated back to one score per item.

The aggregation rule matters more than it looks. ``max`` over many passages is
strongly biased upward: a negative interview with twenty passages each at
p=0.3 aggregates close to 1.0 under a noisy-or intuition and sits at 0.3 under
``max``, and in neither case does the passage-level probability mean what it
says at item level. Calibration is therefore fitted on the **aggregated** score
(§10.3), and the rule is frozen in ``scruple.yml`` before calibration and
recorded in the manifest, because changing it invalidates calibration exactly
as changing a code definition does.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

AggregationRule = Literal["max", "mean"]

AGGREGATION_RULES: tuple[str, ...] = ("max", "mean")

CHARS_PER_TOKEN = 4.0
"""Tokens are estimated from character count rather than by importing a
tokeniser: §13.2 rules out heavyweight dependencies, and the estimate only has
to be conservative enough to stay inside a budget. It is deliberately rough,
and the engine leaves headroom accordingly."""

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for budget arithmetic."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _split_oversized(piece: str, max_chars: int) -> list[str]:
    """Break a single over-long paragraph at sentence, then word, boundaries."""
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_BREAK.split(piece):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)

    # A single sentence can still be longer than the budget; fall back to a hard
    # wrap at word boundaries rather than dropping text.
    wrapped: list[str] = []
    for part in out:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            wrapped.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            wrapped.append(part)
    return wrapped


def chunk_text(text: str, *, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Split ``text`` at paragraph boundaries into chunks within ``max_tokens``.

    Chunks carry ``overlap_tokens`` of trailing context from the previous chunk
    so a passage straddling a boundary is not cut in half.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must not be negative, got {overlap_tokens}")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    stripped = text.strip()
    if not stripped:
        return []
    if estimate_tokens(stripped) <= max_tokens:
        return [stripped]

    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    pieces: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(stripped):
        para = paragraph.strip()
        if not para:
            continue
        if len(para) > max_chars:
            pieces.extend(_split_oversized(para, max_chars))
        else:
            pieces.append(para)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(current)
            tail = current[-overlap_chars:].lstrip() if overlap_chars else ""
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def aggregate(scores: Sequence[float | None], rule: str = "max") -> float | None:
    """Aggregate passage probabilities to one item-level probability.

    Failed passages arrive as ``None`` and are skipped; if every passage failed
    the item has no probability at all, and the result is ``None`` rather than
    0.0. A missing probability is not a negative judgement (§10.2).
    """
    if rule not in AGGREGATION_RULES:
        raise ValueError(f"unknown aggregation rule {rule!r}; expected one of {AGGREGATION_RULES}")
    usable = [s for s in scores if s is not None]
    if not usable:
        return None
    if rule == "max":
        return max(usable)
    return sum(usable) / len(usable)
