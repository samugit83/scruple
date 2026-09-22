"""Synthetic test data (§14.11).

Synthetic only. Never commit real personal data, and never commit a real
corpus.
"""

from .synthetic import (
    SyntheticCoder,
    gold_records,
    synthetic_corpus,
    synthetic_gold,
)

__all__ = [
    "SyntheticCoder",
    "gold_records",
    "synthetic_corpus",
    "synthetic_gold",
]
