"""Corpus loading, normalisation, chunking and split assignment."""

from .chunking import (
    AGGREGATION_RULES,
    CHARS_PER_TOKEN,
    AggregationRule,
    aggregate,
    chunk_text,
    estimate_tokens,
)
from .loader import Corpus, Item, load_corpus, load_csv, load_jsonl, load_text_folder
from .splits import Split, Splits, assign_splits

__all__ = [
    "AGGREGATION_RULES",
    "CHARS_PER_TOKEN",
    "AggregationRule",
    "Corpus",
    "Item",
    "Split",
    "Splits",
    "aggregate",
    "assign_splits",
    "chunk_text",
    "estimate_tokens",
    "load_corpus",
    "load_csv",
    "load_jsonl",
    "load_text_folder",
]
