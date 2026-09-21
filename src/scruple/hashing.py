"""Canonical text normalisation and hashing.

Hashes are cache keys and provenance records, so what counts as "the same text"
has to be decided once, here, and never drift. Two rules, deliberately
different:

* **Definitions** are prose. YAML block folding wraps them at arbitrary
  columns, so all whitespace collapses to single spaces before hashing --
  otherwise re-indenting a codebook would invalidate every cached probability.
* **Item text** keeps its paragraph structure, because §10.3 chunks long
  documents at paragraph boundaries. Collapsing newlines here would destroy the
  structure the chunker needs.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_INLINE_SPACE = re.compile(r"[ \t]+")


def normalise_definition(text: str) -> str:
    """Collapse all whitespace: re-wrapping a definition must not change its hash."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalise_item_text(text: str) -> str:
    """Normalise an item while preserving paragraph breaks for the chunker."""
    normalised = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalised = _INLINE_SPACE.sub(" ", normalised)
    normalised = _TRAILING_SPACE.sub("\n", normalised)
    return _BLANK_LINES.sub("\n\n", normalised).strip()


def sha256_hex(*parts: str) -> str:
    """Hash a sequence of strings with an unambiguous separator.

    The length prefix keeps ("ab", "c") from colliding with ("a", "bc").
    """
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def item_hash(text: str) -> str:
    """Hash of an already-normalised item text."""
    return sha256_hex("item", text)


def code_hash(definition: str, examples_yes: Iterable[str], examples_no: Iterable[str]) -> str:
    """Hash of a code's meaning (§7.1).

    Editing one definition must invalidate only that code's cached
    probabilities, which is what makes the `try` loop of §12 feel instant.
    """
    parts = ["code", normalise_definition(definition), "yes"]
    parts += [normalise_definition(e) for e in examples_yes]
    parts.append("no")
    parts += [normalise_definition(e) for e in examples_no]
    return sha256_hex(*parts)


def corpus_hash(item_hashes: Iterable[str]) -> str:
    """Order-independent hash of a corpus, for the run manifest (§7.3)."""
    return sha256_hex("corpus", *sorted(item_hashes))


def file_hash(path: Path) -> str:
    """Hash a file's bytes, for recording the untouched input in the manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(*, backend: str, model_version: str, item: str, code_id: str, code: str) -> str:
    """The cache key of §9.4. Hashes only -- never text."""
    return sha256_hex("cache", backend, model_version, item, code_id, code)
