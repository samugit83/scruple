"""Fixtures for engine integration tests (plan §14.8)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from scruple.backends.base import BaseBackend
from scruple.codebook import Code, Codebook, parse_codebook
from scruple.config import ProjectConfig, parse_config
from scruple.corpus import Corpus, load_csv
from scruple.engine import Cache
from scruple.errors import BackendError


@dataclass
class StubBackend(BaseBackend):
    """A deterministic in-process backend that counts what it was asked.

    Not a mock of a specific API -- a stand-in for "some model", so integration
    tests measure the engine's behaviour rather than a wire format.
    """

    name: str = "stub"
    version: str = "stub-1"
    probability: float = 0.8
    fail_codes: frozenset[str] = frozenset()
    fail_texts: frozenset[str] = frozenset()
    raise_on_texts: frozenset[str] = frozenset()
    delay: float = 0.0
    """Simulated latency, so a warm-cache test has something to save."""

    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        BaseBackend.__init__(self)

    def model_version(self) -> str:
        return self.version

    @property
    def scored_units(self) -> int:
        return len(self.calls)

    @property
    def scored_code_ids(self) -> list[str]:
        return [code_id for _, ids in self.calls for code_id in ids]

    def _score_batch(self, state: str, codes: Sequence[Code]) -> dict[str, float | None]:
        self.calls.append((state, tuple(code.id for code in codes)))
        self.usage.calls += 1
        if self.delay:
            import time

            time.sleep(self.delay)
        self.usage.input_tokens += 100
        if state in self.raise_on_texts:
            raise BackendError("stub was told to fail")
        out: dict[str, float | None] = {}
        for code in codes:
            if code.id in self.fail_codes or state in self.fail_texts:
                out[code.id] = None
            else:
                # Vary by code so different codes are distinguishable downstream.
                out[code.id] = min(1.0, self.probability + 0.01 * (len(code.id) % 5))
        return out


@pytest.fixture
def codebook() -> Codebook:
    return parse_codebook(
        "version: 1\n"
        "codes:\n"
        "  access_barrier:\n    definition: A practical obstacle.\n"
        "  distrust_pharma:\n    definition: Distrust of manufacturers.\n"
    )


@pytest.fixture
def config() -> ProjectConfig:
    return parse_config("")


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    rows = "rid,text\n" + "".join(f"r{i:03d},answer number {i}\n" for i in range(100))
    path = tmp_path / "corpus.csv"
    path.write_text(rows, encoding="utf-8")
    return load_csv(path, text_column="text", id_column="rid")


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    with Cache(tmp_path / ".scruple" / "cache.sqlite") as handle:
        yield handle
