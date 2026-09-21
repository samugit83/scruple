"""The frozen three-way split (plan §8.4).

Partitioned once, deterministically, at `load` time, and recorded in
`.scruple/splits.json`:

* **dev** -- the only split `try` may draw from. The researcher iterates on code
  definitions here while looking at model output. That is fitting, and it has to
  be quarantined.
* **calibration** -- gold items drawn from here fit the thresholds.
* **test** -- gold items drawn from here produce every number reported as a
  result.

Assignment is by sorted keyed hash rather than by shuffling, so it depends only
on (seed, item id) and on the corpus size: two machines reach the same partition
without sharing an RNG implementation, and the partition does not depend on the
order rows happen to appear in the file.

Slice sizes are exact rather than approximate, which is the deliberate trade:
assigning each item independently by hash value would survive rows being appended
to the corpus, but would give a researcher who asked for 45% calibration whatever
the coin flips produced. Since the partition is frozen on disk and immutable once
written, growth is handled by detecting it -- `splits.json` records the corpus
hash, and a changed corpus is refused rather than silently re-partitioned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from ..errors import ProjectError, SplitViolation
from ..hashing import sha256_hex
from ..stats.sealed import Sealed


class Split(StrEnum):
    """The three partitions. Order is the pipeline order, not alphabetical."""

    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"


@dataclass(frozen=True)
class Splits:
    """A frozen partition of the corpus."""

    seed: int
    corpus_hash: str
    created_at: str
    assignment: Mapping[str, Split]
    fractions: Mapping[Split, float]

    def matches_corpus(self, corpus_hash: str) -> bool:
        """Whether this partition was built from the corpus now on disk."""
        return self.corpus_hash == corpus_hash

    def require_corpus(self, corpus_hash: str) -> None:
        """Refuse to use a partition built from a different corpus.

        Slice boundaries depend on the corpus size, so re-partitioning a changed
        corpus would move items between splits and invalidate every number
        already computed from them.
        """
        if not self.matches_corpus(corpus_hash):
            raise ProjectError(
                "the corpus has changed since the split was frozen "
                f"(split recorded {self.corpus_hash[:12]}, corpus is now {corpus_hash[:12]})",
                hint=(
                    "Calibration and every reported number belong to the old corpus. "
                    "Run `scruple splits reset` to re-partition, which invalidates them."
                ),
            )

    def of(self, item_id: str) -> Split:
        try:
            return self.assignment[item_id]
        except KeyError:
            raise KeyError(f"item {item_id!r} is not in the frozen split") from None

    def ids(self, split: Split) -> tuple[str, ...]:
        return tuple(i for i, s in self.assignment.items() if s is split)

    def counts(self) -> dict[Split, int]:
        return {split: len(self.ids(split)) for split in Split}

    def require(self, item_ids: Sequence[str], allowed: Split, *, what: str) -> None:
        """Refuse if any id falls outside `allowed` (§8.4, §12: `try` is dev-only)."""
        strays = sorted({i for i in item_ids if self.assignment.get(i) is not allowed})
        if strays:
            shown = ", ".join(strays[:5])
            raise SplitViolation(
                f"{what} may only use the {allowed.value} split, but {len(strays)} "
                f"item(s) come from elsewhere: {shown}",
                hint=(
                    "Iterating on definitions while looking at calibration or test data is "
                    "fitting, and it invalidates every number computed from them."
                ),
            )

    def sealed(self, split: Split = Split.TEST) -> Sealed[tuple[str, ...]]:
        """The ids of a split, sealed against fitting code (§8.4)."""
        return Sealed(self.ids(split), label=split.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "seed": self.seed,
            "corpus_hash": self.corpus_hash,
            "created_at": self.created_at,
            "fractions": {s.value: self.fractions[s] for s in Split},
            "counts": {s.value: len(self.ids(s)) for s in Split},
            # The explicit assignment is the audit record. It is what makes a
            # reviewer able to check the partition rather than trust the seed.
            "items": {s.value: list(self.ids(s)) for s in Split},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Splits:
        if data.get("version") != 1:
            raise ProjectError(f"unsupported splits.json version {data.get('version')!r}")

        items = data.get("items")
        if not isinstance(items, dict):
            raise ProjectError("splits.json has a malformed `items` section")
        assignment: dict[str, Split] = {}
        for name, ids in items.items():
            split = Split(str(name))
            if not isinstance(ids, list):
                raise ProjectError(f"splits.json has a malformed `items.{name}` list")
            for item_id in ids:
                assignment[str(item_id)] = split

        raw_fractions = data.get("fractions")
        fractions_map = raw_fractions if isinstance(raw_fractions, dict) else {}
        fractions = {s: float(fractions_map.get(s.value, 0.0)) for s in Split}

        seed = data.get("seed")
        if not isinstance(seed, int):
            raise ProjectError("splits.json is missing an integer `seed`")

        return cls(
            seed=seed,
            corpus_hash=str(data.get("corpus_hash", "")),
            created_at=str(data.get("created_at", "")),
            assignment=assignment,
            fractions=fractions,
        )

    @classmethod
    def load(cls, path: Path) -> Splits:
        if not path.exists():
            raise ProjectError(
                f"no frozen split at {path}",
                hint="Run `scruple load` to register a corpus and freeze the split.",
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def _split_key(seed: int, item_id: str) -> str:
    return sha256_hex("split", str(seed), item_id)


def assign_splits(
    item_ids: Sequence[str],
    *,
    corpus_hash: str,
    seed: int = 42,
    dev: float = 0.10,
    calibration: float = 0.45,
    test: float = 0.45,
) -> Splits:
    """Partition deterministically into dev / calibration / test.

    Sizes are exact rather than approximate: items are ordered by a keyed hash
    and then sliced, so a 1,000-item corpus at 10/45/45 gives exactly 100, 450
    and 450 rather than whatever the coin flips produced.
    """
    total = dev + calibration + test
    if abs(total - 1.0) > 1e-9:
        raise ProjectError(
            f"split fractions must sum to 1.0, got {total:g} "
            f"(dev={dev:g}, calibration={calibration:g}, test={test:g})"
        )
    if min(dev, calibration, test) < 0:
        raise ProjectError("split fractions must not be negative")
    if not item_ids:
        raise ProjectError("cannot partition an empty corpus")
    if len(set(item_ids)) != len(item_ids):
        raise ProjectError("cannot partition a corpus with duplicate item ids")

    ordered = sorted(item_ids, key=lambda i: _split_key(seed, i))
    n = len(ordered)
    n_dev = round(dev * n)
    n_calibration = round(calibration * n)
    # Test takes the remainder so the three counts always sum to n exactly.
    n_test = n - n_dev - n_calibration
    if n_test < 0:  # pragma: no cover - guarded by the fractions check above
        raise ProjectError("rounding produced a negative test split; check the fractions")

    assignment: dict[str, Split] = {}
    for item_id in ordered[:n_dev]:
        assignment[item_id] = Split.DEV
    for item_id in ordered[n_dev : n_dev + n_calibration]:
        assignment[item_id] = Split.CALIBRATION
    for item_id in ordered[n_dev + n_calibration :]:
        assignment[item_id] = Split.TEST

    return Splits(
        seed=seed,
        corpus_hash=corpus_hash,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        assignment=assignment,
        fractions={Split.DEV: dev, Split.CALIBRATION: calibration, Split.TEST: test},
    )
