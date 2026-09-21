"""The gold and review stores (plan §6).

`gold.jsonl` is **append-only** and records the timestamp, the coder id, and
whether model output was visible at coding time -- always ``false`` for gold.
This is the audit trail that makes the study defensible: a reviewer can check
that the human coded blind, rather than take it on trust.

Append-only means a correction is a new line, not an edit. Analysis reads the
last judgement for each (item, code, coder); the earlier one stays on disk, so
the record of what changed is never lost.

Both files contain raw text and are therefore personal data at rest. The cache
deliberately holds hashes only, so `scruple purge --gold` can remove the
personal data while leaving the expensive work intact (§6, §9.4).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..errors import ProjectError


@dataclass(frozen=True)
class GoldRecord:
    """One human judgement on one (item, code) pair."""

    item_id: str
    code_id: str
    label: int
    coder: str
    timestamp: str
    split: str
    stratum: str
    inclusion_probability: float
    model_visible: bool
    sample_seed: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"gold label must be 0 or 1, got {self.label!r}")
        if not 0.0 < self.inclusion_probability <= 1.0:
            raise ValueError("inclusion probability must lie in (0, 1]")

    @property
    def weight(self) -> float:
        return 1.0 / self.inclusion_probability

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.item_id, self.code_id, self.coder)

    @classmethod
    def from_row(cls, row: dict[str, object], *, source: str) -> GoldRecord:
        """Parse a stored line, checking its types rather than trusting them."""
        try:
            return cls(
                item_id=str(row["item_id"]),
                code_id=str(row["code_id"]),
                label=_as_int(row["label"], "label"),
                coder=str(row["coder"]),
                timestamp=str(row["timestamp"]),
                split=str(row.get("split", "")),
                stratum=str(row.get("stratum", "uniform")),
                inclusion_probability=_as_float(
                    row.get("inclusion_probability", 1.0), "inclusion_probability"
                ),
                model_visible=bool(row.get("model_visible", False)),
                sample_seed=_as_int(row.get("sample_seed", 0), "sample_seed"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectError(f"{source} holds an unreadable gold record: {exc}") from exc


@dataclass(frozen=True)
class ReviewRecord:
    """One human decision on an item the engine abstained on (§12)."""

    item_id: str
    code_id: str
    label: int
    coder: str
    timestamp: str
    probability: float | None
    model_visible: bool = True

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"review label must be 0 or 1, got {self.label!r}")

    @classmethod
    def from_row(cls, row: dict[str, object], *, source: str) -> ReviewRecord:
        probability = row.get("probability")
        try:
            return cls(
                item_id=str(row["item_id"]),
                code_id=str(row["code_id"]),
                label=_as_int(row["label"], "label"),
                coder=str(row["coder"]),
                timestamp=str(row["timestamp"]),
                probability=(
                    None if probability is None else _as_float(probability, "probability")
                ),
                model_visible=bool(row.get("model_visible", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectError(f"{source} holds an unreadable review record: {exc}") from exc


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be a number, got {type(value).__name__}")
    return int(value)


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be a number, got {type(value).__name__}")
    return float(value)


@dataclass
class JsonlStore:
    """A tiny append-only JSON Lines store."""

    path: Path

    def append(self, rows: Iterable[dict[str, object]]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
        return written

    def read(self) -> Iterator[dict[str, object]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProjectError(f"{self.path}:{number} is not valid JSON: {exc}") from exc
                yield parsed


@dataclass
class GoldStore:
    """Reads and appends gold judgements."""

    path: Path
    _store: JsonlStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = JsonlStore(self.path)

    def append(self, records: Sequence[GoldRecord]) -> int:
        return self._store.append(asdict(record) for record in records)

    def record(
        self,
        *,
        item_id: str,
        code_id: str,
        label: int,
        coder: str,
        split: str,
        stratum: str,
        inclusion_probability: float,
        sample_seed: int,
    ) -> GoldRecord:
        """Build a gold record. `model_visible` is False by design, not by choice."""
        return GoldRecord(
            item_id=item_id,
            code_id=code_id,
            label=label,
            coder=coder,
            timestamp=_now(),
            split=split,
            stratum=stratum,
            inclusion_probability=inclusion_probability,
            # §6: always false for gold. A coder who has seen the model's answer
            # is no longer an independent second rater, and the whole reliability
            # claim rests on that independence.
            model_visible=False,
            sample_seed=sample_seed,
        )

    def all_records(self) -> list[GoldRecord]:
        return [GoldRecord.from_row(row, source=str(self.path)) for row in self._store.read()]

    def latest(self) -> dict[tuple[str, str, str], GoldRecord]:
        """The most recent judgement per (item, code, coder).

        Append-only means corrections arrive as later lines; this is what makes
        them take effect without losing the history.
        """
        out: dict[tuple[str, str, str], GoldRecord] = {}
        for record in self.all_records():
            out[record.key] = record
        return out

    def coders(self) -> tuple[str, ...]:
        return tuple(sorted({r.coder for r in self.all_records()}))

    def primary_coder(self) -> str | None:
        """The coder who did the most work; treated as coder 1 for reporting."""
        counts: dict[str, int] = {}
        for record in self.all_records():
            counts[record.coder] = counts.get(record.coder, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda coder: (counts[coder], coder))

    def coded_pairs(self, coder: str | None = None) -> set[tuple[str, str]]:
        """(item_id, code_id) pairs already judged, so `gold` can resume."""
        return {
            (r.item_id, r.code_id)
            for r in self.latest().values()
            if coder is None or r.coder == coder
        }

    def labels_for(
        self, code_id: str, *, split: str | None = None, coder: str | None = None
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        """(item ids, labels, IPW weights) for one code.

        Returns the arrays every estimator in `stats/` expects, already weighted
        by the design that drew each item (§8.5).
        """
        chosen = [
            record
            for record in self.latest().values()
            if record.code_id == code_id
            and (split is None or record.split == split)
            and (coder is None or record.coder == coder)
        ]
        chosen.sort(key=lambda r: r.item_id)
        ids = tuple(r.item_id for r in chosen)
        labels = np.array([float(r.label) for r in chosen], dtype=np.float64)
        weights = np.array([r.weight for r in chosen], dtype=np.float64)
        return ids, labels, weights

    def double_coded(self, code_id: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Paired labels from two coders, for the human-human ceiling (§12)."""
        by_item: dict[str, dict[str, int]] = {}
        for record in self.latest().values():
            if record.code_id == code_id:
                by_item.setdefault(record.item_id, {})[record.coder] = record.label
        shared = sorted(item for item, judgements in by_item.items() if len(judgements) >= 2)
        if not shared:
            return np.array([]), np.array([]), ()
        first_coder, second_coder = sorted({coder for item in shared for coder in by_item[item]})[
            :2
        ]
        usable = [i for i in shared if first_coder in by_item[i] and second_coder in by_item[i]]
        return (
            np.array([float(by_item[i][first_coder]) for i in usable]),
            np.array([float(by_item[i][second_coder]) for i in usable]),
            tuple(usable),
        )

    def purge(self) -> bool:
        """Delete the gold file. Personal data at rest (§6)."""
        if self.path.exists():
            self.path.unlink()
            return True
        return False


@dataclass
class ReviewStore:
    """Human decisions on abstained items."""

    path: Path
    _store: JsonlStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = JsonlStore(self.path)

    def append(self, records: Sequence[ReviewRecord]) -> int:
        return self._store.append(asdict(record) for record in records)

    def record(
        self, *, item_id: str, code_id: str, label: int, coder: str, probability: float | None
    ) -> ReviewRecord:
        # Showing the probability is fine and useful here (§12): the researcher
        # is adjudicating a case the machine already declined, not rating blind.
        return ReviewRecord(
            item_id=item_id,
            code_id=code_id,
            label=label,
            coder=coder,
            timestamp=_now(),
            probability=probability,
            model_visible=True,
        )

    def latest(self) -> dict[tuple[str, str], ReviewRecord]:
        out: dict[tuple[str, str], ReviewRecord] = {}
        for row in self._store.read():
            record = ReviewRecord.from_row(row, source=str(self.path))
            out[(record.item_id, record.code_id)] = record
        return out

    def purge(self) -> bool:
        if self.path.exists():
            self.path.unlink()
            return True
        return False
